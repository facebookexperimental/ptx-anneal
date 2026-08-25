# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The search-problem fingerprint: what BOTH channels must agree on, proven rather than assumed.

ptx-anneal reaches the same kernel two ways -- the shipped torch-free PTX-direct channel
(``ptx_anneal --task``) and the on-demand triton-native runway (``e2e/triton_native/``). They are
only a *differential* if they pose the identical search problem; otherwise they are two products
that occasionally disagree, and a disagreement tells you nothing.

So each channel emits this fingerprint at the end of a run and the two are diffed. It deliberately
covers every input that can move a result:

* **kernel identity** -- name, shapes, dtypes, and the pinned config (autotune would make "the
  kernel" ambiguous, which is why the runway refuses it);
* **ptxas** -- path, version *and flags*. Flags matter as much as the version: the same ptxas with a
  different ``-O`` or ``--gpu-name`` is a different compiler for this purpose;
* **search space** -- the *resolved* tag/version/variant/sha, never the requested one. The catalog
  default is ``latest``, so "we both asked for latest" is not agreement;
* **engine** -- name, version and the whole budget (generations, pool, timeout, seed);
* **objective** -- the metric, the timing method *and its budget*, and the correctness oracle *and
  its tolerance*. A channel that benchmarks with a different warmup, or admits with a looser
  tolerance, is answering a different question;
* **environment** -- GPU model, CUDA version, driver and clock state.

Execution mechanism and timing method are pluggable *knobs*, not a fork: the channels may differ, as
long as the fingerprint records that they do. What is forbidden is an unrecorded difference.

Pure stdlib on purpose -- this ships in the wheel, and the whole point is that the torch-free
channel can emit it with nothing but Python and a driver.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess

# The nvidia-smi fields that make up "environment". Requested in one call so the snapshot is
# coherent (two calls could straddle a clock change, which is precisely the thing being recorded).
_SMI_FIELDS = (
    "name",
    "driver_version",
    "clocks.sm",
    "clocks.max.sm",
    "clocks.applications.graphics",
    "persistence_mode",
)

# Renamed in newer drivers; ask for the modern spelling first and fall back, so the field is present
# on both rather than silently dropping out of the fingerprint on one of them.
_SMI_EVENT_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")


def _smi(query: str, index: int | str) -> list[str] | None:
    """One ``nvidia-smi --query-gpu`` call, or ``None`` if it is unavailable/unhappy.

    Best-effort by design: a missing nvidia-smi is a *recordable state* ("environment unknown"), not
    a reason to fail a tuning run that has already completed.
    """
    if not shutil.which("nvidia-smi"):
        return None
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
        "-i",
        str(index),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    first = out.stdout.strip().splitlines()[0]
    return [c.strip() for c in first.split(",")]


def _driver_cuda_version() -> str | None:
    """The CUDA version the *driver* reports (``nvidia-smi -q``), not the toolkit's.

    Distinct from ``ptxas --version``: ptxas decides what the ACF means, the driver decides whether
    the resulting cubin will load. Both belong in the fingerprint.
    """
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.stdout.splitlines():
        if "CUDA Version" in line and ":" in line:
            return line.split(":", 1)[1].strip()
    return None


def gpu_env(index: int | str | None = None) -> dict:
    """GPU model, driver, CUDA version and clock state.

    ``clock state`` is here because NVIDIA's own example calls locking the clocks "crucial": an
    unlocked GPU turns every measurement into a different question, and two channels measured under
    different clock regimes can disagree for no interesting reason at all.
    """
    if index is None:
        # Honour the same device selection the run itself used, so the fingerprint describes the GPU
        # that was actually measured rather than device 0.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
        index = visible if visible else 0

    env: dict = {
        "gpu": None,
        "driver_version": None,
        "cuda_version": _driver_cuda_version(),
        "sm_clock_mhz": None,
        "max_sm_clock_mhz": None,
        "applications_clock_mhz": None,
        "persistence_mode": None,
        "clock_event_reasons": None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    row = _smi(",".join(_SMI_FIELDS), index)
    if row and len(row) == len(_SMI_FIELDS):
        name, driver, sm, sm_max, app, persist = row
        env.update(
            {
                "gpu": name,
                "driver_version": driver,
                "sm_clock_mhz": _int_or_none(sm),
                "max_sm_clock_mhz": _int_or_none(sm_max),
                "applications_clock_mhz": _int_or_none(app),
                "persistence_mode": persist,
            }
        )
    for field in _SMI_EVENT_FIELDS:
        row = _smi(field, index)
        if row:
            env["clock_event_reasons"] = row[0]
            break
    return env


def _int_or_none(v: str) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def ptxas_flags() -> list[str]:
    """The ptxas flags in force, as the fingerprint sees them.

    The ACF itself arrives via ``--apply-controls`` and is *not* a flag for this purpose -- it is the
    candidate under test, and including it would make every candidate a different search problem.
    Everything else the toolchain injects (``PTXAS_OPTIONS`` in the triton-native channel) is a real
    difference between two runs and must show up here.
    """
    raw = os.environ.get("PTXAS_OPTIONS", "")
    return [f for f in raw.split() if f and not f.startswith("--apply-controls")]


def build(
    *,
    channel: str,
    kernel: dict,
    ptxas: dict,
    search_space: dict | None,
    engine: dict,
    objective: dict,
    env: dict | None = None,
    frontend: dict | None = None,
) -> dict:
    """Assemble a fingerprint. Every argument is passed in *resolved* -- nothing is inferred here.

    That is deliberate: the resolution happens where the knowledge is (the adapter knows the search
    space, the runway knows the objective), and this module's only job is to put it in one canonical
    shape so two channels can be diffed line by line.

    ``frontend`` is the Triton that compiled the kernel, and it matters in exactly one channel. The
    triton-native channel compiles the kernel *during* the search, so its frontend (specifically
    **fbtriton** -- ``PTXAS_OPTIONS`` is its knob, not upstream Triton's) is a live input to every
    candidate. The PTX-direct channel tunes a frozen ``kernel.ptx`` and has no frontend at tune time,
    so it records ``null``: the frontend mattered when the task was *collected*, not here. Null is
    the honest answer, not a missing one, which is why the key is always present.
    """
    fp = {
        "channel": channel,
        "kernel": kernel,
        "frontend": frontend or {},
        "ptxas": ptxas,
        "search_space": search_space or {},
        "engine": engine,
        "objective": objective,
        "env": env if env is not None else gpu_env(),
    }
    fp["digest"] = digest(fp)
    return fp


def canonical(fp: dict) -> str:
    """Canonical JSON: sorted keys, no incidental whitespace. Two equal problems must render byte-identical."""
    return json.dumps({k: v for k, v in fp.items() if k != "digest"}, sort_keys=True, separators=(",", ":"))


def digest(fp: dict) -> str:
    """A short stable hash of the whole problem. Equal digests => identical search problem."""
    return hashlib.sha256(canonical(fp).encode()).hexdigest()[:16]


def render(fp: dict) -> str:
    """Human-readable *and* diffable: a header carrying the digest, then indented canonical JSON.

    Printed by both channels. ``diff <(chanA) <(chanB)`` is the intended workflow, so the body is
    sorted and one-key-per-line rather than compact.
    """
    body = json.dumps({k: v for k, v in fp.items() if k != "digest"}, sort_keys=True, indent=2)
    return f"== search-problem fingerprint (digest={fp.get('digest', '?')}) ==\n{body}"
