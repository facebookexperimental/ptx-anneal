# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Toolchain detection for ``ptx-anneal doctor``.

Reports whether the three things a real search needs are present:
1. ``ptxas`` >= the minimum version (applying an ACF via ``--apply-controls`` is a ptxas 13.3+ GA
   feature) — needed to *assemble* tuning artifacts;
2. an optimization engine (bring-your-own; e.g. NVIDIA CompileIQ) registered as a search backend;
3. a GPU + driver stack (cuda-python / torch) — needed to *launch* candidates.

The baseline backend + the store/ABC layer work without any of these; ``tune`` with a real engine
needs all three. ``doctor`` only inspects and prints next steps — it never installs anything.

Engine discovery is automatic: an installed engine plugin registers under the
``ptx_anneal.search_backends`` entry-point group and is found here with no env var to point at it.
The only env that may matter is ``TRITON_PTXAS_BLACKWELL_PATH`` (when ptxas isn't on ``PATH``); any
engine-internal config (its search space, etc.) is the engine's own concern, not ptx-anneal's.

Caveat: ``tune`` drives the engine through a standalone *adapter* subprocess (see ``cli.py``), which
does not need an entry-point plugin. So "no engine registered" here does not mean ``tune`` will fail
with the bundled adapter -- it only reports the plugin registry.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

from .search.base import available_backends

# ACF apply (`ptxas --apply-controls`) is GA from this ptxas version on. Not overridable: an older
# ptxas cannot apply an ACF at all, so "relaxing" the floor only turns a clear error into a silent
# no-op (and, on the frontend side, a store MISS -- see the PTX-ISA note in min_ptxas()).
MIN_PTXAS = "13.3"


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(x) for x in re.findall(r"\d+", v)[:3])


def find_ptxas() -> str | None:
    """The ptxas ptx-anneal would use: explicit env first, then PATH."""
    for env in ("TRITON_PTXAS_BLACKWELL_PATH", "TRITON_PTXAS_PATH"):
        p = os.environ.get(env)
        if p and os.path.exists(p):
            return p
    return shutil.which("ptxas")


def ptxas_version(path: str) -> str | None:
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return None
    m = re.search(r"release\s+([0-9]+\.[0-9]+)", out)
    if m:
        return m.group(1)
    m = re.search(r"V([0-9]+\.[0-9]+\.[0-9]+)", out)
    return m.group(1) if m else None


def min_ptxas() -> str:
    """The required ptxas floor.

    Fixed, by design. Beyond `--apply-controls` being GA only from here on, the frontend derives the
    PTX ISA version it emits from *its* ptxas version, so a task collected under a different ptxas
    hashes to a different store key -- a lower floor would silently produce ACFs nothing can consume.
    """
    return MIN_PTXAS


def gpu_status() -> dict:
    """Whether the launch stack (cuda-python + torch + a visible GPU) is available."""
    status = {"cuda_python": False, "torch": False, "gpu": False, "detail": ""}
    try:
        import cuda.bindings.driver  # noqa: F401

        status["cuda_python"] = True
    except Exception:
        pass
    try:
        import torch

        status["torch"] = True
        status["gpu"] = bool(torch.cuda.is_available())
        if status["gpu"]:
            status["detail"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return status


def report() -> dict:
    """A structured toolchain report (consumed by ``doctor``)."""
    ptxas = find_ptxas()
    ver = ptxas_version(ptxas) if ptxas else None
    need = min_ptxas()
    ptxas_ok = bool(ver) and _version_tuple(ver) >= _version_tuple(need)
    engines = [b for b in available_backends() if b != "baseline"]
    return {
        "ptxas_path": ptxas,
        "ptxas_version": ver,
        "ptxas_min": need,
        "ptxas_ok": ptxas_ok,
        "engines": engines,
        "gpu": gpu_status(),
    }


def format_report(rep: dict) -> str:
    """A human-readable doctor report + the next provisioning step."""
    lines = ["ptx-anneal doctor", "=================="]

    if rep["ptxas_path"]:
        mark = "OK " if rep["ptxas_ok"] else "!! "
        lines.append(
            f"[{mark}] ptxas: {rep['ptxas_path']} (version {rep['ptxas_version'] or '?'}, need >= {rep['ptxas_min']})"
        )
        if not rep["ptxas_ok"]:
            lines.append("       -> install/point at a newer CUDA Toolkit; set TRITON_PTXAS_BLACKWELL_PATH.")
    else:
        lines.append("[!! ] ptxas: not found")
        lines.append("       -> install a CUDA Toolkit and set TRITON_PTXAS_BLACKWELL_PATH (or add ptxas to PATH).")

    if rep["engines"]:
        lines.append(f"[OK ] engine(s): {', '.join(rep['engines'])}")
    else:
        lines.append("[-- ] engine: none registered (only the built-in 'baseline' backend)")
        lines.append("       -> pip-install a bring-your-own engine plugin (e.g. NVIDIA CompileIQ); it self-registers")
        lines.append(
            "          under the 'ptx_anneal.search_backends' entry point and appears here — no env var needed."
        )

    g = rep["gpu"]
    if g["gpu"]:
        lines.append(f"[OK ] gpu: {g['detail']} (cuda-python={g['cuda_python']}, torch={g['torch']})")
    else:
        lines.append(f"[!! ] gpu: not available (cuda-python={g['cuda_python']}, torch={g['torch']})")
        lines.append("       -> 'doctor' and the store/ABCs work without a GPU, but 'tune' needs one.")

    ready = rep["ptxas_ok"] and g["gpu"]
    lines.append("")
    lines.append("baseline flow ready (no engine): " + ("YES" if g["gpu"] else "no — needs a GPU"))
    lines.append("real tuning ready: " + ("YES" if (ready and rep["engines"]) else "no — see the items marked !! / --"))
    return "\n".join(lines)
