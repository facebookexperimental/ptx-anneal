# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The ACF tuning-task schema (``spec_version`` 1).

A *task* is the unit of work a frontend (e.g. fbtriton's collection hook) hands to
the factory. It is **source-free** — the factory never recompiles from Python — and
lives in a directory containing exactly two files:

    <task_dir>/kernel.ptx   the fixed IR the artifact is tuned for
    <task_dir>/spec.json    the launch description defined by this module

``spec.json`` schema (v1)::

    {
      "spec_version": 1,                 # optional; absent == 1 (legacy tasks)
      "entry":  "add_kernel",            # the PTX .visible .entry name
      "arch":   "sm_100a",               # the PTX .target
      "shared": 0,                       # dynamic shared-memory bytes
      "block":  [128, 1, 1],             # CTA dims (threads)
      "grid":   [8, 1, 1],               # launch grid (CTAs)
      "ptxas":  "",                      # ptxas path (filled at run; never hardcode)
      "tensors": [{"shape", "dtype", "strides"}, ...],
      "args":    [{"t": i} | {"i32"|"i64"|"f32": v} | {"tma": j} | {"null": true}, ...],
      "tensordescs": [...],              # optional; host-TMA descriptors
      "ir_hash": "<sha256>",             # content key; alias "ptx_sha256" (legacy)
      "kernel_name": "...", "fn_name": "..."   # provenance (optional)
    }

The store key is ``(target, arch, ir_hash)``. For the ``ptx`` target ``ir_hash`` is
``sha256`` of the normalized PTX (see :mod:`ptx_anneal.target.ptx`), so one artifact
transfers across runtime shapes of the same kernel. ``"args"`` is the final
kernel-param order (post constexpr / ``equal_to_1`` specialization), including the two
trailing null scratch params.

The loader accepts specs that omit ``spec_version`` (treated as v1) so tasks already
collected by older frontends keep working.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

SPEC_VERSION = 1

# Interop env var (external contract — shared with the frontend; do NOT rename).
TASK_DIR_ENV = "COMPILE_IQ_TASK_DIR"

# Fields every v1 spec must carry to describe a launch.
REQUIRED_FIELDS = ("entry", "arch", "shared", "block", "grid", "tensors", "args")


class TaskError(ValueError):
    """Raised when a task directory or its spec.json is missing/malformed."""


def default_task_root() -> str:
    """Where tasks live by default (overridable via ``COMPILE_IQ_TASK_DIR``)."""
    return os.environ.get(TASK_DIR_ENV, os.path.expanduser("~/.compile_iq/tasks"))


def spec_ir_hash(spec: dict) -> str | None:
    """The content key from a spec, accepting the legacy ``ptx_sha256`` alias."""
    return spec.get("ir_hash") or spec.get("ptx_sha256")


def validate_spec(spec: dict) -> None:
    """Raise :class:`TaskError` unless ``spec`` is a well-formed v1 launch spec."""
    if not isinstance(spec, dict):
        raise TaskError("spec.json must be a JSON object")
    version = int(spec.get("spec_version", 1))
    if version != SPEC_VERSION:
        raise TaskError(f"unsupported spec_version {version} (this build supports {SPEC_VERSION})")
    missing = [f for f in REQUIRED_FIELDS if f not in spec]
    if missing:
        raise TaskError(f"spec.json missing required field(s): {missing}")
    if not spec_ir_hash(spec):
        raise TaskError("spec.json missing ir_hash (or legacy ptx_sha256)")
    if not isinstance(spec["tensors"], list) or not isinstance(spec["args"], list):
        raise TaskError("spec.json 'tensors' and 'args' must be lists")
    for axis in ("block", "grid"):
        v = spec[axis]
        if not isinstance(v, list) or not v:
            raise TaskError(f"spec.json '{axis}' must be a non-empty list")


@dataclass
class Task:
    """A loaded tuning task: its directory, fixed IR, and validated spec."""

    dir: str
    ir: str  # the fixed kernel source IR (PTX for the ``ptx`` target)
    spec: dict

    @property
    def entry(self) -> str:
        return self.spec["entry"]

    @property
    def arch(self) -> str:
        return self.spec["arch"]

    @property
    def ir_hash(self) -> str:
        h = spec_ir_hash(self.spec)
        assert h is not None  # guaranteed by validate_spec
        return h

    @property
    def spec_version(self) -> int:
        return int(self.spec.get("spec_version", 1))


def load(task_dir: str) -> Task:
    """Load and validate a task directory (``kernel.ptx`` + ``spec.json``)."""
    spec_path = os.path.join(task_dir, "spec.json")
    ir_path = os.path.join(task_dir, "kernel.ptx")
    if not os.path.isfile(spec_path):
        raise TaskError(f"no spec.json in task dir: {task_dir}")
    if not os.path.isfile(ir_path):
        raise TaskError(f"no kernel.ptx in task dir: {task_dir}")
    try:
        with open(spec_path) as f:
            spec = json.load(f)
    except json.JSONDecodeError as e:
        raise TaskError(f"spec.json is not valid JSON: {e}") from e
    with open(ir_path) as f:
        ir = f.read()
    validate_spec(spec)
    return Task(dir=task_dir, ir=ir, spec=spec)
