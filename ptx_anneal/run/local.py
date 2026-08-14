# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""``LocalRunner`` — score each candidate in a throwaway subprocess on the current host.

The candidate worker is ``python -m <target_worker> <task_dir> <acf|NONE> <warmup> <rep>`` (for the
``ptx`` target, :mod:`ptx_anneal.target.ptx`), which prints exactly one line ``MS <float>`` on
success or ``INVALID`` otherwise. Running it out-of-process means a candidate that wedges or crashes
the GPU can only kill that child — the parent reaps it at ``timeout`` and scores it :data:`INVALID`,
so the search never hangs the driver.

The worker interpreter defaults to the current ``sys.executable`` and is overridable via
``PTX_ANNEAL_WORKER_PYTHON`` (used when the search engine lives in a different environment than the
GPU/benchmark stack).
"""

from __future__ import annotations

import os
import subprocess
import sys

from .base import INVALID, Runner

# Map each target to the module run as its isolated per-candidate worker.
_WORKER_MODULE = {"ptx": "ptx_anneal.target.ptx"}


class LocalRunner(Runner):
    def __init__(self, worker_python: str | None = None):
        self.worker_python = worker_python or os.environ.get("PTX_ANNEAL_WORKER_PYTHON") or sys.executable

    def score(
        self,
        target,
        task,
        artifact_path: str | None,
        *,
        timeout: float | None = None,
        warmup: int = 100,
        rep: int = 200,
    ) -> float:
        module = _WORKER_MODULE.get(target.name)
        if module is None:
            raise ValueError(f"no local worker registered for target {target.name!r}")
        cmd = [self.worker_python, "-m", module, task.dir, artifact_path or "NONE", str(warmup), str(rep)]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return INVALID  # wedged/too-slow candidate: reaped, scored invalid
        for line in out.stdout.splitlines():
            if line.startswith("MS "):
                try:
                    return float(line.split()[1])
                except (IndexError, ValueError):
                    return INVALID
        return INVALID
