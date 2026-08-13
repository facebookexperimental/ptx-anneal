# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""``CudaPyRunner`` -- a torch-free Runner that scores candidates via cuda-python + numpy.

It spawns the torch-free worker :mod:`ptx_anneal.run.cudapy_worker` in a throwaway base-interpreter
subprocess, so candidate scoring runs in environments without a (working) torch install -- the only
moving parts are ptxas (assemble), the CUDA driver (launch), and numpy (alloc/compare). This is the
counterpart to :class:`ptx_anneal.run.local.LocalRunner` (which uses the torch-based
``ptx_anneal.target.ptx`` worker and is the right choice for TMA/strided kernels).

This module imports only stdlib, so it can be imported by an orchestrator running in an environment
that has no cuda-python/numpy/torch; those are needed only in the worker interpreter, selected via
``PTX_ANNEAL_WORKER_PYTHON``.
"""

from __future__ import annotations

import os
import subprocess
import sys

from .base import INVALID, Runner

#: the module run as the per-candidate worker (torch-free; cuda-python + numpy).
_WORKER_MODULE = "ptx_anneal.run.cudapy_worker"


class CudaPyRunner(Runner):
    def __init__(self, worker_python: str | None = None):
        self.worker_python = worker_python or os.environ.get("PTX_ANNEAL_WORKER_PYTHON") or sys.executable

    def score(self, target, task, artifact_path, *, timeout=None, warmup=100, rep=200) -> float:
        """Score one candidate by running the worker in a throwaway subprocess.

        Worker contract (see :mod:`ptx_anneal.run.cudapy_worker`): it prints exactly one line
        ``MS <float>`` to stdout on success, or ``INVALID``; all diagnostics go to stderr. We
        communicate over stdout *text* because the worker runs in a separate interpreter (possibly a
        different env than the orchestrator) -- there is no Python return value to pass back. A
        candidate that wedges/crashes/times out simply yields no ``MS`` line and scores
        :data:`INVALID`, so a bad ACF can never take down the search.

        Parsing is deliberately forgiving:
          - ``splitlines()`` + ``startswith("MS ")``: stdout is not guaranteed to be *only* the
            result -- import banners or driver chatter can leak onto stdout -- so we scan for the
            result line and ignore the rest (no ``MS`` line at all -> INVALID).
          - ``float(line.split()[1])``: the ms value crosses the process boundary as a string;
            deserialize the second token for the factory's numeric compare. A missing/unparseable
            token (IndexError/ValueError) -> INVALID rather than a crash.
        """
        cmd = [self.worker_python, "-m", _WORKER_MODULE, task.dir, artifact_path or "NONE", str(warmup), str(rep)]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return INVALID  # wedged / too-slow candidate: reaped at timeout, scored INVALID
        for line in out.stdout.splitlines():
            if line.startswith("MS "):
                try:
                    return float(line.split()[1])
                except (IndexError, ValueError):
                    return INVALID
        return INVALID
