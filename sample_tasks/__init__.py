# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unified pre-captured ACF task fixtures for ptx-anneal tests (torch-free).

Each subdirectory is one task: ``kernel.ptx`` + ``spec.json`` (no source), the
version-agnostic input the factory replays. These are plain data, so the factory
unit tests and the tuning-only e2e can use them without torch/triton.

``discover()`` returns ``{name: path}`` for every task dir here.
"""

from __future__ import annotations

import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def _tasks_in(root: str) -> dict[str, str]:
    if not os.path.isdir(root):
        return {}
    out = {}
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        if os.path.isfile(os.path.join(d, "spec.json")) and os.path.isfile(os.path.join(d, "kernel.ptx")):
            out[name] = d
    return out


def discover() -> dict[str, str]:
    """Map task name -> task dir."""
    return _tasks_in(_HERE)


def path(name: str) -> str:
    """Absolute path to a named task dir; raises KeyError if unknown."""
    tasks = discover()
    if name not in tasks:
        raise KeyError(f"unknown sample task {name!r}; have {sorted(tasks)}")
    return tasks[name]
