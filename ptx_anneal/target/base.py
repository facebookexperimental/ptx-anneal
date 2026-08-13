# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The ``Target`` ABC — the per-IR-kind backend.

A target abstracts everything vendor/IR-specific: how to hash the IR into a store key,
how to read its arch, and how to *score one candidate* (assemble the fixed IR with the
candidate artifact applied, launch it, validate numerical self-consistency against the
untuned baseline, and time it). The factory and runner stay target-agnostic and call
through this interface; v1 ships only the ``ptx`` target.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import metadata

ENTRY_POINT_GROUP = "ptx_anneal.targets"


class ScoreInvalid(Exception):
    """A candidate failed to assemble/launch or diverged from the baseline → not a valid time."""


class Target(ABC):
    #: registry name (matches the ``ptx_anneal.targets`` entry-point key)
    name: str = ""

    @abstractmethod
    def ir_hash(self, ir: str) -> str:
        """The content key for this IR (store key dimension, with arch).

        The full store key is ``(target, arch, ir_hash)`` -- launch-time args (grid, block, tensor
        shapes) are intentionally excluded. This is correct: those are runtime params to the launch,
        not inputs to assembly, so the cubin is a pure function of ``(IR, arch, artifact)``. The only
        caveat is performance, not correctness: an artifact is admitted after a benchmark at one launch
        config, so reusing it across very different grid/block/shapes is an unvalidated perf bet.
        TODO: revisit whether the key (or the consume-time admission gate) should factor in a coarse
        launch profile if tuned artifacts turn out not to transfer across launch configs.
        """

    @abstractmethod
    def arch(self, ir: str) -> str:
        """The hardware arch the IR targets."""

    @abstractmethod
    def score_one(self, task, artifact_path: str | None, *, warmup: int = 100, rep: int = 200) -> float:
        """Assemble + launch + validate the IR with ``artifact_path`` applied (``None`` = baseline);
        return the launch time in ms (lower is better). Raise :class:`ScoreInvalid` if the candidate
        cannot be assembled/launched or diverges from the baseline beyond tolerance.

        This runs the actual GPU work and is intended to be invoked inside a :class:`Runner`'s
        isolation boundary (e.g. a throwaway subprocess), so a candidate that wedges the device can
        only take down that boundary.
        """


def _builtin_targets() -> dict[str, type[Target]]:
    from .ptx import PtxTarget

    return {PtxTarget.name: PtxTarget}


def available_targets() -> list[str]:
    """All resolvable target names: built-ins + out-of-tree entry-point plugins."""
    names = set(_builtin_targets())
    try:
        for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
            names.add(ep.name)
    except Exception:  # noqa: BLE001, S110 - a third-party plugin with broken metadata must not
        pass  # take out discovery; the built-ins below are still returned.
    return sorted(names)


def load_target(name: str) -> Target:
    """Resolve a target by name (built-in or entry-point plugin) and instantiate it."""
    builtins = _builtin_targets()
    if name in builtins:
        return builtins[name]()
    try:
        for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
            if ep.name == name:
                return ep.load()()
    except Exception as e:  # pragma: no cover - environment-dependent
        raise RuntimeError(f"failed loading target {name!r}: {e}") from e
    raise RuntimeError(f"unknown target {name!r}. Available: {available_targets()}")
