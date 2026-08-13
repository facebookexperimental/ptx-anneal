# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The ``SearchBackend`` ABC + the backend registry.

A backend proposes candidate artifacts and scores them through a caller-supplied ``objective``
(``objective(artifact_bytes) -> ms`` or ``INVALID``); it returns the best candidate it found. The
engine itself (e.g. NVIDIA CompileIQ) is bring-your-own — registered out-of-tree under the
``ptx_anneal.search_backends`` entry-point group and resolved by :func:`load_backend`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib import metadata

ENTRY_POINT_GROUP = "ptx_anneal.search_backends"


@dataclass
class SearchResult:
    """The outcome of a search: the best candidate (if any) and provenance."""

    artifact: bytes | None  # the best candidate's bytes, or None if nothing was found
    score: float  # the best candidate's ms (INVALID/inf if none)
    evaluated: int = 0  # how many candidates were scored
    meta: dict = field(default_factory=dict)  # engine-specific provenance


class SearchBackend(ABC):
    #: registry name (matches the entry-point key)
    name: str = ""

    @abstractmethod
    def search(self, task, objective, *, config: dict | None = None) -> SearchResult:
        """Propose + score candidates via ``objective(artifact_bytes) -> ms|INVALID`` and return the
        best :class:`SearchResult`. ``config`` carries engine knobs (depth, pool, per-candidate
        timeout, ...)."""


def _builtin_backends() -> dict[str, type[SearchBackend]]:
    from .baseline import BaselineSearch

    return {BaselineSearch.name: BaselineSearch}


def available_backends() -> list[str]:
    """All resolvable backend names: built-ins + out-of-tree entry-point plugins."""
    names = set(_builtin_backends())
    try:
        for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
            names.add(ep.name)
    except Exception:  # noqa: BLE001, S110 - a third-party plugin with broken metadata must not
        pass          # take out discovery; the built-ins below are still returned.
    return sorted(names)


def _load_import_path(spec: str) -> SearchBackend:
    """Resolve a ``"module.path:ClassName"`` backend spec by import + instantiate.

    This is the escape hatch for a backend that is **not** pip-installed (so it has no entry point) —
    e.g. an engine plugin available only on ``PYTHONPATH``. Keeps the engine out-of-tree while still
    letting the OSS CLI select it by spec, with no hard-coded reference to it in this package."""
    import importlib

    mod_name, _, cls_name = spec.partition(":")
    try:
        cls = getattr(importlib.import_module(mod_name), cls_name)
    except Exception as e:
        raise RuntimeError(f"failed loading search backend from import path {spec!r}: {e}") from e
    return cls()


def load_backend(name: str) -> SearchBackend:
    """Resolve a backend and instantiate it. ``name`` may be a built-in name, an entry-point plugin
    name, or a ``"module.path:ClassName"`` import path (for a backend on ``PYTHONPATH`` that isn't
    pip-installed).

    Raises a clear error naming the resolvable backends when ``name`` cannot be resolved — a
    bring-your-own engine that isn't provisioned must fail loudly, never silently."""
    builtins = _builtin_backends()
    if name in builtins:
        return builtins[name]()
    if ":" in name:
        return _load_import_path(name)
    try:
        for ep in metadata.entry_points(group=ENTRY_POINT_GROUP):
            if ep.name == name:
                return ep.load()()
    except Exception as e:  # pragma: no cover - environment-dependent
        raise RuntimeError(f"failed loading search backend {name!r}: {e}") from e
    raise RuntimeError(
        f"unknown search backend {name!r}. Available: {available_backends()}. "
        f"Engines (e.g. CompileIQ) are bring-your-own — install the engine's plugin so it registers "
        f"under the {ENTRY_POINT_GROUP!r} entry-point group, or use the built-in 'baseline' backend."
    )
