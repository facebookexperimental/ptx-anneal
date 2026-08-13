# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The ``Store`` ABC — persistence for tuning artifacts (ACFs).

An artifact is keyed by ``(target, arch, ir_hash)`` and carries an opaque body (the
ACF bytes) plus a provenance sidecar (engine, versions, search-time win). A
:class:`Store` is the only thing the factory and the consumer share; the local
filesystem implementation ships here, remote ones are injected out-of-tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Store(ABC):
    """A content-addressed artifact store keyed by ``(target, arch, ir_hash)``."""

    @abstractmethod
    def has(self, target: str, arch: str, ir_hash: str, toolchain_version: str = "") -> bool:
        """Whether an artifact exists for this key."""

    @abstractmethod
    def read(self, target: str, arch: str, ir_hash: str, toolchain_version: str = "") -> bytes | None:
        """The artifact bytes, or ``None`` on a miss."""

    @abstractmethod
    def read_meta(self, target: str, arch: str, ir_hash: str, toolchain_version: str = "") -> dict | None:
        """The provenance sidecar for an artifact, or ``None`` if absent."""

    @abstractmethod
    def write(
        self, target: str, arch: str, ir_hash: str, data: bytes,
        meta: dict | None = None, toolchain_version: str = "",
    ) -> str:
        """Admit an artifact (and optional sidecar). Returns an identifier (e.g. a path).

        ``toolchain_version`` (e.g. the ptxas version) scopes the artifact: the kernel identity stays
        the version-independent ``ir_hash``, but the ACF itself is only valid for the ptxas that
        produced it, so it is stored under a version-tagged path when given."""

    @abstractmethod
    def list(self, target: str | None = None, arch: str | None = None) -> list[dict]:
        """Enumerate stored artifacts as dicts, optionally filtered by target/arch."""
