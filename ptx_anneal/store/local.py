# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""``LocalStore`` — a filesystem artifact store.

Layout (all files for one kernel share a directory)::

    <root>/<arch>/<ir_hash>/<ir_hash>.<toolchain>.acf        # version-tagged artifact
    <root>/<arch>/<ir_hash>/<ir_hash>.<toolchain>.acf.json   # provenance sidecar
    <root>/<arch>/<ir_hash>/<ir_hash>.<toolchain>.acf.cubin  # optional assembled-cubin cache
    <root>/<arch>/<ir_hash>/<ir_hash>.acf                  # no toolchain version given

``target`` is recorded in the sidecar. Both the ``.acf`` and the assembled-cubin
cache are tagged by toolchain (ptxas) version: an ACF is opaque controls applied at
assemble time and is only valid for the ptxas that produced it (a mismatched ptxas can
reject it), so version-tagging lets one kernel keep a distinct tuned ACF per ptxas
version. The kernel *identity* (``ir_hash``) stays version-independent. When no
toolchain version is given, the untagged ``<ir_hash>.acf`` filename is used.

Reads and listings also support the older flat ``<root>/<arch>/<filename>`` layout,
preferring the nested artifact when both exist. New writes always use the kernel directory.
Consumers constructing paths directly must adopt the nested layout to find new artifacts.

The store stays free of any frontend/engine dependency: it takes a precomputed
``ir_hash`` (the ``ptx`` target computes it) and opaque bytes, nothing more.
"""

from __future__ import annotations

import json
import os
import re

from .base import Store

# Interop env var (external contract — shared with the frontend; do NOT rename).
DEFAULT_STORE_ENV = "COMPILE_IQ_STORE"


def default_store_root() -> str:
    """Where artifacts live by default (overridable via ``COMPILE_IQ_STORE``)."""
    return os.environ.get(DEFAULT_STORE_ENV, os.path.expanduser("~/.compile_iq/store"))


def _toolchain_tag(version: str) -> str:
    """A filesystem-safe, bounded tag for a toolchain (ptxas) version string."""
    return re.sub(r"[^0-9A-Za-z._]", "", version or "")[:32] or "unknown"


class LocalStore(Store):
    """A content-addressed artifact store backed by a local directory tree."""

    def __init__(self, root: str | None = None):
        self.root = root or default_store_root()

    # --- key -> path -----------------------------------------------------------------------------
    def _acf_path(self, arch: str, ir_hash: str, toolchain_version: str = "") -> str:
        name = f"{ir_hash}.{_toolchain_tag(toolchain_version)}.acf" if toolchain_version else f"{ir_hash}.acf"
        return os.path.join(self.root, arch, ir_hash, name)

    def _cubin_path(self, arch: str, ir_hash: str, toolchain_version: str) -> str:
        return os.path.join(self.root, arch, ir_hash, f"{ir_hash}.{_toolchain_tag(toolchain_version)}.acf.cubin")

    @staticmethod
    def _legacy_path(path: str) -> str:
        return os.path.join(os.path.dirname(os.path.dirname(path)), os.path.basename(path))

    def _read_path(self, arch: str, ir_hash: str, toolchain_version: str) -> str:
        path = self._acf_path(arch, ir_hash, toolchain_version)
        return path if os.path.exists(path) else self._legacy_path(path)

    # --- Store ABC -------------------------------------------------------------------------------
    def has(self, target: str, arch: str, ir_hash: str, toolchain_version: str = "") -> bool:
        return os.path.exists(self._read_path(arch, ir_hash, toolchain_version))

    def read(self, target: str, arch: str, ir_hash: str, toolchain_version: str = "") -> bytes | None:
        p = self._read_path(arch, ir_hash, toolchain_version)
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return f.read()

    def read_meta(self, target: str, arch: str, ir_hash: str, toolchain_version: str = "") -> dict | None:
        # Choose the sidecar paired with the selected ACF, never metadata from the other layout.
        p = self._read_path(arch, ir_hash, toolchain_version) + ".json"
        if not os.path.exists(p):
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def write(
        self,
        target: str,
        arch: str,
        ir_hash: str,
        data: bytes,
        meta: dict | None = None,
        toolchain_version: str = "",
    ) -> str:
        p = self._acf_path(arch, ir_hash, toolchain_version)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.tmp.{os.getpid()}"  # write+rename so a concurrent reader never sees a partial file
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, p)
        sidecar = {"target": target, "arch": arch, "ir_hash": ir_hash, **(meta or {})}
        with open(p + ".json", "w") as f:
            json.dump(sidecar, f, indent=2, default=str)
        return p

    def list(self, target: str | None = None, arch: str | None = None) -> list[dict]:
        out: list[dict] = []
        if not os.path.isdir(self.root):
            return out
        for arch_name in sorted(os.listdir(self.root)):
            if arch is not None and arch_name != arch:
                continue
            arch_dir = os.path.join(self.root, arch_name)
            if not os.path.isdir(arch_dir):
                continue
            paths: dict[str, str] = {}
            for name in sorted(os.listdir(arch_dir)):
                path = os.path.join(arch_dir, name)
                if os.path.isdir(path):
                    for fn in sorted(os.listdir(path)):
                        artifact = os.path.join(path, fn)
                        if fn.endswith(".acf") and fn.split(".", 1)[0] == name and os.path.isfile(artifact):
                            paths[fn] = artifact
                elif name.endswith(".acf") and os.path.isfile(path):
                    # Nested artifacts take precedence over a flat copy, regardless of scan order.
                    paths.setdefault(name, path)
            for fn, path in sorted(paths.items()):
                # "<ir_hash>.acf" (untagged) or "<ir_hash>.<toolchain>.acf" (version-tagged);
                # ir_hash is a dot-free sha256 hex, so it's the leading component.
                stem = fn[: -len(".acf")]
                ir_hash, _, toolchain = stem.partition(".")
                meta = self.read_meta(target or "", arch_name, ir_hash, toolchain) or {}
                if target is not None and meta.get("target", target) != target:
                    continue
                out.append(
                    {
                        "target": meta.get("target"),
                        "arch": arch_name,
                        "ir_hash": ir_hash,
                        "toolchain_version": toolchain or None,
                        "path": path,
                        "meta": meta,
                    }
                )
        return out

    # --- assembled-cubin cache (optional; not part of the Store ABC) ------------------------------
    def read_cubin(self, arch: str, ir_hash: str, toolchain_version: str) -> bytes | None:
        p = self._cubin_path(arch, ir_hash, toolchain_version)
        if not os.path.exists(p) and not os.path.exists(self._acf_path(arch, ir_hash, toolchain_version)):
            # A flat cache belongs to the flat ACF; never pair it with a newly written nested ACF.
            p = self._legacy_path(p)
        if not os.path.exists(p):
            return None
        with open(p, "rb") as f:
            return f.read()

    def write_cubin(self, arch: str, ir_hash: str, toolchain_version: str, data: bytes) -> str:
        p = self._cubin_path(arch, ir_hash, toolchain_version)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.tmp.{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, p)
        return p
