# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the local artifact store (ptx_anneal.store.LocalStore)."""

import os

from ptx_anneal.store import LocalStore, default_store_root

ARCH = "sm_100a"
IR_HASH = "6cb1f53d5136f5b8e3e3c25727534f6d006f6564d26b0e0b51cfd4826e254000"
ACF = b"\x00\x01\x02fake-acf-bytes"


def test_write_read_roundtrip(tmp_path):
    s = LocalStore(str(tmp_path))
    assert not s.has("ptx", ARCH, IR_HASH)
    assert s.read("ptx", ARCH, IR_HASH) is None

    p = s.write("ptx", ARCH, IR_HASH, ACF, meta={"engine": "baseline", "best_ms": 1.23})
    assert s.has("ptx", ARCH, IR_HASH)
    assert s.read("ptx", ARCH, IR_HASH) == ACF

    meta = s.read_meta("ptx", ARCH, IR_HASH)
    assert meta is not None
    assert meta["target"] == "ptx" and meta["arch"] == ARCH and meta["ir_hash"] == IR_HASH
    assert meta["engine"] == "baseline"
    # path layout must be the frontend-compatible flat <arch>/<hash>.acf
    assert p == os.path.join(str(tmp_path), ARCH, f"{IR_HASH}.acf")


def test_list(tmp_path):
    s = LocalStore(str(tmp_path))
    assert s.list() == []
    s.write("ptx", ARCH, IR_HASH, ACF, meta={"engine": "baseline"})
    entries = s.list()
    assert len(entries) == 1
    e = entries[0]
    assert e["arch"] == ARCH and e["ir_hash"] == IR_HASH and e["target"] == "ptx"
    # filters
    assert s.list(arch="sm_90a") == []
    assert len(s.list(arch=ARCH)) == 1
    assert len(s.list(target="ptx")) == 1
    assert s.list(target="gcn") == []


def test_cubin_cache_tagged_by_toolchain(tmp_path):
    s = LocalStore(str(tmp_path))
    assert s.read_cubin(ARCH, IR_HASH, "13.3") is None
    s.write_cubin(ARCH, IR_HASH, "13.3", b"cubin-A")
    assert s.read_cubin(ARCH, IR_HASH, "13.3") == b"cubin-A"
    # a different toolchain version must not serve the same cubin
    assert s.read_cubin(ARCH, IR_HASH, "13.4") is None
    # the cubin file is not mistaken for an artifact by list()
    assert all(not e["path"].endswith(".acf.cubin") for e in s.list())


def test_acf_is_tagged_by_toolchain_version(tmp_path):
    # The CLI always passes toolchain_version, so the shipped layout is <hash>.<ptxas>.acf. ACFs from
    # different ptxas versions must coexist rather than overwrite each other.
    s = LocalStore(str(tmp_path))
    p133 = s.write("ptx", ARCH, IR_HASH, b"acf-133", meta={}, toolchain_version="13.3")
    p134 = s.write("ptx", ARCH, IR_HASH, b"acf-134", meta={}, toolchain_version="13.4")
    assert p133 == os.path.join(str(tmp_path), ARCH, f"{IR_HASH}.13.3.acf")
    assert p133 != p134
    assert s.read("ptx", ARCH, IR_HASH, toolchain_version="13.3") == b"acf-133"
    assert s.read("ptx", ARCH, IR_HASH, toolchain_version="13.4") == b"acf-134"
    # an untagged lookup must not silently serve a tagged artifact
    assert s.read("ptx", ARCH, IR_HASH) is None


def test_sidecar_carries_full_provenance(tmp_path):
    # The search space and engine build are tuning *inputs* that the store key does NOT distinguish
    # (and the catalog defaults to "latest"), so the sidecar is the only record of which produced a
    # given ACF. Dropping these fields would make an admitted ACF unexplainable.
    s = LocalStore(str(tmp_path))
    meta = {
        "ptxas_version": "13.3",
        "engine": "compileiq",
        "engine_info": {"name": "compileiq", "version": "1.0.0.dev1", "path": "/sp/compileiq", "python": "/py"},
        "search_space": {"resolved_tag": "search-spaces-2026.05.22", "sha256": "deadbeef", "source": "cache"},
    }
    s.write("ptx", ARCH, IR_HASH, ACF, meta=meta, toolchain_version="13.3")
    got = s.read_meta("ptx", ARCH, IR_HASH, toolchain_version="13.3")
    assert got["ptxas_version"] == "13.3"
    assert got["engine_info"]["version"] == "1.0.0.dev1"
    assert got["search_space"]["resolved_tag"] == "search-spaces-2026.05.22"
    assert got["search_space"]["sha256"] == "deadbeef"


def test_default_store_root_env(monkeypatch):
    monkeypatch.setenv("COMPILE_IQ_STORE", "/tmp/ciq_store_xyz")
    assert default_store_root() == "/tmp/ciq_store_xyz"
    monkeypatch.delenv("COMPILE_IQ_STORE", raising=False)
    assert default_store_root().endswith("/.compile_iq/store")
