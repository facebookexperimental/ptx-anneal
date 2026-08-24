# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CompileIQ adapter: search-space resolution + engine identity (no GPU, no engine installed).

The adapter is the bring-your-own-engine boundary: it imports ONLY ``compileiq`` + stdlib. That is
what lets these tests run anywhere -- we inject a stub ``compileiq`` into ``sys.modules``, so CI
never needs the real engine (which is >=3.11,<3.14 + glibc 2.34) to cover this logic.
"""

import importlib.util
import sys
import types

import pytest

ADAPTER = "ptx_anneal.adapters.compileiq_adapter"


def _load_adapter():
    """Import the adapter fresh (it is a standalone script, normally run by its own interpreter)."""
    sys.modules.pop(ADAPTER, None)
    spec = importlib.util.find_spec(ADAPTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_compileiq(monkeypatch):
    """A minimal stand-in for the engine, recording what the adapter asked it for."""
    calls = {}

    class LocalSearchSpaceBin:
        def __init__(self, path):
            calls["local_path"] = str(path)
            self.path = str(path)

    class PtxasSearchSpace:
        def __init__(self, **kw):
            calls["selector"] = kw
            self.resolution_metadata = None

        def retrieve(self):
            calls["retrieved"] = True
            self.resolution_metadata = types.SimpleNamespace(
                as_dict=lambda: {
                    "compiler": "ptxas",
                    "compiler_version": "13.3",
                    "variant": "default",
                    "requested_tag": "latest",
                    "resolved_tag": "search-spaces-2026.05.22",
                    "sha256": "deadbeef",
                    "size_bytes": 13632,
                    "source": "github_release",
                }
            )
            return "/cache/ptxas13.3_search_space.bin"

    pkg = types.ModuleType("compileiq")
    pkg.__path__ = []  # namespace-ish package, like the real one
    compilers = types.ModuleType("compileiq.search_spaces.compilers")
    compilers.LocalSearchSpaceBin = LocalSearchSpaceBin
    compilers.PtxasSearchSpace = PtxasSearchSpace
    search_spaces = types.ModuleType("compileiq.search_spaces")
    ciq = types.ModuleType("compileiq.ciq")
    ciq.__file__ = "/fake/site-packages/compileiq/ciq.py"
    ciq.Search = object

    # A real `import a.b` binds b onto a; sys.modules injection alone does not, so do it by hand or
    # attribute access (`compileiq.ciq.__file__`) fails in a way the real engine never would.
    pkg.ciq = ciq
    pkg.search_spaces = search_spaces
    search_spaces.compilers = compilers

    for name, mod in [
        ("compileiq", pkg),
        ("compileiq.search_spaces", search_spaces),
        ("compileiq.search_spaces.compilers", compilers),
        ("compileiq.ciq", ciq),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def test_explicit_ss_wins_and_is_recorded(fake_compileiq, tmp_path):
    ss = tmp_path / "my.bin"
    ss.write_bytes(b"\x00")
    provider, meta = _load_adapter()._resolve_search_space(str(ss))
    assert provider.path == str(ss)
    assert meta["source"] == "PTX_ANNEAL_SS" and meta["path"] == str(ss)
    assert "selector" not in fake_compileiq  # the catalog was never consulted


def test_missing_explicit_ss_fails_loudly(fake_compileiq, tmp_path):
    with pytest.raises(SystemExit, match="search space not found"):
        _load_adapter()._resolve_search_space(str(tmp_path / "nope.bin"))


def test_unset_ss_resolves_the_catalog_and_returns_provenance(fake_compileiq, monkeypatch):
    for var in ("CIQ_SS_VERSION", "CIQ_SS_VARIANT", "CIQ_SS_TAG"):
        monkeypatch.delenv(var, raising=False)
    _, meta = _load_adapter()._resolve_search_space("")
    # Nothing forwarded: the catalog's own defaults stay the single source of truth.
    assert fake_compileiq["selector"] == {}
    # Resolved eagerly, so a fetch failure surfaces here and the sidecar gets its provenance.
    assert fake_compileiq["retrieved"] is True
    assert meta["resolved_tag"] == "search-spaces-2026.05.22" and meta["sha256"] == "deadbeef"


def test_ciq_ss_env_is_forwarded_as_a_selector(fake_compileiq, monkeypatch):
    monkeypatch.setenv("CIQ_SS_TAG", "search-spaces-2026.05.22")
    monkeypatch.setenv("CIQ_SS_VARIANT", "att")
    monkeypatch.delenv("CIQ_SS_VERSION", raising=False)
    _load_adapter()._resolve_search_space("")
    assert fake_compileiq["selector"] == {"variant": "att", "tag": "search-spaces-2026.05.22"}


def test_engine_info_identifies_the_engine(fake_compileiq):
    info = _load_adapter()._engine_info()
    assert info["name"] == "compileiq"
    # compileiq is a namespace package (__file__ is None), so the path comes from a real submodule.
    assert info["path"] == "/fake/site-packages/compileiq"
    assert info["python"] == sys.executable
    assert info["version"]  # "unknown" when metadata is unavailable, never missing


def test_task_timeout_and_clock_are_off_by_default(fake_compileiq, monkeypatch):
    """The doc-conformance knobs must not change the shipped torch-free path unless asked for.

    Unset means *absent*, not zero: ``start()`` keeps the engine's own default, and the GPU clocks
    are not touched at all (``gpu_benchmark_mode(None)`` would warn, or raise).
    """
    mod = _load_adapter()
    for var in ("CIQ_TASK_TIMEOUT", "CIQ_CLOCK_MHZ"):
        monkeypatch.delenv(var, raising=False)
    assert mod._env_num("CIQ_TASK_TIMEOUT", float) is None
    assert mod._env_num("CIQ_CLOCK_MHZ", int) is None
    # nullcontext, not the engine's clock manager -- and reached without importing compileiq.utils.
    with mod._benchmark_mode(None):
        pass


def test_task_timeout_and_clock_are_read_when_set(fake_compileiq, monkeypatch):
    mod = _load_adapter()
    monkeypatch.setenv("CIQ_TASK_TIMEOUT", "20")
    monkeypatch.setenv("CIQ_CLOCK_MHZ", "1965")
    assert mod._env_num("CIQ_TASK_TIMEOUT", float) == 20.0
    assert mod._env_num("CIQ_CLOCK_MHZ", int) == 1965


def test_clock_lock_uses_the_engines_context_manager(fake_compileiq, monkeypatch):
    """When a clock IS requested we must use the engine's own gpu_benchmark_mode, non-fatally: a box
    that cannot lock clocks should still produce a (noisier) result, and the fingerprint records the
    clock state that actually applied."""
    seen = {}
    gpu = types.ModuleType("compileiq.utils.gpu")

    def gpu_benchmark_mode(**kw):
        seen.update(kw)
        return __import__("contextlib").nullcontext()

    gpu.gpu_benchmark_mode = gpu_benchmark_mode
    utils = types.ModuleType("compileiq.utils")
    utils.gpu = gpu
    sys.modules["compileiq"].utils = utils
    monkeypatch.setitem(sys.modules, "compileiq.utils", utils)
    monkeypatch.setitem(sys.modules, "compileiq.utils.gpu", gpu)

    with _load_adapter()._benchmark_mode(1965):
        pass
    assert seen["clock_mhz"] == 1965 and seen["raise_on_failure"] is False


def test_adapter_never_imports_ptx_anneal():
    """The engine-free boundary: the adapter must be runnable in an interpreter without ptx_anneal."""
    src = importlib.util.find_spec(ADAPTER).origin
    with open(src) as f:
        body = f.read()
    assert "import ptx_anneal" not in body and "from ptx_anneal" not in body
