# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The triton-native channel (CPU-only: no torch, no triton, no GPU, no engine).

Everything covered here is a place where a mistake is *invisible at runtime*:

* the ACF environment -- get it wrong and the search happily measures the untuned kernel and
  reports a win;
* the autotune refusal -- accept an autotuned kernel and the search tunes something undefined;
* the ``ref_fn=None`` labelling -- drop the label and a self-referential correctness check is
  indistinguishable from a real one;
* SCORE_CMD -- the one plumbing contract shared with the PTX-direct channel.

The modules under test import torch/triton strictly inside functions, which is what makes these
tests possible at all -- and testing them here is what keeps it true.
"""

import json
import sys
import types

import pytest

# e2e/ is deliberately excluded from the sdist, so it can genuinely be absent. In a checkout it is
# importable via the repo-root conftest.py.
pytest.importorskip("e2e.triton_native.score", reason="e2e/ is not packaged (sdist excludes it)")

from e2e.registry import WEAKER_VERDICT, KernelSpec  # noqa: E402
from e2e.triton_native import runway, score  # noqa: E402


# -- the ACF environment: the entire mechanism of the channel --------------------------------------
def test_acf_env_applies_the_controls_file():
    env = score.acf_env("/tmp/cand.acf", "/opt/ptxas")
    # FBTriton's knob is PTXAS_OPTIONS, NOT the OAI spelling TRITON_PTXAS_OPTIONS. The wrong name is
    # silently ignored by triton, so the search would score the untuned kernel N times.
    assert env["PTXAS_OPTIONS"] == "--apply-controls=/tmp/cand.acf"
    assert "TRITON_PTXAS_OPTIONS" not in env


def test_acf_env_forces_recompilation():
    """Without this every candidate after the first is served from triton's cache."""
    assert score.acf_env("/tmp/x.acf", None)["TRITON_ALWAYS_COMPILE"] == "1"


def test_acf_env_pins_ptxas_for_both_arch_families():
    """Blackwell reads TRITON_PTXAS_BLACKWELL_PATH; older arches TRITON_PTXAS_PATH. One ptxas, both vars."""
    env = score.acf_env("/tmp/x.acf", "/wheel/bin/ptxas")
    assert env["TRITON_PTXAS_BLACKWELL_PATH"] == "/wheel/bin/ptxas"
    assert env["TRITON_PTXAS_PATH"] == "/wheel/bin/ptxas"


def test_acf_env_baseline_has_no_apply_controls():
    """The no-ACF baseline must compile exactly as an untouched triton kernel: absent, not empty."""
    env = score.acf_env(None, "/opt/ptxas")
    assert "PTXAS_OPTIONS" not in env
    assert env["TRITON_ALWAYS_COMPILE"] == "1"


def test_acf_env_never_sets_the_magnon_consume_hook():
    """TRITON_COMPILE_IQ_APPLY would append a SECOND --apply-controls."""
    assert "TRITON_COMPILE_IQ_APPLY" not in score.acf_env("/tmp/x.acf", "/opt/ptxas")


def test_refuses_to_run_under_the_magnon_consume_hook():
    with pytest.raises(SystemExit, match="TRITON_COMPILE_IQ_APPLY"):
        score.check_env_clean({"TRITON_COMPILE_IQ_APPLY": "1"})


def test_clean_env_is_accepted():
    assert score.check_env_clean({}) is None


# -- autotune refusal ------------------------------------------------------------------------------
class Autotuner:
    """Stands in for triton.runtime.autotuner.Autotuner (what @triton.autotune returns).

    Detection is by type NAME on purpose, so it works without importing triton -- which is what lets
    the refusal be covered by a CPU-only test at all.
    """


def _spec_with_globals(g, name="k"):
    fn = types.FunctionType(compile("x=1", "<t>", "exec"), g, name)
    return KernelSpec(name=name, make_inputs=lambda: (), run_fn=fn, ref_fn=lambda: None)


def test_autotuned_kernel_is_refused():
    spec = _spec_with_globals({"__name__": "mykernel", "matmul_kernel": Autotuner()})
    with pytest.raises(SystemExit) as e:
        score.refuse_autotune(spec)
    msg = str(e.value)
    assert "mykernel.matmul_kernel" in msg
    # The refusal must say WHY and HOW to proceed -- a bare "unsupported" sends people to the source.
    assert "UNSUPPORTED" in msg and "Pin it to ONE config" in msg


def test_autotune_is_found_one_import_away():
    """A KernelSpec usually wraps a kernel defined elsewhere (ws_gemm wraps a tlx tutorial)."""
    tutorial = {"__name__": "tlx.tutorial", "matmul_kernel": Autotuner()}
    wrapper = types.FunctionType(compile("x=1", "<t>", "exec"), tutorial, "matmul")
    spec = _spec_with_globals({"__name__": "wrapper_mod", "matmul": wrapper})
    assert score.find_autotuners(spec.run_fn) == ["tlx.tutorial.matmul_kernel (0 configs)"]


def test_an_autotuner_narrowed_to_one_config_is_a_pin_not_autotuning():
    """Triton's own rule: Autotuner.run only benchmarks when len(configs) > 1. With exactly one it
    uses that config directly -- deterministic, no sweep -- which is what "pinned" means. Refusing
    this form would block 16 of the 19 Blackwell TLX tutorials for no safety gain."""
    one = Autotuner()
    one.configs = ["cfg"]
    spec = _spec_with_globals({"__name__": "tut", "k": one})
    assert score.find_autotuners(spec.run_fn) == []
    assert score.refuse_autotune(spec) is None


def test_two_configs_is_still_refused_and_the_count_is_named():
    two = Autotuner()
    two.configs = ["a", "b"]
    spec = _spec_with_globals({"__name__": "tut", "k": two})
    with pytest.raises(SystemExit, match=r"tut\.k \(2 configs\)"):
        score.refuse_autotune(spec)


def test_pin_autotuner_narrows_in_place_and_reports_the_config():
    """The helper is how a tutorial gets pinned WITHOUT editing it."""
    from e2e.kernels import pin_autotuner

    class Cfg:
        def __init__(self, **kw):
            self.kwargs = kw
            self.num_warps, self.num_stages, self.num_ctas = 8, 3, 1

    class Tuner:
        def __init__(self):
            self.configs = [Cfg(BLOCK_SIZE_M=1), Cfg(BLOCK_SIZE_M=4)]
            self.cache = {"stale": "entry"}
            self.base_fn = lambda: None
            self.base_fn.__name__ = "k"

    t = Tuner()
    got = pin_autotuner(t, match={"BLOCK_SIZE_M": 4})
    assert len(t.configs) == 1 and t.configs[0].kwargs["BLOCK_SIZE_M"] == 4
    assert t.cache == {}  # a cached selection would defeat the pin
    assert got["BLOCK_SIZE_M"] == 4 and got["num_warps"] == 8
    assert "of 2 configs" in got["pinned_from"]


def test_pin_autotuner_refuses_an_ambiguous_match():
    from e2e.kernels import pin_autotuner

    class Cfg:
        def __init__(self, **kw):
            self.kwargs = kw

    class Tuner:
        configs = [Cfg(A=1), Cfg(A=1)]

    with pytest.raises(ValueError, match="selected 2 of 2"):
        pin_autotuner(Tuner(), match={"A": 1})


def test_pinned_kernel_is_accepted():
    spec = _spec_with_globals({"__name__": "pinned", "CONFIG": {"BLOCK_M": 128}})
    assert score.refuse_autotune(spec) is None


def test_autotune_walk_terminates_on_recursive_globals():
    """A module referencing a function defined in itself must not loop forever."""
    g = {"__name__": "selfref"}
    g["f"] = types.FunctionType(compile("x=1", "<t>", "exec"), g, "f")
    assert score.find_autotuners(g["f"]) == []


# -- ref_fn=None: allowed, and always labelled weaker -----------------------------------------------
def test_ref_fn_defaults_to_none_and_is_reported_as_weaker():
    spec = KernelSpec(name="pasted", make_inputs=lambda: (), run_fn=lambda: None)
    assert spec.ref_fn is None
    assert spec.has_independent_ref is False


def test_ref_fn_present_is_independent():
    spec = KernelSpec(name="k", make_inputs=lambda: (), run_fn=lambda: None, ref_fn=lambda: None)
    assert spec.has_independent_ref is True


def test_weaker_verdict_text_names_the_actual_risk():
    assert "self-referential" in WEAKER_VERDICT and "no ref_fn" in WEAKER_VERDICT


def test_fingerprint_labels_a_missing_ref_fn_as_weaker():
    args = runway.build_parser().parse_args([])
    fp = runway._fingerprint("k", {"has_independent_ref": False}, "/p/ptxas", "13.3", {}, "/tmp/r.pt", args)
    assert fp["objective"]["correctness"]["strength"] == "weaker"
    assert fp["objective"]["correctness"]["oracle"] == "baseline-output (self-referential)"


def test_fingerprint_labels_a_present_ref_fn_as_independent():
    args = runway.build_parser().parse_args([])
    fp = runway._fingerprint("k", {"has_independent_ref": True}, "/p/ptxas", "13.3", {}, None, args)
    assert fp["objective"]["correctness"]["strength"] == "independent"
    assert fp["objective"]["correctness"]["oracle"] == "ref_fn"


def test_fingerprint_records_that_autotune_was_ruled_out():
    args = runway.build_parser().parse_args([])
    fp = runway._fingerprint("k", {"has_independent_ref": True}, "/p/ptxas", "13.3", {}, None, args)
    assert fp["kernel"]["autotune"] is False
    assert fp["objective"]["seed"] == score.SEED  # inputs are identical in every candidate process


def test_the_two_channels_declare_different_channel_names():
    """Trivial, but it is what stops a diff of two fingerprints from silently comparing a run to itself."""
    args = runway.build_parser().parse_args([])
    fp = runway._fingerprint("k", {}, "/p/ptxas", "13.3", {}, None, args)
    assert fp["channel"] == "triton-native"


# -- SCORE_CMD: the plumbing shared with the PTX-direct channel -------------------------------------
def test_search_drives_the_bundled_adapter_through_score_cmd(monkeypatch, tmp_path):
    """No second mechanism: the engine reaches this scorer via $PTX_ANNEAL_SCORE_CMD, as documented."""
    captured = {}

    def fake_run(cmd, env=None, cwd=None, check=False):
        captured["cmd"], captured["env"] = cmd, env
        (tmp_path / "k.result.json").write_text(json.dumps({"valid": 1, "evaluated": 6}))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(runway.subprocess, "run", fake_run)
    monkeypatch.setattr(runway, "_adapter_path", lambda: "/fake/compileiq_adapter.py")
    args = runway.build_parser().parse_args([])
    args.ptxas_version = "13.3"

    res = runway._search(sys.executable, "k", "/opt/ptxas", "/tmp/ref.pt", str(tmp_path), args)
    assert res["valid"] == 1

    score_cmd = json.loads(captured["env"]["PTX_ANNEAL_SCORE_CMD"])
    assert score_cmd[1:3] == ["-m", "e2e.triton_native.score"]
    assert "--kernel" in score_cmd and "k" in score_cmd
    assert "--ref" in score_cmd  # the ref_fn=None fallback reference is threaded through
    # The adapter's own contract field. The scorer cross-checks it against --kernel.
    assert captured["env"]["PTX_ANNEAL_TASK"] == "k"
    assert captured["cmd"] == [sys.executable, "/fake/compileiq_adapter.py"]


def test_search_passes_the_documented_budget(monkeypatch, tmp_path):
    """The doc's numbers reach the engine as env, through knobs the adapter already owns."""
    captured = {}

    def fake_run(cmd, env=None, cwd=None, check=False):
        captured.update(env)
        (tmp_path / "k.result.json").write_text("{}")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(runway.subprocess, "run", fake_run)
    monkeypatch.setattr(runway, "_adapter_path", lambda: "/fake/a.py")
    args = runway.build_parser().parse_args([])
    args.ptxas_version = "13.3"
    runway._search(sys.executable, "k", "/opt/ptxas", None, str(tmp_path), args)

    assert captured["CIQ_GENERATIONS"] == str(runway.DOC_GENERATIONS)
    assert captured["CIQ_POOL"] == str(runway.DOC_POOL)
    assert captured["CIQ_TASK_TIMEOUT"] == str(runway.DOC_TASK_TIMEOUT)
    assert captured["CIQ_CLOCK_MHZ"] == str(runway.B200_MAX_SM_MHZ)
    assert captured["PTX_ANNEAL_WARMUP"] == str(runway.DOC_WARMUP)
    assert captured["PTX_ANNEAL_REP"] == str(runway.DOC_REP)
    # Doc conformance: the search space is derived from the LOCAL ptxas version, not a pinned tag.
    assert captured["CIQ_SS_VERSION"] == "13.3"


def test_caller_can_still_pin_the_search_space(monkeypatch, tmp_path):
    """CIQ_SS_* stays a pin, not a prerequisite -- an explicit one must win over the doc default."""
    captured = {}
    monkeypatch.setenv("CIQ_SS_VERSION", "13.4")
    monkeypatch.setattr(runway, "_adapter_path", lambda: "/fake/a.py")

    def fake_run(cmd, env=None, cwd=None, check=False):
        captured.update(env)
        (tmp_path / "k.result.json").write_text("{}")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(runway.subprocess, "run", fake_run)
    args = runway.build_parser().parse_args([])
    args.ptxas_version = "13.3"
    runway._search(sys.executable, "k", "/opt/ptxas", None, str(tmp_path), args)
    assert captured["CIQ_SS_VERSION"] == "13.4"


# -- degradation and doc conformance ---------------------------------------------------------------
def test_reports_itself_unavailable_without_triton(monkeypatch, capsys):
    """Requirement: absent triton/torch, the runway degrades cleanly and the repo is unaffected."""
    monkeypatch.setattr(runway, "availability", lambda: (False, ["triton -- this channel compiles through Triton"]))
    assert runway.main([]) == 2
    err = capsys.readouterr().err
    assert "unavailable on this box" in err and "triton" in err
    # It must also say the rest of the repo still works -- this channel is optional by construction.
    assert "MODE=tune" in err


def test_availability_reports_actionable_reasons():
    """Runs on any box: with the stack present it is simply available, without it every missing piece
    must come back as a sentence saying what it is for -- a bare False is not actionable."""
    ok, missing = runway.availability()
    assert isinstance(ok, bool) and isinstance(missing, list)
    assert ok == (not missing)
    assert all(isinstance(m, str) and " -- " in m for m in missing)


def test_pool_size_floor_is_enforced_before_anything_runs():
    """CompileIQ pydantic-validates pool_size > 5; failing here beats failing after a GPU warmup."""
    with pytest.raises(SystemExit, match="--pool must be > 5"):
        runway.main(["--pool", "4"])


def test_defaults_match_nvidias_documented_example():
    """Deviating from the doc for no reason is what makes a CompileIQ bug report deflectable."""
    args = runway.build_parser().parse_args([])
    assert (args.generations, args.pool) == (5, 32)
    # task_timeout deviates from the doc's 20s on purpose -- see DOC_TASK_TIMEOUT.
    assert args.task_timeout == 120
    assert (args.warmup, args.rep) == (100, 1000)
    assert args.oracle == "allclose" and args.atol == 1e-2
    assert args.clock_mhz == 1965  # B200 max SM clock; the doc calls locking clocks crucial


def test_canary_runs_first_by_default():
    args = runway.build_parser().parse_args([])
    assert args.canary is True
    assert runway.CANARY == "doc_matmul"
    assert runway.build_parser().parse_args(["--no-canary"]).canary is False


# -- the frontend is fbtriton, and the ACF application is verified rather than assumed -------------
def _cuda_options(default):
    """A stand-in for triton's CUDAOptions dataclass with a given ptx_options default."""
    import dataclasses

    @dataclasses.dataclass
    class CUDAOptions:
        ptx_options: str | None = default

    return CUDAOptions


def _install_backend(monkeypatch, cuda_options):
    """Inject triton.backends.nvidia.compiler without needing triton."""
    mods = {}
    for name in ("triton", "triton.backends", "triton.backends.nvidia", "triton.backends.nvidia.compiler"):
        mods[name] = types.ModuleType(name)
    mods["triton.backends.nvidia.compiler"].CUDAOptions = cuda_options
    mods["triton"].backends = mods["triton.backends"]
    mods["triton.backends"].nvidia = mods["triton.backends.nvidia"]
    mods["triton.backends.nvidia"].compiler = mods["triton.backends.nvidia.compiler"]
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, name, mod)


def test_verify_accepts_when_the_acf_reached_the_compiler(monkeypatch):
    _install_backend(monkeypatch, _cuda_options("--apply-controls=/tmp/c.acf"))
    assert score.verify_acf_applied("/tmp/c.acf") == "--apply-controls=/tmp/c.acf"


def test_verify_accepts_the_untuned_baseline(monkeypatch):
    """The baseline must compile with NO ptx_options -- otherwise it is not a baseline."""
    _install_backend(monkeypatch, _cuda_options(None))
    assert score.verify_acf_applied(None) is None


def test_verify_refuses_when_the_acf_was_silently_dropped(monkeypatch):
    """The dangerous case: wrong knob name, or triton imported too early. The kernel would compile
    untuned, run fine, and be scored as a valid candidate -- a win the search never actually made."""
    _install_backend(monkeypatch, _cuda_options(None))
    with pytest.raises(SystemExit) as e:
        score.verify_acf_applied("/tmp/c.acf")
    msg = str(e.value)
    assert "did not reach the compiler" in msg
    assert "TRITON_PTXAS_OPTIONS" in msg  # names the upstream spelling, the likeliest cause
    assert "untuned" in msg


def test_verify_refuses_a_triton_without_the_ptx_options_field(monkeypatch):
    """Upstream Triton: no ptx_options on CUDAOptions at all, so an ACF can never be applied."""
    import dataclasses

    @dataclasses.dataclass
    class Upstream:
        num_warps: int = 4

    _install_backend(monkeypatch, Upstream)
    with pytest.raises(SystemExit, match="fbtriton"):
        score.verify_acf_applied("/tmp/c.acf")


def test_verify_refuses_a_mismatched_acf(monkeypatch):
    """A stale ptx_options from a previous candidate would score kernel A's ACF as kernel B's."""
    _install_backend(monkeypatch, _cuda_options("--apply-controls=/tmp/OTHER.acf"))
    with pytest.raises(SystemExit, match="did not reach the compiler"):
        score.verify_acf_applied("/tmp/c.acf")


def test_availability_rejects_an_upstream_triton_layout(monkeypatch, tmp_path):
    """fbtriton is detected by tlx/ or magnon/ on disk -- no import, so the runway stays stdlib."""
    import importlib.util

    fake = tmp_path / "triton"
    (fake / "language" / "extra").mkdir(parents=True)  # no tlx/, no magnon/
    real = importlib.util.find_spec

    def spec_for(name):
        if name == "triton":
            return types.SimpleNamespace(submodule_search_locations=[str(fake)])
        return real(name)

    monkeypatch.setattr(importlib.util, "find_spec", spec_for)
    _, missing = runway.availability()
    assert any(m.startswith("fbtriton --") for m in missing)
    assert any("untuned" in m for m in missing)


def test_availability_accepts_an_fbtriton_layout(monkeypatch, tmp_path):
    import importlib.util

    fake = tmp_path / "triton"
    (fake / "language" / "extra" / "tlx").mkdir(parents=True)
    real = importlib.util.find_spec

    def spec_for(name):
        if name == "triton":
            return types.SimpleNamespace(submodule_search_locations=[str(fake)])
        return real(name)

    monkeypatch.setattr(importlib.util, "find_spec", spec_for)
    _, missing = runway.availability()
    assert not any(m.startswith("fbtriton --") for m in missing)


def test_fingerprint_records_the_frontend_and_the_verified_mechanism():
    args = runway.build_parser().parse_args([])
    identity = {"has_independent_ref": True, "frontend": {"name": "fbtriton", "version": "3.8.0+fb"}}
    fp = runway._fingerprint("k", identity, "/p/ptxas", "13.3", {}, None, args)
    assert fp["frontend"]["name"] == "fbtriton"
    assert fp["frontend"]["acf_mechanism"] == "PTXAS_OPTIONS=--apply-controls=<acf>"
    assert fp["frontend"]["acf_applied_verified"] is True


def test_autotune_is_found_through_a_closure():
    """A factory like `_make_run(attention)` hides the tutorial in a closure cell; its __globals__
    are the WRAPPER's module, where no autotuner lives. Following only globals would give a clean
    bill of health to a fully autotuned kernel -- the one wrong answer this check must never give."""
    tuner = Autotuner()
    tuner.configs = ["a", "b", "c"]
    tutorial = {"__name__": "tlx.tut", "attn": tuner}
    entry = types.FunctionType(compile("x=1", "<t>", "exec"), tutorial, "attention")

    def make_run(fn):
        def _run():
            return fn()

        return _run

    wrapper = make_run(entry)  # __globals__ is THIS test module, not tlx.tut
    assert score.find_autotuners(wrapper) == ["tlx.tut.attn (3 configs)"]


def test_closure_walk_survives_an_empty_cell():
    def outer():
        def inner():
            return outer  # a cell that is still being filled at definition time

        return inner

    assert score.find_autotuners(outer()) == []


# -- search space is pluggable, and whatever is used is recorded ------------------------------------
def _capture_search_env(monkeypatch, tmp_path, argv):
    captured = {}

    def fake_run(cmd, env=None, cwd=None, check=False):
        captured.update(env)
        (tmp_path / "k.result.json").write_text("{}")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(runway.subprocess, "run", fake_run)
    monkeypatch.setattr(runway, "_adapter_path", lambda: "/fake/a.py")
    args = runway.build_parser().parse_args(argv)
    args.ptxas_version = "13.3"
    runway._search(sys.executable, "k", "/opt/ptxas", None, str(tmp_path), args)
    return captured


def test_explicit_ss_bin_is_forwarded_and_absolute(monkeypatch, tmp_path):
    """An explicit .bin short-circuits the catalog in the adapter, so it must arrive as a real path."""
    env = _capture_search_env(monkeypatch, tmp_path, ["--ss", "rel/path.bin"])
    assert env["PTX_ANNEAL_SS"].endswith("/rel/path.bin")
    assert env["PTX_ANNEAL_SS"].startswith("/")


def test_ss_tag_and_variant_are_forwarded(monkeypatch, tmp_path):
    env = _capture_search_env(monkeypatch, tmp_path, ["--ss-tag", "search-spaces-2026.09.01", "--ss-variant", "att"])
    assert env["CIQ_SS_TAG"] == "search-spaces-2026.09.01"
    assert env["CIQ_SS_VARIANT"] == "att"


def test_ss_flag_beats_the_environment(monkeypatch, tmp_path):
    """Flags beat env beats the doc default -- otherwise a stale shell export silently decides which
    catalog an experiment used, and the digest would be the only place it showed up."""
    monkeypatch.setenv("CIQ_SS_TAG", "stale-from-the-shell")
    env = _capture_search_env(monkeypatch, tmp_path, ["--ss-tag", "search-spaces-2026.09.01"])
    assert env["CIQ_SS_TAG"] == "search-spaces-2026.09.01"


# -- the O0/O3 injection canary --------------------------------------------------------------------
def _canary(monkeypatch, base, o0, o3):
    times = {None: base, "-O0": o0, "-O3": o3}
    monkeypatch.setattr(runway, "_time_with_opts", lambda py, k, p, opts, a: times[opts])
    args = runway.build_parser().parse_args([])
    return runway.injection_canary(sys.executable, "k", "/opt/ptxas", args)


def test_injection_canary_passes_when_o0_regresses(monkeypatch):
    ok, note = _canary(monkeypatch, base=1.0, o0=15.0, o3=1.0)
    assert ok and "15.00x" in note


def test_injection_canary_fails_when_o0_does_not_regress(monkeypatch):
    """The whole point: an ignored option and a no-op option produce identical scores. Only a flag
    with a known-bad effect distinguishes them, so a flat -O0 means nothing reached ptxas."""
    ok, note = _canary(monkeypatch, base=1.0, o0=1.0, o3=1.0)
    assert not ok
    assert "did not regress" in note and "untuned kernel" in note


def test_injection_canary_warns_but_passes_when_o3_drifts(monkeypatch):
    """-O3 is the default, so a gap is noise or a non-default baseline -- worth saying, not fatal."""
    ok, note = _canary(monkeypatch, base=1.0, o0=8.0, o3=2.0)
    assert ok and "WARN" in note


def test_injection_canary_fails_when_a_probe_does_not_run(monkeypatch):
    ok, note = _canary(monkeypatch, base=1.0, o0=None, o3=1.0)
    assert not ok and "did not run" in note


def test_raw_ptxas_opts_bypass_the_acf_path():
    """The canary must travel the SAME env var as an ACF, or it proves nothing about ACF injection."""
    env = score.acf_env("/tmp/ignored.acf", "/opt/ptxas", raw_opts="-O0")
    assert env["PTXAS_OPTIONS"] == "-O0"
    assert score.expected_ptx_options("/tmp/ignored.acf", "-O0") == "-O0"
    assert score.expected_ptx_options("/tmp/x.acf", None) == "--apply-controls=/tmp/x.acf"
    assert score.expected_ptx_options(None, None) is None


def _stub_main(monkeypatch, seen):
    """Make runway.main() reach its kernel loop without touching the machine.

    main() resolves a real ptxas, so left unstubbed these tests pass on a devgpu and fail in CI --
    the exact CPU-only-tests trap AGENTS.md warns about. `provision` is imported inside main(), so
    the stubs go on the module object, not on a runway attribute.
    """
    from ptx_anneal import provision

    monkeypatch.setattr(runway, "availability", lambda: (True, []))
    monkeypatch.setattr(runway, "_ensure_repo_on_path", lambda: "/repo")
    monkeypatch.setattr(runway, "injection_canary", lambda *a, **k: (True, "stub"))
    monkeypatch.setattr(provision, "find_ptxas", lambda: "/fake/ptxas")
    monkeypatch.setattr(provision, "ptxas_version", lambda p: "13.3")
    monkeypatch.delenv("PTXAS", raising=False)
    # Returning baseline_ms=None makes main() report the kernel and move on, so the loop is
    # exercised without a search ever being spawned.
    monkeypatch.setattr(runway, "_run_probe", lambda py, k, p, w, a: (seen.append(k), (None, {}, None))[1])


def test_naming_the_canary_with_no_canary_still_runs_it(monkeypatch):
    """`--kernel doc_matmul --no-canary` is how you tune the canary itself. The dedupe that stops it
    running twice must not fire when the implicit canary is switched off."""
    seen = []
    _stub_main(monkeypatch, seen)
    runway.main(["--kernel", runway.CANARY, "--no-canary", "--work", "/tmp/x"])
    assert seen == [runway.CANARY]


def test_canary_is_not_run_twice_when_also_named(monkeypatch):
    seen = []
    _stub_main(monkeypatch, seen)
    runway.main(["--kernel", runway.CANARY, "--work", "/tmp/x"])
    assert seen == [runway.CANARY]
