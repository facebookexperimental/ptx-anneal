# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Standalone replay and clustered driver launch coverage, with CPU-only stubs."""

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ptx_anneal import cli, replay
from ptx_anneal import ptx_launch as L
from ptx_anneal.task import Task

SAMPLE_TASK = str(Path(__file__).resolve().parents[1] / "sample_tasks" / "sample_task")


@pytest.fixture(autouse=True)
def isolated_ptxas_env(monkeypatch):
    monkeypatch.setenv("TRITON_PTXAS_BLACKWELL_PATH", "")


def test_replay_cli_skips_engine_and_overrides_captured_ptxas(monkeypatch, capsys):
    run = Mock(return_value=0.25)
    monkeypatch.setattr(replay, "replay", run)
    monkeypatch.setattr(cli.provision, "ptxas_version", lambda p: "13.3")
    monkeypatch.setattr(cli, "_cmd_tune", Mock(side_effect=AssertionError("tuning was invoked")))
    monkeypatch.setattr(cli, "_bundled_adapter", Mock(side_effect=AssertionError("engine was resolved")))
    assert cli.main(["replay", "--task", SAMPLE_TASK, "--ptxas", "/explicit/ptxas"]) == 0
    assert run.call_args.args[1] == "/explicit/ptxas"
    assert "MS 0.25" in capsys.readouterr().out


@pytest.mark.parametrize("captured, expected", [(False, "inputs=synthetic"), (True, "inputs=captured 1/3")])
def test_replay_cli_reports_captured_inputs(monkeypatch, capsys, tmp_path, captured, expected):
    task_dir = tmp_path / "task"
    shutil.copytree(SAMPLE_TASK, task_dir)
    spec = json.loads((task_dir / "spec.json").read_text())
    spec["tensors"][0]["pickle_sha256"] = digest = "a" * 64
    spec["tensors"][1]["pickle_sha256"] = "b" * 64  # named but absent: synthesized
    (task_dir / "spec.json").write_text(json.dumps(spec))
    if captured:
        (task_dir / f"{digest}.pkl").write_bytes(b"")
    monkeypatch.setattr(replay, "replay", Mock(return_value=0.25))
    monkeypatch.setattr(cli.provision, "ptxas_version", lambda p: "13.3")
    assert cli.main(["replay", "--task", str(task_dir), "--ptxas", "/ptxas"]) == 0
    assert expected in capsys.readouterr().out


@pytest.mark.parametrize("version", [None, "13.2"])
def test_replay_enforces_ptxas_floor(monkeypatch, version):
    monkeypatch.setattr(cli.provision, "ptxas_version", lambda p: version)
    monkeypatch.setattr(replay, "replay", Mock(side_effect=AssertionError("replay started")))
    with pytest.raises(SystemExit, match="13.3"):
        cli.main(["replay", "--task", SAMPLE_TASK, "--ptxas", "/old/ptxas"])


@pytest.mark.parametrize("outcome", [RuntimeError("launch failed"), float("nan"), float("inf"), 0.0])
def test_replay_cli_failure_is_nonzero(monkeypatch, outcome, capsys):
    monkeypatch.setattr(cli.provision, "ptxas_version", lambda p: "13.3")
    run = Mock(side_effect=outcome) if isinstance(outcome, Exception) else Mock(return_value=outcome)
    monkeypatch.setattr(replay, "replay", run)
    assert cli.main(["replay", "--task", SAMPLE_TASK, "--ptxas", "/ptxas"]) == 1
    captured = capsys.readouterr()
    assert "[replay]" in captured.err
    assert "MS " not in captured.out


@pytest.mark.parametrize("bench", ["cudagraph", "do_bench"])
def test_replay_launches_spec_without_acf(monkeypatch, bench):
    events = []
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: events.append("sync")))
    )
    spec = {
        "entry": "k",
        "arch": "sm_100a",
        "shared": 100,
        "cluster": [2, 1, 1],
        "grid": [4, 1, 1],
        "block": [256, 1, 1],
    }
    task = Task("/task", "PTX", spec)
    tensors, descriptors, addresses = [object()], [object()], [1234]
    args = [("tma", 1234)]
    compile_ptx = Mock(return_value=b"cubin")
    kernel = SimpleNamespace(launch=lambda *a: events.append("launch"), unload=Mock())
    load = Mock(return_value=kernel)
    monkeypatch.setattr(L, "ptxas_compile", compile_ptx)
    monkeypatch.setattr(L, "load_cubin", load)
    alloc = Mock(return_value=tensors)
    monkeypatch.setattr(L, "alloc_tensors", alloc)
    maps = Mock(return_value=(descriptors, addresses))
    monkeypatch.setattr(L, "build_tensormaps", maps)
    scratch = object()
    alloc_scratch = Mock(return_value=scratch)
    monkeypatch.setattr(L, "alloc_global_scratch", alloc_scratch)
    make_args = Mock(return_value=args)
    monkeypatch.setattr(L, "kernel_args_from_spec", make_args)

    def measure(*a, **kw):
        assert events == ["sync", "launch", "sync"]
        assert a == (kernel, spec["grid"], spec["block"], args)
        assert kw == {"warmup": 2, "reps": 3 if bench == "cudagraph" else 20}
        return 0.125

    monkeypatch.setattr(L, "bench_ms_" + bench, measure)
    assert replay.replay(task, "/ptxas", warmup=2, rep=20, bench=bench) == 0.125
    compile_ptx.assert_called_once_with("PTX", "/ptxas", arch="sm_100a")
    load.assert_called_once_with(b"cubin", "k", 100, cluster=[2, 1, 1])
    alloc.assert_called_once_with(spec, seed=0, task_dir="/task")
    maps.assert_called_once_with(spec, tensors)
    alloc_scratch.assert_called_once_with(spec)
    make_args.assert_called_once_with(spec, tensors, addresses, scratch)
    kernel.unload.assert_called_once()

    def failed_measure(*a, **kw):
        raise RuntimeError("device failure")

    monkeypatch.setattr(L, "bench_ms_" + bench, failed_measure)
    with pytest.raises(RuntimeError, match="device failure"):
        replay.replay(task, "/ptxas", bench=bench)
    assert kernel.unload.call_count == 2


@pytest.mark.parametrize("cluster", [None, [2, 1, 1]])
def test_driver_uses_extended_launch_for_clusters(monkeypatch, cluster):
    calls = []
    driver = SimpleNamespace(
        CUresult=SimpleNamespace(CUDA_SUCCESS=0),
        cuModuleLoadData=lambda b: (0, "module"),
        cuModuleGetFunction=lambda *a: (0, "function"),
        CUlaunchAttribute=lambda: SimpleNamespace(value=SimpleNamespace(clusterDim=SimpleNamespace())),
        CUlaunchAttributeID=SimpleNamespace(CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION="cluster"),
        CUlaunchConfig=SimpleNamespace,
        cuLaunchKernel=lambda *a: (calls.append(("plain", a)) or 0,),
        cuLaunchKernelEx=lambda *a: (calls.append(("extended", a)) or 0,),
    )
    monkeypatch.setattr(L, "_drv", driver)
    monkeypatch.setattr(L, "_ensure_init", lambda: None)
    kernel = L.load_cubin(b"cubin", "k", 123, cluster=cluster)
    kernel.launch([4, 1, 1], [256, 1, 1], [("i32", 7)], stream=42)
    kind, args = calls[0]
    assert kind == ("extended" if cluster else "plain")
    if cluster:
        config = args[0]
        assert (config.gridDimX, config.gridDimY, config.gridDimZ) == (4, 1, 1)
        assert (config.blockDimX, config.blockDimY, config.blockDimZ) == (256, 1, 1)
        assert config.hStream == 42 and config.sharedMemBytes == 123
        assert config.numAttrs == 1 and config.attrs[0].id == "cluster"
        dim = config.attrs[0].value.clusterDim
        assert (dim.x, dim.y, dim.z) == (2, 1, 1)
        with pytest.raises(ValueError, match="not divisible"):
            kernel.launch([3, 1, 1], [256, 1, 1], [])


@pytest.mark.parametrize("cluster", [[0, 1, 1], [2, 1], [1.5, 1, 1]])
def test_bad_cluster_rejected_before_cuda_initialization(cluster):
    with pytest.raises(ValueError, match="invalid cluster"):
        L.load_cubin(b"cubin", "k", 0, cluster=cluster)


def _report(numerics_error=None, deviations=(0.0, 0.0, 0.0)):
    return replay.AcfReport(0.2, 0.1, list(deviations), numerics_error, "RelTolValidator(rel_tol=0.01)")


@pytest.mark.parametrize("ok", [True, False])
def test_replay_cli_acf_reports_numerics_and_perf(monkeypatch, capsys, tmp_path, ok):
    acf = tmp_path / "k.acf"
    acf.write_bytes(b"acf")
    run = Mock(return_value=_report(None if ok else "candidate diverged", (0.0, 0.0, 0.5 if not ok else 0.0)))
    monkeypatch.setattr(replay, "replay_acf", run)
    monkeypatch.setattr(replay, "replay", Mock(side_effect=AssertionError("plain replay ran")))
    monkeypatch.setattr(cli.provision, "ptxas_version", lambda p: "13.3")
    argv = ["replay", "--task", SAMPLE_TASK, "--ptxas", "/ptxas", "--acf", str(acf), "--bench", "do_bench"]
    assert cli.main(argv) == (0 if ok else 1)
    assert run.call_args.args[1:] == ("/ptxas", str(acf))
    assert run.call_args.kwargs == {"warmup": 25, "rep": 50, "bench": "do_bench"}
    out = capsys.readouterr().out
    assert f"NUMERICS {'PASS' if ok else 'FAIL'}" in out
    assert "BASELINE_MS 0.2" in out and "ACF_MS 0.1" in out
    assert "speedup=2.000x (+50.00%)" in out
    assert out.count("  tensor[") == 3
    assert ("candidate diverged" in out) is not ok


def test_replay_cli_acf_warns_on_sidecar_mismatch(monkeypatch, capsys, tmp_path):
    acf = tmp_path / "k.acf"
    acf.write_bytes(b"acf")
    (tmp_path / "k.acf.json").write_text(json.dumps({"ir_hash": "f" * 64, "ptxas_version": "13.4"}))
    monkeypatch.setattr(replay, "replay_acf", Mock(return_value=_report()))
    monkeypatch.setattr(cli.provision, "ptxas_version", lambda p: "13.3")
    assert cli.main(["replay", "--task", SAMPLE_TASK, "--ptxas", "/ptxas", "--acf", str(acf)]) == 0
    out = capsys.readouterr().out
    assert "tuned for ir_hash ffffffffffffffff" in out
    assert "tuned with ptxas 13.4, replaying with 13.3" in out


def test_replay_cli_acf_missing_file_is_nonzero(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(replay, "replay_acf", Mock(side_effect=AssertionError("replay started")))
    monkeypatch.setattr(cli.provision, "ptxas_version", lambda p: "13.3")
    assert cli.main(["replay", "--task", SAMPLE_TASK, "--ptxas", "/p", "--acf", str(tmp_path / "no.acf")]) == 1
    assert "ACF not found" in capsys.readouterr().err


@pytest.mark.parametrize("diverge", [False, True])
def test_replay_acf_compares_before_benchmarking(monkeypatch, diverge):
    """Numerics are checked on the first launch of each kernel, before any benchmark re-launch."""
    events = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)))
    spec = {"entry": "k", "arch": "sm_100a", "shared": 0, "grid": [1, 1, 1], "block": [32, 1, 1]}
    task = Task("/task", "PTX", spec)
    kernels = {}

    def compile_ptx(ptx, ptxas, arch, acf_path=None):
        return acf_path or "base"

    def load(cubin, *a, **kw):
        kernels[cubin] = SimpleNamespace(
            name=cubin,
            launch=lambda *a: events.append(("launch", cubin)),
            unload=lambda: events.append(("unload", cubin)),
        )
        return kernels[cubin]

    monkeypatch.setattr(L, "ptxas_compile", compile_ptx)
    monkeypatch.setattr(L, "load_cubin", load)
    monkeypatch.setattr(L, "alloc_tensors", lambda spec, seed, task_dir: [object(), object()])
    monkeypatch.setattr(L, "build_tensormaps", lambda spec, tensors: ([], []))
    monkeypatch.setattr(L, "alloc_global_scratch", lambda spec: None)
    monkeypatch.setattr(L, "kernel_args_from_spec", lambda *a: [])
    devs = iter([0.0, 0.5 if diverge else 1e-4])

    def deviation(ref, cur):
        events.append("compare")
        return next(devs)

    from ptx_anneal.target import ptx as ptx_target

    monkeypatch.setattr(ptx_target, "relative_deviation", deviation)
    monkeypatch.delenv("PTX_ANNEAL_VALIDATOR", raising=False)
    monkeypatch.delenv("PTX_ANNEAL_REL_TOL", raising=False)

    def measure(kernel, *a, **kw):
        events.append(("bench", kernel.name))
        return {"base": 0.3, "/x.acf": 0.2}[kernel.name]

    monkeypatch.setattr(L, "bench_ms_cudagraph", measure)
    report = replay.replay_acf(task, "/ptxas", "/x.acf", warmup=1, rep=10)
    assert events == [
        ("launch", "base"),
        ("launch", "/x.acf"),
        "compare",
        "compare",
        ("bench", "base"),
        ("bench", "/x.acf"),
        ("unload", "/x.acf"),
        ("unload", "base"),
    ]
    assert (report.baseline_ms, report.acf_ms) == (0.3, 0.2)
    assert report.numerics_ok is not diverge
    assert report.validator == "RelTolValidator(rel_tol=0.01)"
    assert report.speedup == pytest.approx(1.5)


def test_replay_acf_unloads_baseline_when_acf_compile_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)))
    task = Task("/task", "PTX", {"entry": "k", "arch": "sm_100a", "shared": 0, "grid": [1], "block": [1]})
    kernel = SimpleNamespace(launch=lambda *a: None, unload=Mock())

    def compile_ptx(ptx, ptxas, arch, acf_path=None):
        if acf_path:
            raise RuntimeError("ptxas rejected controls")
        return b"cubin"

    monkeypatch.setattr(L, "ptxas_compile", compile_ptx)
    monkeypatch.setattr(L, "load_cubin", lambda *a, **kw: kernel)
    monkeypatch.setattr(L, "alloc_tensors", lambda *a, **kw: [])
    monkeypatch.setattr(L, "build_tensormaps", lambda *a: ([], []))
    monkeypatch.setattr(L, "alloc_global_scratch", lambda spec: None)
    monkeypatch.setattr(L, "kernel_args_from_spec", lambda *a: [])
    with pytest.raises(RuntimeError, match="rejected controls"):
        replay.replay_acf(task, "/ptxas", "/bad.acf")
    kernel.unload.assert_called_once()


def _store_with_sample_acf(root, version):
    from ptx_anneal.store import LocalStore
    from ptx_anneal.target import load_target

    t = load_target("ptx")
    ir = (Path(SAMPLE_TASK) / "kernel.ptx").read_text()
    return LocalStore(str(root)).write("ptx", t.arch(ir), t.ir_hash(ir), b"acf", {}, toolchain_version=version)


@pytest.mark.parametrize("explicit_root", [True, False])
def test_replay_cli_store_uses_acf_for_task_and_ptxas(monkeypatch, capsys, tmp_path, explicit_root):
    path = _store_with_sample_acf(tmp_path / "store", "13.3")
    _store_with_sample_acf(tmp_path / "store", "13.2")  # another toolchain's ACF must not be picked
    monkeypatch.setenv("COMPILE_IQ_STORE", str(tmp_path / "store" if not explicit_root else tmp_path / "other"))
    run = Mock(return_value=_report())
    monkeypatch.setattr(replay, "replay_acf", run)
    monkeypatch.setattr(cli.provision, "ptxas_version", lambda p: "13.3")
    store_args = ["--store", str(tmp_path / "store")] if explicit_root else ["--store"]
    assert cli.main(["replay", "--task", SAMPLE_TASK, "--ptxas", "/ptxas", *store_args]) == 0
    assert run.call_args.args[2] == path
    assert f"acf: {path} (from store " in capsys.readouterr().out


def test_replay_cli_store_miss_lists_other_ptxas_versions(monkeypatch, capsys, tmp_path):
    _store_with_sample_acf(tmp_path, "13.2")
    monkeypatch.setattr(replay, "replay_acf", Mock(side_effect=AssertionError("replay started")))
    monkeypatch.setattr(cli.provision, "ptxas_version", lambda p: "13.3")
    assert cli.main(["replay", "--task", SAMPLE_TASK, "--ptxas", "/p", "--store", str(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "no ACF in store" in err and "at ptxas 13.3; it has ACFs for ptxas 13.2 only" in err


def test_replay_cli_acf_and_store_are_exclusive(capsys):
    with pytest.raises(SystemExit):
        cli.main(["replay", "--task", SAMPLE_TASK, "--acf", "x.acf", "--store"])
    assert "not allowed with" in capsys.readouterr().err


def test_replay_cli_resolves_task_name_under_task_root(monkeypatch, capsys, tmp_path):
    shutil.copytree(SAMPLE_TASK, tmp_path / "6cb1f53d5136f5b8")
    monkeypatch.setenv("COMPILE_IQ_TASK_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path.parent)
    run = Mock(return_value=0.25)
    monkeypatch.setattr(replay, "replay", run)
    monkeypatch.setattr(cli.provision, "ptxas_version", lambda p: "13.3")
    assert cli.main(["replay", "--task", "6cb1f53d5136f5b8", "--ptxas", "/ptxas"]) == 0
    assert run.call_args.args[0].dir == str(tmp_path / "6cb1f53d5136f5b8")
    assert f"task: 6cb1f53d5136f5b8 -> {tmp_path / '6cb1f53d5136f5b8'}" in capsys.readouterr().out
