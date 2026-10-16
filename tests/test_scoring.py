# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CLI scorer selection and torch candidate validation using CPU-only substitutes."""

import sys
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ptx_anneal import cli, scoring
from ptx_anneal import ptx_launch as L
from ptx_anneal.target.base import ScoreInvalid
from ptx_anneal.target.ptx import PtxTarget
from ptx_anneal.task import Task


def _spec(**tensor_overrides):
    return {
        "entry": "k",
        "arch": "sm_100a",
        "shared": 100,
        "grid": [4, 1, 1],
        "block": [256, 1, 1],
        "ptxas": "/stale/captured/ptxas",
        "tensors": [{"dtype": "float32", "shape": [2, 3], "strides": [3, 1], **tensor_overrides}],
        "args": [{"t": 0}],
    }


@pytest.mark.parametrize(
    "tensor, torch_needed",
    [
        ({}, False),
        ({"dtype": "float16"}, False),
        ({"dtype": "bfloat16"}, True),
        ({"dtype": "float64"}, True),
        ({"strides": [1, 2]}, True),
        ({"shape": [2, 1, 3], "strides": [3, 99, 1]}, False),
    ],
)
def test_tensor_scorer_selection(tensor, torch_needed):
    assert scoring.needs_torch(_spec(**tensor)) is torch_needed


def test_tma_uses_torch_even_for_float32():
    spec = _spec()
    spec["args"] = [{"tma": 0}]
    assert scoring.needs_torch(spec)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_dispatch_preserves_acf_timing_and_ptxas_pin(monkeypatch, dtype):
    task = Task("/task", "PTX", _spec(dtype=dtype))
    monkeypatch.setattr(scoring.task_mod, "load", lambda path: task)
    gpu_score = Mock(return_value=0.25)
    if dtype == "bfloat16":
        monkeypatch.setattr(PtxTarget, "score_one", gpu_score)
    else:
        monkeypatch.setitem(sys.modules, "torch", None)
        monkeypatch.setitem(sys.modules, "ptx_anneal.run.cudapy_worker", SimpleNamespace(score=gpu_score))
    assert scoring.score("/task", "/candidate.acf", 2, 30, "do_bench", ptxas="/pinned/ptxas") == 0.25
    if dtype == "bfloat16":
        gpu_score.assert_called_once_with(
            task, "/candidate.acf", warmup=2, rep=30, bench="do_bench", ptxas="/pinned/ptxas"
        )
    else:
        gpu_score.assert_called_once_with("/task", "/candidate.acf", 2, 30, "do_bench", ptxas="/pinned/ptxas")


def test_cli_baseline_and_candidates_share_dispatch(monkeypatch):
    score = Mock(return_value=0.5)
    monkeypatch.setattr(scoring, "score", score)
    monkeypatch.setenv("TRITON_PTXAS_BLACKWELL_PATH", "")
    assert cli._score_one("/task", None, 2, 30, "do_bench", "/pin") == 0.5
    assert (
        cli.main(
            [
                "_score",
                "--task",
                "/task",
                "--acf",
                "/cand",
                "--ptxas",
                "/pin",
                "--warmup",
                "2",
                "--rep",
                "30",
                "--bench",
                "do_bench",
            ]
        )
        == 0
    )
    assert score.call_args_list[0].args == ("/task", None, 2, 30, "do_bench")
    assert score.call_args_list[1].args == ("/task", "/cand", 2, 30, "do_bench")
    assert all(call.kwargs == {"ptxas": "/pin"} for call in score.call_args_list)


class Tensor:
    """The few tensor operations used by validation; never imports torch or numpy."""

    def __init__(self, values):
        self.values = values

    def float(self):
        return self

    def abs(self):
        return Tensor([abs(v) for v in self.values])

    def max(self):
        return Tensor([max(self.values)])

    def item(self):
        return self.values[0]

    def __sub__(self, other):
        return Tensor([a - b for a, b in zip(self.values, other.values, strict=True)])


@pytest.fixture
def torch_launch(monkeypatch):
    state = SimpleNamespace(kernels=[], compiled=[], descriptors=[], events=[], measured=[], candidate=[2.0, 4.0])
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: state.events.append("sync")))
    )
    monkeypatch.delenv("PTX_ANNEAL_VALIDATOR", raising=False)
    monkeypatch.delenv("PTX_ANNEAL_REL_TOL", raising=False)

    def compile_ptx(ptx, compiler, *, arch, acf_path):
        state.compiled.append((compiler, acf_path))
        return acf_path or "baseline"

    def load(cubin, entry, shared, *, cluster):
        assert cluster == [2, 1, 1]
        kernel = SimpleNamespace(unload=Mock(), tensors=[])

        def launch(*args):
            assert state.events[-1] == "sync"
            kernel.tensors[0].values = [2.0, 4.0] if cubin == "baseline" else state.candidate
            state.events.append("launch")

        kernel.launch = launch
        state.kernels.append(kernel)
        return kernel

    def alloc(spec, seed):
        assert seed == 0
        tensors = [Tensor([0.0, 0.0])]
        state.kernels[-1].tensors = tensors
        return tensors

    def maps(spec, tensors):
        descriptor = Tensor([])  # weakref-able object; must remain alive through benchmark
        state.descriptors.append(weakref.ref(descriptor))
        return [descriptor], [1234]

    def measure(*args, **kwargs):
        assert state.descriptors[-1]() is not None
        state.measured.append(kwargs)
        return 0.125

    monkeypatch.setattr(L, "ptxas_compile", compile_ptx)
    monkeypatch.setattr(L, "load_cubin", load)
    monkeypatch.setattr(L, "alloc_tensors", alloc)
    monkeypatch.setattr(L, "build_tensormaps", maps)
    monkeypatch.setattr(L, "kernel_args_from_spec", lambda *args: [("tma", 1234)])
    monkeypatch.setattr(L, "bench_ms_cudagraph", measure)
    monkeypatch.setattr(L, "bench_ms_do_bench", measure)
    return state


@pytest.mark.parametrize("bench", ["cudagraph", "do_bench"])
def test_torch_scorer_validates_and_honors_launch_and_timing(torch_launch, bench):
    spec = {**_spec(dtype="bfloat16"), "cluster": [2, 1, 1]}
    assert (
        PtxTarget().score_one(Task("/task", "PTX", spec), "/candidate.acf", ptxas="/pin", bench=bench, warmup=2, rep=40)
        == 0.125
    )
    assert torch_launch.compiled == [("/pin", None), ("/pin", "/candidate.acf")]
    assert torch_launch.measured == [{"warmup": 2, "reps": 4 if bench == "cudagraph" else 40}]
    assert all(k.unload.call_count == 1 for k in torch_launch.kernels)


@pytest.mark.parametrize("bad_output", [[2.0, 8.0], [float("nan"), 4.0]])
def test_torch_scorer_rejects_incorrect_bf16_candidate(torch_launch, bad_output):
    torch_launch.candidate = bad_output
    spec = {**_spec(dtype="bfloat16"), "cluster": [2, 1, 1]}
    with pytest.raises(ScoreInvalid, match="candidate diverged"):
        PtxTarget().score_one(Task("/task", "PTX", spec), "/candidate.acf", ptxas="/pin")
    assert not torch_launch.measured
    assert all(k.unload.call_count == 1 for k in torch_launch.kernels)
