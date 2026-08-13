# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the GPU-light bits of ptx_launch: auto-N math + kernelParams building (no GPU)."""

import json
from types import SimpleNamespace

import pytest

from ptx_anneal.ptx_launch import _build_kernel_params, _pick_n, build_spec


# --- CUDA-graph unroll count (auto-N) -----------------------------------------------------------
def test_pick_n_scales_region_to_rep_ms():
    assert _pick_n(0.5, 100.0) == 200  # 100ms / 0.5ms
    assert _pick_n(2.0, 100.0) == 50


def test_pick_n_floor_and_degenerate():
    assert _pick_n(1000.0, 100.0) == 1  # est bigger than target -> at least 1
    assert _pick_n(0.0, 100.0) == 1  # nonpositive est -> 1
    assert _pick_n(float("inf"), 100.0) == 1
    assert _pick_n(float("nan"), 100.0) == 1


def test_pick_n_clamped_to_max():
    assert _pick_n(1e-12, 100.0) == 100000  # tiny est -> clamp at n_max


# --- kernelParams building --------------------------------------------------------------------
def test_build_kernel_params_len_and_holders():
    args = [("ptr", 0x1000), ("i32", 7), ("i64", 9), ("f32", 1.5), ("null",), ("tma", 0x2000)]
    arr, holders = _build_kernel_params(args)
    assert len(arr) == len(args)
    # ptr/i32/i64/f32/null each back a value holder; tma is passed by value (no holder).
    assert len(holders) == 5
    assert arr[5] == 0x2000  # tma slot points straight at the descriptor address


def test_build_kernel_params_rejects_unknown_kind():
    with pytest.raises(ValueError):
        _build_kernel_params([("weird", 1)])


# --- build_spec cluster handling (ctas_per_cga; no GPU) ----------------------------------------
# Minimal kernel: one user pointer param + the two trailing scratch pointers.
_PTX = """//
.version 8.3
.target sm_90a
.address_size 64

.visible .entry k(
    .param .u64 .ptr .global .align 16 p0,
    .param .u64 .ptr .global .align 16 p1,
    .param .u64 .ptr .global .align 16 p2
)
{
    ret;
}
"""
_ARGS = [("tensor", {"id": 0, "shape": [16], "dtype": "float32", "strides": [1]})]


def _md(**over):
    base = {
        "global_scratch_size": 0,
        "profile_scratch_size": 0,
        "num_ctas": 1,
        "ctas_per_cga": None,
        "tensordesc_meta": [],
        "num_warps": 4,
        "shared": 0,
    }
    base.update(over)
    return SimpleNamespace(**base)


def test_build_spec_records_ctas_per_cga_cluster():
    spec = build_spec(_PTX, _md(ctas_per_cga=(2, 1, 1)), (4, 1, 1), _ARGS, "ptxas")
    assert spec["cluster"] == [2, 1, 1]
    assert spec["arch"] == "sm_90a"
    assert spec["grid"] == [4, 1, 1]


def test_build_spec_no_cluster_key_when_unclustered():
    spec = build_spec(_PTX, _md(ctas_per_cga=None), (4, 1, 1), _ARGS, "ptxas")
    assert "cluster" not in spec


def test_build_spec_reads_ctas_per_cga_only_ignoring_cluster_dims():
    # The decision under test: cluster shape comes from ctas_per_cga alone. A stray cluster_dims must
    # NOT be read (the nvidia backend mirrors ctas_per_cga into cluster_dims, so ctas_per_cga is the
    # single source of truth here).
    spec = build_spec(_PTX, _md(ctas_per_cga=None, cluster_dims=(2, 1, 1)), (4, 1, 1), _ARGS, "ptxas")
    assert "cluster" not in spec


def test_build_spec_rejects_num_ctas_gt_1():
    with pytest.raises(NotImplementedError):
        build_spec(_PTX, _md(num_ctas=2), (4, 1, 1), _ARGS, "ptxas")


def test_build_spec_rejects_grid_not_divisible_by_cluster():
    with pytest.raises(NotImplementedError):
        build_spec(_PTX, _md(ctas_per_cga=(2, 1, 1)), (3, 1, 1), _ARGS, "ptxas")


def test_build_spec_cluster_survives_json_roundtrip():
    # The spec IS the ACF task's spec.json, so the cluster must survive serialization intact.
    spec = build_spec(_PTX, _md(ctas_per_cga=(2, 1, 1)), (4, 1, 1), _ARGS, "ptxas")
    assert json.loads(json.dumps(spec))["cluster"] == [2, 1, 1]
