# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the ptx target: hashing/parsing and the frontend normalize-contract (GPU-light)."""

import importlib.util
import os

import pytest

from ptx_anneal import ptx_launch
from ptx_anneal.target.ptx import PtxTarget, normalize_ptx, ptx_ir_hash

SAMPLE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_tasks", "sample_task")
EXPECTED_HASH = "8455732ca25e114a10a065ef0f2c3345916d85a69ed4776973c77552dfab5415"

# The frontend's normalizer (fbtriton). The factory's key MUST match it byte-for-byte or a tuned
# artifact would never be found at consume time. Skip if the frontend checkout isn't present.
_FBTRITON_STORE = "/data/users/daohang/fbtriton/third_party/compile_iq/compile_iq/store.py"


def _ptx():
    with open(os.path.join(SAMPLE, "kernel.ptx")) as f:
        return f.read()


def test_ir_hash_matches_fixture():
    assert ptx_ir_hash(_ptx()) == EXPECTED_HASH
    assert PtxTarget().ir_hash(_ptx()) == EXPECTED_HASH


def test_arch_and_entry_parse():
    ptx = _ptx()
    assert PtxTarget().arch(ptx) == "sm_100a"
    name, kinds = ptx_launch.parse_entry(ptx)
    assert name == "add_kernel"
    assert kinds == ["ptr", "ptr", "ptr", "i32", "ptr", "ptr"]


@pytest.mark.skipif(not os.path.exists(_FBTRITON_STORE), reason="frontend (fbtriton) checkout not present")
def test_normalize_contract_matches_frontend():
    spec = importlib.util.spec_from_file_location("_fbtriton_store", _FBTRITON_STORE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ptx = _ptx()
    assert normalize_ptx(ptx) == mod.normalize_ptx(ptx)
    assert ptx_ir_hash(ptx) == mod.ptx_sha256(ptx)
