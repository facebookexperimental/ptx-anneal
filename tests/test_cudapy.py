# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for CudaPyRunner stdout parsing + load_backend import-path resolution (no GPU)."""

import math
import subprocess
from types import SimpleNamespace

import pytest

from ptx_anneal.run import CudaPyRunner
from ptx_anneal.run.base import INVALID
from ptx_anneal.search import load_backend


# --- load_backend import-path ("module:Class") --------------------------------------------------
def test_load_backend_import_path_resolves_class():
    # The ":" form imports an arbitrary backend not pip-installed (e.g. an engine on PYTHONPATH).
    b = load_backend("ptx_anneal.search.baseline:BaselineSearch")
    assert b.name == "baseline"


def test_load_backend_import_path_bad_spec_errors():
    with pytest.raises(RuntimeError) as ei:
        load_backend("ptx_anneal.search.baseline:DoesNotExist")
    assert "import path" in str(ei.value)


def test_load_backend_unknown_name_still_errors_clearly():
    with pytest.raises(RuntimeError) as ei:
        load_backend("nope")
    assert "bring-your-own" in str(ei.value)


# --- CudaPyRunner stdout-protocol parsing -------------------------------------------------------
def _runner_with_fake(monkeypatch, *, stdout=None, raise_timeout=False):
    r = CudaPyRunner(worker_python="/bin/true")

    # **kwargs, not a fixed signature: this stands in for subprocess.run, so it must tolerate the
    # caller passing kwargs the test doesn't care about (e.g. an explicit check=False).
    def fake_run(cmd, *, timeout=None, **kwargs):
        if raise_timeout:
            raise subprocess.TimeoutExpired(cmd, timeout)
        return SimpleNamespace(stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return r


def test_cudapy_runner_parses_ms_amid_noise(monkeypatch):
    r = _runner_with_fake(monkeypatch, stdout="some import banner\nMS 1.5\ntrailing\n")
    ms = r.score(None, SimpleNamespace(dir="/tmp/task"), None, warmup=1, rep=1)
    assert ms == 1.5


def test_cudapy_runner_invalid_when_no_ms_line(monkeypatch):
    r = _runner_with_fake(monkeypatch, stdout="INVALID\n")
    ms = r.score(None, SimpleNamespace(dir="/tmp/task"), "acf", warmup=1, rep=1)
    assert math.isinf(ms) and ms == INVALID


def test_cudapy_runner_invalid_on_unparseable_token(monkeypatch):
    r = _runner_with_fake(monkeypatch, stdout="MS not_a_number\n")
    ms = r.score(None, SimpleNamespace(dir="/tmp/task"), None, warmup=1, rep=1)
    assert ms == INVALID


def test_cudapy_runner_invalid_on_timeout(monkeypatch):
    r = _runner_with_fake(monkeypatch, raise_timeout=True)
    ms = r.score(None, SimpleNamespace(dir="/tmp/task"), None, timeout=0.01, warmup=1, rep=1)
    assert ms == INVALID
