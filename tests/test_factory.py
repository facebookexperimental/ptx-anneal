# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Factory + search-registry tests using fakes (no GPU)."""

import os

import pytest

from ptx_anneal import factory, task
from ptx_anneal.log import Logger
from ptx_anneal.run.base import INVALID, Runner
from ptx_anneal.search import SearchResult, available_backends, load_backend
from ptx_anneal.store import LocalStore


class RecordingLogger(Logger):
    """Captures the factory's event/metric stream for assertions."""

    def __init__(self):
        self.events = []  # (kind, fields)
        self.metrics = []  # (name, value, unit)

    def event(self, kind, **fields):
        self.events.append((kind, fields))

    def metric(self, name, value, unit=""):
        self.metrics.append((name, value, unit))

    def kinds(self):
        return [k for k, _ in self.events]

    def metric_names(self):
        return [n for n, _, _ in self.metrics]


SAMPLE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_tasks", "sample_task")
HASH = "6cb1f53d5136f5b8e3e3c25727534f6d006f6564d26b0e0b51cfd4826e254000"


class FakeTarget:
    name = "ptx"

    def ir_hash(self, ir):
        return HASH

    def arch(self, ir):
        return "sm_100a"


class FakeRunner(Runner):
    """Returns a fixed baseline for the no-ACF run and a fixed candidate time otherwise."""

    def __init__(self, baseline_ms, candidate_ms):
        self.baseline_ms, self.candidate_ms = baseline_ms, candidate_ms

    def score(self, target, task_, artifact_path, *, timeout=None, warmup=100, rep=200):
        return self.baseline_ms if artifact_path is None else self.candidate_ms


class FakeBackend:
    name = "fake"

    def __init__(self, artifact, score):
        self.artifact, self.score = artifact, score

    def search(self, task_, objective, *, config=None):
        if self.artifact is not None:
            objective(self.artifact)  # exercise the runner path
        return SearchResult(artifact=self.artifact, score=self.score, evaluated=1, meta={"engine": "fake"})


# --- registry ------------------------------------------------------------------------------------
def test_load_baseline_backend():
    b = load_backend("baseline")
    assert b.name == "baseline"
    assert "baseline" in available_backends()


def test_unknown_backend_errors_clearly():
    with pytest.raises(RuntimeError) as ei:
        load_backend("nope")
    msg = str(ei.value)
    assert "bring-your-own" in msg and "baseline" in msg


# --- factory admit logic -------------------------------------------------------------------------
def test_baseline_backend_admits_nothing(tmp_path):
    t = task.load(SAMPLE)
    store = LocalStore(str(tmp_path))
    res = factory.tune(
        t, target=FakeTarget(), store=store, runner=FakeRunner(2.0, 2.0), search=load_backend("baseline")
    )
    assert res["admitted"] is False
    assert res["baseline_ms"] == 2.0
    assert store.list() == []


def test_winning_candidate_admitted(tmp_path):
    t = task.load(SAMPLE)
    store = LocalStore(str(tmp_path))
    res = factory.tune(
        t, target=FakeTarget(), store=store, runner=FakeRunner(2.0, 1.5), search=FakeBackend(b"acf-bytes", 1.5)
    )
    assert res["admitted"] is True
    assert res["search_win"] == pytest.approx(0.25)
    assert store.read("ptx", "sm_100a", HASH) == b"acf-bytes"
    meta = store.read_meta("ptx", "sm_100a", HASH)
    assert meta["baseline_ms"] == 2.0 and meta["best_ms"] == 1.5 and meta["engine"] == "fake"


def test_slower_candidate_not_admitted(tmp_path):
    t = task.load(SAMPLE)
    store = LocalStore(str(tmp_path))
    res = factory.tune(
        t, target=FakeTarget(), store=store, runner=FakeRunner(2.0, 2.5), search=FakeBackend(b"acf-bytes", 2.5)
    )
    assert res["admitted"] is False
    assert store.list() == []


def test_invalid_baseline_admits_nothing(tmp_path):
    t = task.load(SAMPLE)
    store = LocalStore(str(tmp_path))
    res = factory.tune(
        t, target=FakeTarget(), store=store, runner=FakeRunner(INVALID, 1.0), search=FakeBackend(b"acf-bytes", 1.0)
    )
    assert res["admitted"] is False
    assert res["baseline_ms"] is None


# --- logging -------------------------------------------------------------------------------------
def test_logger_records_admitted_run(tmp_path):
    log = RecordingLogger()
    t = task.load(SAMPLE)
    factory.tune(
        t,
        target=FakeTarget(),
        store=LocalStore(str(tmp_path)),
        runner=FakeRunner(2.0, 1.5),
        search=FakeBackend(b"acf-bytes", 1.5),
        logger=log,
    )
    assert log.kinds() == ["baseline", "candidate", "admitted"]
    base_fields = log.events[0][1]
    assert base_fields["ok"] is True and base_fields["baseline_ms"] == 2.0
    cand_fields = log.events[1][1]
    assert cand_fields["ok"] is True and cand_fields["ms"] == 1.5
    assert cand_fields["index"] == 1 and cand_fields["bytes"] == len(b"acf-bytes")
    admit_fields = log.events[2][1]
    assert admit_fields["win"] == pytest.approx(0.25) and admit_fields["best_ms"] == 1.5
    # timings ("baseline_ms"/"search_ms") + evaluated + search_win all flow through metric().
    assert {"baseline_ms", "search_ms", "evaluated", "search_win"} <= set(log.metric_names())


def test_logger_records_no_candidate_run(tmp_path):
    log = RecordingLogger()
    t = task.load(SAMPLE)
    factory.tune(
        t,
        target=FakeTarget(),
        store=LocalStore(str(tmp_path)),
        runner=FakeRunner(2.0, 2.5),
        search=FakeBackend(b"acf-bytes", 2.5),
        logger=log,
    )
    assert log.kinds() == ["baseline", "candidate", "no_candidate"]
    # the candidate ran and validated (finite ms) -- it just didn't beat the baseline.
    assert log.events[1][1]["ok"] is True and log.events[1][1]["ms"] == 2.5
    assert log.events[2][1]["reason"] == "no candidate beat the baseline"
    assert "search_win" not in log.metric_names()
