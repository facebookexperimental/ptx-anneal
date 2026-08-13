# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Logger ABC + built-in implementations (NullLogger / StderrLogger / timing)."""

import io
import json

from ptx_anneal.log import Logger, NullLogger, StderrLogger


class RecordingLogger(Logger):
    """Captures the event/metric stream for assertions (mirrors what an out-of-tree logger sees)."""

    def __init__(self):
        self.events = []  # (kind, fields)
        self.metrics = []  # (name, value, unit)

    def event(self, kind, **fields):
        self.events.append((kind, fields))

    def metric(self, name, value, unit=""):
        self.metrics.append((name, value, unit))


def test_null_logger_no_ops():
    log = NullLogger()
    log.event("anything", a=1)
    log.metric("m", 1.0, "ms")
    with log.timing("phase"):
        pass  # no exception, nothing recorded anywhere


def test_stderr_logger_emits_jsonl():
    buf = io.StringIO()
    log = StderrLogger(stream=buf)
    log.event("admitted", arch="sm_100a", win=0.25)
    log.metric("search_win", 0.25, "ratio")
    lines = [json.loads(line) for line in buf.getvalue().splitlines()]
    assert lines[0] == {"type": "event", "kind": "admitted", "arch": "sm_100a", "win": 0.25}
    assert lines[1] == {"type": "metric", "name": "search_win", "value": 0.25, "unit": "ratio"}


def test_timing_emits_ms_metric():
    log = RecordingLogger()
    with log.timing("search"):
        pass
    assert len(log.metrics) == 1
    name, value, unit = log.metrics[0]
    assert name == "search_ms" and unit == "ms" and value >= 0.0


def test_timing_emits_metric_even_on_exception():
    log = RecordingLogger()
    try:
        with log.timing("phase"):
            raise ValueError("boom")
    except ValueError:
        pass
    assert [m[0] for m in log.metrics] == ["phase_ms"]
