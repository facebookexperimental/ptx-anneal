# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The ``Logger`` ABC — structured observability for a tuning run.

A logger receives three kinds of signal from the factory orchestrator:

- :meth:`Logger.event` — a discrete, named occurrence with arbitrary fields
  (e.g. ``baseline`` measured, a candidate ``admitted`` / ``no_candidate``).
- :meth:`Logger.metric` — a single numeric measurement with a unit
  (e.g. ``search_win`` as a ratio, ``evaluated`` as a count).
- :meth:`Logger.timing` — a context manager that times a block and emits its
  duration as a ``*_ms`` metric.

Logging is the generic extension point a downstream integrator implements to ship
these signals somewhere (a metrics service, a log pipeline, ...). The OSS package
ships only :class:`NullLogger` (the default — no-ops) and :class:`StderrLogger` (a
JSON-lines dump for local development); remote implementations are injected
out-of-tree, like the runner and the store.

Subclasses only implement :meth:`event` and :meth:`metric`; :meth:`timing` is
provided concretely on top of :meth:`metric`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager


class Logger(ABC):
    """Structured observability sink: events, metrics, and timed blocks."""

    @abstractmethod
    def event(self, kind: str, **fields) -> None:
        """Record a named event with arbitrary structured ``fields``."""

    @abstractmethod
    def metric(self, name: str, value: float, unit: str = "") -> None:
        """Record one numeric measurement (``unit`` is free-form, e.g. ``ms``/``ratio``/``count``)."""

    @contextmanager
    def timing(self, name: str):
        """Time the wrapped block and emit ``metric(f"{name}_ms", elapsed_ms, "ms")`` on exit.

        The metric is emitted even if the block raises, so a failed/aborted phase is still
        accounted for; the exception then propagates as usual.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self.metric(f"{name}_ms", (time.perf_counter() - start) * 1e3, "ms")


class NullLogger(Logger):
    """The default logger: discards everything. ``timing`` still runs (its metric is dropped)."""

    def event(self, kind: str, **fields) -> None:
        pass

    def metric(self, name: str, value: float, unit: str = "") -> None:
        pass


class StderrLogger(Logger):
    """A development logger: one JSON object per line on stderr (``-v`` / ``--verbose``)."""

    def __init__(self, stream=None):
        import sys

        self._stream = stream if stream is not None else sys.stderr

    def _emit(self, record: dict) -> None:
        import json

        self._stream.write(json.dumps(record, default=str) + "\n")
        self._stream.flush()

    def event(self, kind: str, **fields) -> None:
        self._emit({"type": "event", "kind": kind, **fields})

    def metric(self, name: str, value: float, unit: str = "") -> None:
        self._emit({"type": "metric", "name": name, "value": value, "unit": unit})
