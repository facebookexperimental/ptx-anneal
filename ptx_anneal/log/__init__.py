# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Logging — structured observability for a tuning run (events, metrics, timed blocks).

:class:`~ptx_anneal.log.base.Logger` is the ABC. The OSS package ships
:class:`~ptx_anneal.log.base.NullLogger` (the default — no-ops) and
:class:`~ptx_anneal.log.base.StderrLogger` (JSON-lines for local development). Remote
loggers (e.g. shipping to a metrics service) are injected out-of-tree and implement the
same ABC.
"""

from .base import Logger, NullLogger, StderrLogger

__all__ = ["Logger", "NullLogger", "StderrLogger"]
