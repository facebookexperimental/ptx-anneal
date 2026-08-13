# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The ``Runner`` ABC — the candidate-scoring isolation boundary.

A runner takes a target + task + a candidate artifact and returns its score in ms (lower is
better), or :data:`INVALID` if the candidate failed to assemble/launch, diverged from the
baseline, or exceeded the per-candidate timeout. The local runner isolates each candidate in a
subprocess; a remote runner would distribute scoring over a fleet. The search engine never calls
the target directly — it goes through a runner so a wedging candidate can't take down the search.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

#: Sentinel score for a candidate that could not be validly timed (engines minimize, so +inf).
INVALID = math.inf


class Runner(ABC):
    @abstractmethod
    def score(
        self,
        target,
        task,
        artifact_path: str | None,
        *,
        timeout: float | None = None,
        warmup: int = 100,
        rep: int = 200,
    ) -> float:
        """Score one candidate. Returns ms (lower better) or :data:`INVALID`. Never raises for a bad
        candidate — a candidate failure is a score, not an error."""
