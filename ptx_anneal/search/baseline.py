# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The built-in ``baseline`` search backend — proposes no candidates.

It exists so a fresh install runs the full collect -> tune -> store flow end-to-end with **no
engine**: ``tune`` measures the untuned baseline (the factory does that), this backend admits
nothing, and the run exits cleanly ("no candidate admitted"). It is the engine-free default and the
smoke/CI target. Swap in a real engine (bring-your-own) to actually search for a faster ACF.
"""

from __future__ import annotations

from ..run.base import INVALID
from .base import SearchBackend, SearchResult


class BaselineSearch(SearchBackend):
    name = "baseline"

    def search(self, task, objective, *, config: dict | None = None) -> SearchResult:
        # No candidates proposed -> objective is never called -> nothing to admit.
        return SearchResult(
            artifact=None,
            score=INVALID,
            evaluated=0,
            meta={"engine": "baseline", "note": "no candidates proposed (untuned baseline only)"},
        )
