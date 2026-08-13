# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Candidate validation -- the in-search correctness gate (pluggable).

An ACF only changes ptxas codegen, never the math, so a candidate must reproduce the no-ACF
baseline. A worker computes, per output buffer, a relative deviation
``max|cand - ref| / max(|ref|, eps)`` and hands the list to a :class:`Validator`, which decides
whether the candidate is acceptable (raising :class:`ValidationError` if not). Working on the
*deviations* (plain floats) rather than raw arrays keeps the validator framework-agnostic: the
torch worker and the cuda-python/numpy worker feed the same interface.

The default is :class:`RelTolValidator` (a relative-tolerance bound). Swap in your own rule by
setting ``PTX_ANNEAL_VALIDATOR='module.path:ClassName'`` (it must subclass :class:`Validator`); the
default tolerance is overridable via ``PTX_ANNEAL_REL_TOL``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

DEFAULT_REL_TOL = 1e-2
VALIDATOR_ENV = "PTX_ANNEAL_VALIDATOR"
REL_TOL_ENV = "PTX_ANNEAL_REL_TOL"


class ValidationError(Exception):
    """A candidate's output diverged from the baseline beyond what the validator accepts."""


class Validator(ABC):
    @abstractmethod
    def check(self, deviations: list[float]) -> None:
        """Accept or reject a candidate from its per-output-buffer relative deviations.

        ``deviations[i] = max|cand_i - ref_i| / max(|ref_i|, eps)`` for each compared (float) buffer.
        Raise :class:`ValidationError` to reject; return ``None`` to accept."""


class RelTolValidator(Validator):
    """Accept iff every buffer's relative deviation is finite and within ``rel_tol`` (NaN/inf reject)."""

    def __init__(self, rel_tol: float = DEFAULT_REL_TOL):
        self.rel_tol = rel_tol

    def check(self, deviations: list[float]) -> None:
        for dd in deviations:
            # `dd == dd` is the NaN guard (NaN != NaN); rejects NaN / +inf / over-tolerance.
            if not (dd == dd and dd <= self.rel_tol):
                raise ValidationError(f"candidate diverged (rel={dd} > tol={self.rel_tol})")


def load_validator() -> Validator:
    """The validator to use: ``$PTX_ANNEAL_VALIDATOR`` ('module:Class') if set, else
    :class:`RelTolValidator` with tolerance ``$PTX_ANNEAL_REL_TOL`` (default ``1e-2``)."""
    spec = os.environ.get(VALIDATOR_ENV)
    if spec:
        import importlib

        mod_name, _, cls_name = spec.partition(":")
        try:
            cls = getattr(importlib.import_module(mod_name), cls_name)
        except Exception as e:
            raise RuntimeError(f"failed loading validator from {spec!r}: {e}") from e
        return cls()
    return RelTolValidator(float(os.environ.get(REL_TOL_ENV, DEFAULT_REL_TOL)))
