# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Targets — the per-IR-kind backend that assembles, launches, and scores a candidate.

:class:`~ptx_anneal.target.base.Target` is the ABC. v1 ships only the ``ptx`` target
(:class:`~ptx_anneal.target.ptx.PtxTarget`); other targets (e.g. AMD ``gcn``) are future
registered plugins discovered via the ``ptx_anneal.targets`` entry-point group.
"""

from .base import Target, available_targets, load_target

__all__ = ["Target", "load_target", "available_targets"]
