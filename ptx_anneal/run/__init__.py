# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Runners — where/how a candidate is scored (the isolation boundary).

:class:`~ptx_anneal.run.base.Runner` is the ABC; :class:`~ptx_anneal.run.local.LocalRunner`
scores each candidate in a throwaway subprocess on the current host so a candidate that wedges
the GPU is killed at a timeout rather than hanging the search.
:class:`~ptx_anneal.run.cudapy.CudaPyRunner` is a torch-free variant (cuda-python + numpy worker)
for environments without a working torch. Remote runners (e.g. fleet execution) are injected
out-of-tree.
"""

from .base import INVALID, Runner
from .cudapy import CudaPyRunner
from .local import LocalRunner

__all__ = ["Runner", "LocalRunner", "CudaPyRunner", "INVALID"]
