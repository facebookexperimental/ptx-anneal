# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Search backends — the optimization engine that proposes candidate artifacts.

:class:`~ptx_anneal.search.base.SearchBackend` is the ABC. The built-in ``baseline`` backend
(:class:`~ptx_anneal.search.baseline.BaselineSearch`) proposes nothing — it just measures the
untuned baseline, so a fresh install runs end-to-end with no engine. Real engines (NVIDIA
CompileIQ) are **bring-your-own**, registered out-of-tree via the ``ptx_anneal.search_backends``
entry-point group; they are never vendored here.
"""

from .base import SearchBackend, SearchResult, available_backends, load_backend

__all__ = ["SearchBackend", "SearchResult", "load_backend", "available_backends"]
