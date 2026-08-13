# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""ptx-anneal — an offline ptxas tuning framework (the "ACF factory").

Given a fixed compiled kernel (for the ``ptx`` target, a ``kernel.ptx``) and a
source-free launch ``spec``, ptx-anneal drives a bring-your-own search engine over
``ptxas`` advanced controls, validates each candidate against the untuned baseline,
and admits the fastest correct artifact (an ACF, Advanced Control File) to a
content-addressed store keyed by ``(target, arch, ir_hash)``.

This package is the orchestration harness only. The optimization engine (NVIDIA
CompileIQ) is bring-your-own and registered out-of-tree via ``entry_points`` behind
:class:`ptx_anneal.search.base.SearchBackend`; it is never vendored here.

Public surface (the dependency-injected pieces the orchestrator wires together):
    - :mod:`ptx_anneal.task`   — the ACF-task schema (``load`` / ``Task``).
    - :mod:`ptx_anneal.target` — the per-IR backend (``Target`` ABC, ``PtxTarget``).
    - :mod:`ptx_anneal.search` — the search engine (``SearchBackend`` ABC, bring-your-own).
    - :mod:`ptx_anneal.run`    — the candidate-scoring boundary (``Runner`` ABC, ``LocalRunner``).
    - :mod:`ptx_anneal.store`  — the artifact store (``Store`` ABC, ``LocalStore``).
    - :mod:`ptx_anneal.log`    — observability (``Logger`` ABC, ``NullLogger``/``StderrLogger``).
    - :func:`ptx_anneal.factory.tune` — the orchestrator that ties them together.

``Runner``, ``Store``, and ``Logger`` ship a local/null implementation here; remote
implementations (fleet execution, a hosted artifact service, a metrics pipeline) are
injected out-of-tree by constructing them and passing them to :func:`factory.tune`.
"""

__version__ = "0.0.1"
