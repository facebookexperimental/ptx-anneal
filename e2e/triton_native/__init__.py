# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The triton-native channel: an on-demand runway for scoring a kernel through **fbtriton** itself.

Two modules, split by what they are allowed to import:

* :mod:`e2e.triton_native.runway` -- the orchestrator. **Pure stdlib.** It never imports torch or
  fbtriton, so it can report itself unavailable on a box that has neither instead of failing to load.
* :mod:`e2e.triton_native.score` -- the per-candidate scorer, run as a subprocess. The *only* place
  torch/fbtriton are imported, and only after the ACF environment is in place.

The frontend is **fbtriton** specifically. ``PTXAS_OPTIONS`` is its knob (upstream Triton spells
nothing equivalent), so on the wrong Triton the ACF is ignored and every candidate is scored as the
untuned kernel -- which is why the scorer verifies application rather than assuming it.

The module split is not tidiness: ``PTXAS_OPTIONS`` is read when the nvidia backend module is
imported, so the ACF must be in the environment *before* the first ``import triton`` in the process.
A process that has already imported triton cannot be given a different ACF. One candidate, one
process.

Never packaged -- it lives under ``e2e/``, which is excluded from both the wheel and the sdist.
"""
