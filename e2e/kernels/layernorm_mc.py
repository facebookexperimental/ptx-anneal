# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Kernel: TLX multi-CTA LayerNorm, **no warp specialization, no tensor cores**.

Thin wrapper over ``...tlx.tutorials.blackwell-multi-cta-layernorm_test`` (a dashed module name, so
it is reached with ``import_module`` rather than an ``import`` statement).

The partner of ``layernorm_ws``: same algorithm and the same multi-CTA cluster reduction, but
without the ``tlx.async_tasks`` warp specialization. Together the pair isolates warp specialization
as a single variable, with tcgen5 and TMA absent from both.

Autotuned upstream; pinned here with ``pin_autotuner``.
"""

import importlib

import torch

from ..registry import KernelSpec, register
from . import pin_autotuner

# pyre-ignore[21]: dashed tlx tutorial module, resolvable only by name.
_tut = importlib.import_module("triton.language.extra.tlx.tutorials.blackwell-multi-cta-layernorm_test")

_M, _N = 4096, 1024
_EPS = 1e-5

FINGERPRINT_CONFIG = pin_autotuner(_tut.kernel_layernorm_multi_cta, prune_args={"M": _M, "N": _N})
FINGERPRINT_CONFIG["features"] = ["cluster", "multi_cta_reduction"]
FINGERPRINT_CONFIG["absent"] = ["tcgen5", "tma", "warp_specialization"]


def _inputs():
    x = torch.randn((_M, _N), device="cuda", dtype=torch.float16)
    w = torch.randn((_N,), device="cuda", dtype=torch.float16)
    b = torch.randn((_N,), device="cuda", dtype=torch.float16)
    return (x, w, b)


def _run(x, w, b):
    return _tut.multi_cta_layernorm(x, w, b, eps=_EPS)[0]


def _ref(x, w, b):
    return torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), _EPS).to(x.dtype)


register(
    KernelSpec(
        name="layernorm_mc",
        make_inputs=_inputs,
        run_fn=_run,
        ref_fn=_ref,
        rtol=5e-2,
        desc="TLX multi-CTA LayerNorm: cluster reduction, NO WS, no tcgen5/TMA (WS control partner)",
    )
)
