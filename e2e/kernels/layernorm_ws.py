# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Kernel: TLX multi-CTA warp-specialized LayerNorm -- **warp specialization with no tensor cores**.

Thin wrapper over the upstream TLX tutorial
(``...tlx.tutorials.blackwell_multi_cta_layernorm_ws_test.multi_cta_layernorm_ws``).

This is the *control* for warp specialization. Every Blackwell TLX kernel that fails under an
applied ACF so far has been a GEMM or attention kernel, i.e. tcgen5 plus warp specialization plus
(usually) TMA -- three suspects moving together. This kernel keeps the warp specialization
(``tlx.async_tasks``, a 2-workgroup producer/consumer split, cluster reduction) and drops the other
two: LayerNorm has no ``tl.dot``, so no tcgen5 and no TMA. If ACFs break warp specialization as
such, this must fail; if it passes, warp specialization is exonerated and the suspect list shortens.

Autotune is pinned via ``pin_autotuner`` rather than by editing the tutorial -- see that helper for
why narrowing the config list to one is equivalent to pinning.
"""

import torch

# pyre-ignore[21]: the tlx tutorials ship in the triton-beta runtime but are not in pyre's source DB.
from triton.language.extra.tlx.tutorials.blackwell_multi_cta_layernorm_ws_test import (
    kernel_layernorm_multi_cta_ws,
    multi_cta_layernorm_ws,
)

from ..registry import KernelSpec, register
from . import pin_autotuner

# Narrow-N is the path this WS kernel exists for; N must divide into num_reduction_ctas=2 cleanly or
# the tutorial's own prune hook drops every config.
_M, _N = 4096, 1024
_EPS = 1e-5

# BLOCK_SIZE_M=4 with num_warps=8 / num_reduction_ctas=2 is the tutorial's own B200 sweet spot.
FINGERPRINT_CONFIG = pin_autotuner(
    kernel_layernorm_multi_cta_ws, match={"BLOCK_SIZE_M": 4}, prune_args={"M": _M, "N": _N}
)
FINGERPRINT_CONFIG["features"] = ["warp_specialization", "cluster", "cp.async"]
FINGERPRINT_CONFIG["absent"] = ["tcgen5", "tma"]


def _inputs():
    x = torch.randn((_M, _N), device="cuda", dtype=torch.float16)
    w = torch.randn((_N,), device="cuda", dtype=torch.float16)
    b = torch.randn((_N,), device="cuda", dtype=torch.float16)
    return (x, w, b)


def _run(x, w, b):
    # The tutorial returns (out, mean, rstd); the harness compares one tensor, and `out` is the one
    # a miscompile would corrupt.
    return multi_cta_layernorm_ws(x, w, b, eps=_EPS)[0]


def _ref(x, w, b):
    return torch.nn.functional.layer_norm(x.float(), (x.shape[-1],), w.float(), b.float(), _EPS).to(x.dtype)


register(
    KernelSpec(
        name="layernorm_ws",
        make_inputs=_inputs,
        run_fn=_run,
        ref_fn=_ref,
        rtol=5e-2,
        desc="TLX multi-CTA WS LayerNorm: warp specialization WITHOUT tcgen5 or TMA (WS control)",
    )
)
