# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Kernel A: authentic TLX warp-specialized Blackwell GEMM.

Thin wrapper over the proven upstream TLX tutorial kernel
(``triton.language.extra.tlx.tutorials.blackwell_gemm_ws.matmul``) so the smoke
test exercises the exact warp-specialized TMA path magnon tunes in production.
The kernel is hand-tuned (255 regs, no spills), so the factory finds no ptxas
headroom and (without the force-apply knob) the free-win A/B correctly declines
to regress -- a *valid* MISS that still proves the whole collect->consume path.

We PIN one config via ``get_heuristic_config`` so exactly one kernel compiles and
the collector emits a single ptx/spec -- autotuning would emit one task per config
(a config sweep, i.e. the not-yet-shipped "Magnon Plus"), which a smoke test must
not do.
"""

import torch

# pyre-ignore[21]: the tlx tutorials ship in the triton-beta runtime but are not in pyre's source DB.
from triton.language.extra.tlx.tutorials.blackwell_gemm_ws import (
    get_heuristic_config,
    matmul,
)

from ..registry import KernelSpec, register

# Proven collecting/consuming shape (row-major A, row-major B, 16B-aligned TMA).
_M, _N, _K = 1024, 12800, 1152
_NUM_SMS = 148  # B200


def _inputs():
    a = torch.randn((_M, _K), device="cuda", dtype=torch.float16)
    b = torch.randn((_K, _N), device="cuda", dtype=torch.float16)
    return (a, b)


def _run(a, b):
    # Fresh config each call: matmul() pops keys (ctas_per_cga, pre_hook) from it.
    return matmul(a, b, config=get_heuristic_config(_M, _N, _K, num_sms=_NUM_SMS))


register(
    KernelSpec(
        name="ws_gemm",
        make_inputs=_inputs,
        run_fn=_run,
        ref_fn=lambda a, b: torch.matmul(a, b),
        rtol=5e-2,
        desc="TLX WS Blackwell GEMM, single pinned config (no headroom -> valid MISS)",
    )
)
