# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Kernel: TLX pipelined Blackwell GEMM -- TMA + tcgen5, software-pipelined, **no warp specialization**.

Thin wrapper over ``...tlx.tutorials.blackwell_gemm_pipelined.matmul``.

A second, independent point in the "tcgen5 + TMA without WS" cell (``gemm_2cta`` is the other). Two
kernels reaching the same verdict by different structures -- 2-CTA clustering vs software pipelining
-- is what turns one observation into a pattern.

Autotuned upstream; pinned here with ``pin_autotuner`` rather than by editing the tutorial.
"""

import torch

# pyre-ignore[21]: the tlx tutorials ship in the triton-beta runtime but are not in pyre's source DB.
from triton.language.extra.tlx.tutorials.blackwell_gemm_pipelined import (
    matmul,
    matmul_kernel_tma_pipelined_blackwell,
)

from ..registry import KernelSpec, register
from . import pin_autotuner

_M, _N, _K = 1024, 1024, 1024

FINGERPRINT_CONFIG = pin_autotuner(matmul_kernel_tma_pipelined_blackwell)
FINGERPRINT_CONFIG["features"] = ["tma", "tcgen5", "software_pipelining"]
FINGERPRINT_CONFIG["absent"] = ["warp_specialization"]


def _inputs():
    a = torch.randn((_M, _K), device="cuda", dtype=torch.float16)
    b = torch.randn((_K, _N), device="cuda", dtype=torch.float16)
    return (a, b)


register(
    KernelSpec(
        name="gemm_pipelined",
        make_inputs=_inputs,
        run_fn=matmul,
        ref_fn=lambda a, b: torch.matmul(a, b),
        rtol=5e-2,  # K=1024 fp16: run with --oracle relerr
        desc="TLX pipelined Blackwell GEMM: TMA + tcgen5, no WS (second point in that cell)",
    )
)
