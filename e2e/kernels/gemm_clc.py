# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Kernel: TLX Blackwell GEMM with CLC -- TMA + tcgen5 + WS + cluster launch control.

Thin wrapper over ``...tlx.tutorials.blackwell_gemm_clc.matmul``.

The most feature-dense GEMM in the tutorial set: on top of TMA, tcgen5 and warp specialization it
adds cluster launch control (dynamic tile scheduling). Included as the upper bound of the matrix --
if anything is going to be perturbed by an applied ACF, it is this.

Autotuned upstream; pinned here with ``pin_autotuner``.
"""

import torch

# pyre-ignore[21]: the tlx tutorials ship in the triton-beta runtime but are not in pyre's source DB.
from triton.language.extra.tlx.tutorials.blackwell_gemm_clc import (
    matmul,
    matmul_kernel_tma_ws_blackwell_clc,
)

from ..registry import KernelSpec, register
from . import pin_autotuner

_M, _N, _K = 1024, 1024, 1024

FINGERPRINT_CONFIG = pin_autotuner(matmul_kernel_tma_ws_blackwell_clc)
FINGERPRINT_CONFIG["features"] = ["tma", "tcgen5", "warp_specialization", "clc"]


def _inputs():
    a = torch.randn((_M, _K), device="cuda", dtype=torch.float16)
    b = torch.randn((_K, _N), device="cuda", dtype=torch.float16)
    return (a, b)


register(
    KernelSpec(
        name="gemm_clc",
        make_inputs=_inputs,
        run_fn=matmul,
        ref_fn=lambda a, b: torch.matmul(a, b),
        rtol=5e-2,  # K=1024 fp16: run with --oracle relerr
        desc="TLX Blackwell GEMM + cluster launch control: TMA + tcgen5 + WS + CLC (densest cell)",
    )
)
