# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Kernel: TLX 2-CTA Blackwell GEMM -- TMA + tcgen5, **no warp specialization**.

Thin wrapper over the upstream TLX tutorial
(``triton.language.extra.tlx.tutorials.blackwell_gemm_2cta.matmul``).

This kernel exists to *deconfound* an earlier finding. Applied ACFs break Blackwell TLX GEMMs, and
the obvious suspect was warp specialization -- but this kernel has no ``tlx.async_tasks`` at all and
fails the same way, which rules that out. What it still shares with ``ws_gemm`` is the async-memory
plus tensor-core path: ``tl.make_tensor_descriptor`` + ``tlx.async_descriptor_load`` (TMA) feeding a
tcgen5 MMA into TMEM. TMA and tcgen5 remain confounded *with each other* here; separating them needs
a kernel that uses one without the other.

Nothing to pin: the tutorial's ``matmul`` hardcodes its config (BLOCK 128x128x128, ``num_stages=1``,
``ctas_per_cga=(4, 2, 1)``) and the kernel is plain ``@triton.jit`` with no autotuner, so it is
already in the single-config form the runway requires.
"""

import torch

# pyre-ignore[21]: the tlx tutorials ship in the triton-beta runtime but are not in pyre's source DB.
from triton.language.extra.tlx.tutorials.blackwell_gemm_2cta import matmul

from ..registry import KernelSpec, register

# Multiples of the tutorial's hardcoded 128x128x128 tiling: its grid is (M // BLOCK_M, N // BLOCK_N)
# with no remainder handling, so a non-multiple shape would silently drop tiles.
_M, _N, _K = 1024, 1024, 1024

FINGERPRINT_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 128,
    "num_stages": 1,
    "ctas_per_cga": (4, 2, 1),
    "pinned_from": "hardcoded in the tutorial's matmul() -- no autotuner",
    "features": ["tma", "tcgen5", "tmem"],
}


def _inputs():
    a = torch.randn((_M, _K), device="cuda", dtype=torch.float16)
    b = torch.randn((_K, _N), device="cuda", dtype=torch.float16)
    return (a, b)


register(
    KernelSpec(
        name="gemm_2cta",
        make_inputs=_inputs,
        run_fn=matmul,
        ref_fn=lambda a, b: torch.matmul(a, b),
        # K=1024 fp16 accumulation will not meet the doc's atol=1e-2; run this kernel with
        # `--oracle relerr`, which uses this tolerance.
        rtol=5e-2,
        desc="TLX 2-CTA Blackwell GEMM: TMA + tcgen5, NO warp specialization (WS deconfounder)",
    )
)
