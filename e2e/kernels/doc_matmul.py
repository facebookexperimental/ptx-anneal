# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The CANARY: NVIDIA's CompileIQ doc-example matmul, transcribed from the CURRENT example.

Source: CompileIQ ``examples/compilers/triton_example/triton_ptx.py`` as of upstream #37/#38
("Fix Triton example in Blackwell"). Plain Triton, fixed config, exact-tile 4096x4096.

**Track the upstream example when it changes.** An earlier revision of this file was transcribed
from the pre-#37 example -- autotuned over three configs, of which the first was BLOCK 128/256/64 at
8 warps. That tile is large enough that Triton lowers ``tl.dot`` to tcgen5 on sm_100, so the "known
good" control was silently exercising the Blackwell async-MMA path and failing, which made every
downstream verdict unreadable. Same kernel body, two configs, measured:

    BLOCK  32/64/32  w4  ->  mbarrier 0,  tcgen05 0,  mma.sync 8
    BLOCK 128/256/64 w8  ->  mbarrier 8,  tcgen05 17, mma.sync 0

The canary's job is to answer "is the toolchain healthy?", so it has to be the configuration
upstream actually validates -- not a plausible-looking variation of it.

One deliberate deviation, recorded in the fingerprint: the ACF arrives via ``PTXAS_OPTIONS`` rather
than the doc's per-launch ``ptx_options=``. Same knob underneath, and it is what lets a pasted
kernel be tuned without edits.
"""

from collections.abc import Callable

import torch
import triton
import triton.language as tl

from ..registry import KernelSpec, register

# Upstream's fixed config and exact-tile shape. M/N/K must divide the block sizes -- the kernel does
# no masking, exactly as upstream wrote it.
BLOCK_M, BLOCK_N, BLOCK_K = 32, 64, 32
NUM_WARPS, NUM_STAGES = 4, 3
_DIM = 4096

FINGERPRINT_CONFIG = {
    "BLOCK_M": BLOCK_M,
    "BLOCK_N": BLOCK_N,
    "BLOCK_K": BLOCK_K,
    "num_warps": NUM_WARPS,
    "num_stages": NUM_STAGES,
    "pinned_from": "compileiq examples/compilers/triton_example/triton_ptx.py (fixed config, post-#37)",
    "features": ["mma.sync"],
    "absent": ["tcgen5", "tma", "mbarrier", "warp_specialization"],
}


@triton.jit
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Simple exact-tile matmul: C = A @ B."""
    pid = tl.program_id(0)
    num_pid_n = N // BLOCK_N
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_idxs = k + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_idxs[None, :] * stride_ak
        b_ptrs = b_ptr + k_idxs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)

    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c)


def _run(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = ((M // BLOCK_M) * (N // BLOCK_N),)
    # @triton.jit rewrites the launch signature -- BLOCK_* are constexpr and num_warps/num_stages are
    # launch meta-params -- but the type checker still sees the undecorated def, so the launch is
    # called through a widened alias rather than suppressing an error on every argument line.
    launch: Callable[..., None] = matmul_kernel[grid]
    launch(
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return c


def _inputs():
    # Upstream's distribution: uniform in [-0.5, 0.5), which keeps the fp16 accumulation small
    # enough that its atol=1e-2 check is meaningful rather than vacuous.
    a = torch.rand((_DIM, _DIM), device="cuda", dtype=torch.float16) - 0.5
    b = torch.rand((_DIM, _DIM), device="cuda", dtype=torch.float16) - 0.5
    return (a, b)


register(
    KernelSpec(
        name="doc_matmul",
        make_inputs=_inputs,
        run_fn=_run,
        ref_fn=lambda a, b: torch.matmul(a, b),
        rtol=5e-2,
        desc="CANARY: NVIDIA CompileIQ doc-example matmul (fixed config, post-#37) -- known good",
    )
)
