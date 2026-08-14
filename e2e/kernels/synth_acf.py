# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Synthetic register-spilling kernel for the magnon smoke suite: a compute kernel
made deliberately register-heavy so ptxas ACF search has real headroom to tune."""

import torch
import triton
import triton.language as tl

from ..registry import KernelSpec, register

_BLOCK = 1024
_ITERS = 48  # more iterations = deeper spill; too many makes the in-process A/B hit SCORE_TIMEOUT
_NACC = 16  # accumulators kept live at once -> register pressure that forces ptxas to spill
_N = 1 << 20  # multiple of _BLOCK -> no masking needed


@triton.jit
def _synth_acf_kernel(x_ptr, y_ptr, BLOCK: tl.constexpr, ITERS: tl.constexpr, NACC: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs)
    # NACC accumulators, each seeded slightly differently.
    s = [x + 0.1 * i for i in range(NACC)]
    # Every step, each accumulator mixes with its ring neighbour, so all NACC must
    # stay live together -> register pressure -> ptxas spills. The 0.5 factors keep
    # the map contractive (|slope| < 1), so the output stays bounded and stable.
    for _ in range(ITERS):
        # NACC is a constexpr int at trace time, so the ring index is plain Python; pyre models
        # NACC as `constexpr` and rejects `int % constexpr`, so suppress that operand check.
        s = [0.5 * tl.sin(s[i] + 0.5 * s[(i + 1) % NACC]) for i in range(NACC)]  # pyre-ignore[58]
    acc = s[0]
    for i in tl.static_range(1, NACC):  # static_range unrolls at compile time so s[i] is a static index
        acc = acc + s[i]
    tl.store(y_ptr + offs, acc * (1.0 / NACC))


def _ref(x: torch.Tensor) -> torch.Tensor:
    s = [x + 0.1 * i for i in range(_NACC)]
    for _ in range(_ITERS):
        s = [0.5 * torch.sin(s[i] + 0.5 * s[(i + 1) % _NACC]) for i in range(_NACC)]
    acc = s[0]
    for i in range(1, _NACC):  # kept line-parallel with the kernel reduction
        acc = acc + s[i]
    return acc * (1.0 / _NACC)


def _run(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    grid = (x.numel() // _BLOCK,)
    # @triton.jit rewrites the launch signature: BLOCK/ITERS/NACC are constexpr and num_warps is a
    # launch meta-param, neither of which pyre models.
    # pyre-ignore[6, 28]
    _synth_acf_kernel[grid](x, y, BLOCK=_BLOCK, ITERS=_ITERS, NACC=_NACC, num_warps=4)
    return y


def _inputs():
    return (torch.randn((_N,), device="cuda", dtype=torch.float32),)


register(
    KernelSpec(
        name="synth_acf",
        make_inputs=_inputs,
        run_fn=_run,
        ref_fn=_ref,
        rtol=5e-2,
        desc="synthetic register-spilling kernel; ptxas-ACF headroom + measurement-fidelity probe",
    )
)
