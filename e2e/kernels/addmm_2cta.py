# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Kernels: TLX 2-CTA host-TMA matmul, with and without warp specialization.

From ``...tlx.tutorials.blackwell-triton-addmm-2cta_test`` (dashed module name, so ``import_module``).

Both entry points are **already autotune-free** -- the tutorial hardcodes BLOCK 128x128x64 -- which
makes them the cheapest possible additions and a useful check that the pinning machinery is not
itself changing outcomes elsewhere. The pair differs only by warp specialization, giving a second
WS/no-WS comparison on top of the LayerNorm pair, this time *with* tensor cores present.
"""

import importlib

import torch

from ..registry import KernelSpec, register

# pyre-ignore[21]: dashed tlx tutorial module, resolvable only by name.
_tut = importlib.import_module("triton.language.extra.tlx.tutorials.blackwell-triton-addmm-2cta_test")

# The tutorial's own correctness tests only cover small shapes -- (128,128,64) and (256,256,128) for
# the non-WS variant. At 1024^3 the non-WS 2-CTA kernel hangs even with NO ACF applied, so it is an
# upstream shape limitation, not an ACF finding. Stay inside the range upstream actually exercises.
_M, _N, _K = 256, 256, 128

FINGERPRINT_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 64,
    "pinned_from": "hardcoded in the tutorial's wrappers -- no autotuner",
    "features": ["host_tma", "tcgen5", "2cta"],
}


def _inputs():
    a = torch.randn((_M, _K), device="cuda", dtype=torch.float16)
    b = torch.randn((_K, _N), device="cuda", dtype=torch.float16)
    return (a, b)


_REF = lambda a, b: torch.matmul(a, b)  # noqa: E731

for _name, _fn, _feat in (
    ("addmm_2cta", _tut.matmul_2cta, "no warp specialization"),
    ("addmm_2cta_ws", _tut.matmul_2cta_ws, "warp specialized"),
):
    register(
        KernelSpec(
            name=_name,
            make_inputs=_inputs,
            run_fn=_fn,
            ref_fn=_REF,
            rtol=5e-2,
            desc=f"TLX 2-CTA matmul, {_feat}: autotune-free upstream",
        )
    )
