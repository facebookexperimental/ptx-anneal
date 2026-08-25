# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Kernel: TLX warp-specialized Blackwell flash attention -- TMA + tcgen5 + WS.

Thin wrapper over the upstream TLX tutorial
(``...tlx.tutorials.blackwell_fa_ws.attention``).

Included because it is the kernel that previously failed *differently*: a prior investigation saw it
fault with ``CUDA_ERROR_ILLEGAL_ADDRESS`` under an applied ACF, where the Blackwell GEMMs merely
produced no valid candidate. Two distinct failure modes on the same feature set is worth pinning
down, so it belongs in the matrix rather than being assumed to be "another GEMM-like failure".

Not autotuned in practice: the tutorial's ``configs`` list already holds exactly one entry, so
triton selects it directly without a sweep (see ``pin_autotuner``'s note on why one config is a pin).
"""

import torch

# pyre-ignore[21]: the tlx tutorials ship in the triton-beta runtime but are not in pyre's source DB.
from triton.language.extra.tlx.tutorials.blackwell_fa_ws import attention, configs

from ..registry import KernelSpec, register

# N_CTX must be a multiple of the pinned BLOCK_M (256); HEAD_DIM 128 is the tutorial's shape.
_Z, _H, _N_CTX, _HEAD_DIM = 4, 8, 1024, 128
_SM_SCALE = _HEAD_DIM**-0.5

_cfg = configs[0]
FINGERPRINT_CONFIG = {
    **dict(_cfg.kwargs),
    "num_warps": _cfg.num_warps,
    "num_stages": _cfg.num_stages,
    "pinned_from": f"blackwell_fa_ws.configs[0] (list already has {len(configs)})",
    "features": ["tma", "tcgen5", "warp_specialization"],
}


def _inputs():
    shape = (_Z, _H, _N_CTX, _HEAD_DIM)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)
    return (q, k, v)


def _run(q, k, v):
    return attention(q, k, v, _SM_SCALE)


def _ref(q, k, v):
    # Non-causal: the kernel's inner loop runs lo, hi = 0, N_CTX.
    p = torch.matmul(q.float(), k.float().transpose(2, 3)) * _SM_SCALE
    return torch.matmul(torch.softmax(p, dim=-1), v.float()).to(q.dtype)


register(
    KernelSpec(
        name="fa_ws",
        make_inputs=_inputs,
        run_fn=_run,
        ref_fn=_ref,
        rtol=5e-2,
        desc="TLX WS Blackwell flash attention: TMA + tcgen5 + WS (previously faulted, not just failed)",
    )
)
