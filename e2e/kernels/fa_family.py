# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Kernels: the TLX Blackwell flash-attention family (4 variants), autotuning disabled.

All four upstream tutorials expose the same entry point --
``attention(q, k, v, sm_scale, causal, config=None)`` -- and differ only in how the attention loop is
structured: cluster launch control, persistence, software pipelining, or both. Registering them from
one module rather than four near-identical files keeps the differences visible: the shapes, the
reference and the oracle are shared, so any divergence in verdict is the kernel structure, not the
harness.

They are all TMA + tcgen5 + warp specialization, so on the current evidence they are all expected to
behave alike -- which is exactly why they are worth running. Four independent kernels reaching the
same verdict is a pattern; one kernel is an anecdote.

Autotuning is disabled per variant: two ship a single config already, two ship 24 and 42, and those
are pinned via ``pin_autotuner``. The multi-config ones carry an ``early_config_prune`` hook that is
a pure filter on ``(HEAD_DIM, STAGE)``, so no derived constexprs are at stake here -- but the hook is
still run, because pinning cannot tell a filter from a deriver and guessing wrong is silent.
"""

import importlib

import torch

from ..registry import KernelSpec, register
from . import pin_autotuner

# Non-causal keeps the reference a plain softmax(QK^T)@V with no masking to get wrong.
_Z, _H, _N_CTX, _HEAD_DIM = 4, 8, 1024, 128
_SM_SCALE = _HEAD_DIM**-0.5
_CAUSAL = False
# STAGE is triton-FA convention: 3 when causal, 1 otherwise. The variants' prune hooks are pure
# filters keyed on (HEAD_DIM, STAGE) -- no derived constexprs -- but pin_autotuner cannot tell a
# filter from a deriver, so it demands the args either way.
_PRUNE_ARGS = {"HEAD_DIM": _HEAD_DIM, "STAGE": 3 if _CAUSAL else 1}

# (registered name, tutorial module, distinguishing feature)
_VARIANTS = [
    ("fa_clc", "blackwell_fa_clc", "cluster launch control"),
    ("fa_ws_persistent", "blackwell_fa_ws_persistent", "persistent"),
    ("fa_ws_pipelined", "blackwell_fa_ws_pipelined", "software pipelined"),
    ("fa_ws_pipelined_persistent", "blackwell_fa_ws_pipelined_persistent", "pipelined + persistent"),
]


def _inputs():
    shape = (_Z, _H, _N_CTX, _HEAD_DIM)
    return tuple(torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3))


def _ref(q, k, v):
    p = torch.matmul(q.float(), k.float().transpose(2, 3)) * _SM_SCALE
    return torch.matmul(torch.softmax(p, dim=-1), v.float()).to(q.dtype)


def _force_pin(tuner) -> dict:
    """Last resort: keep configs[0] when the prune hook rejects everything for our shapes.

    Only reachable for a secondary kernel in the module (e.g. an unused backward pass) whose hook is
    keyed on arguments this forward-only KernelSpec never supplies. Still leaves exactly one config,
    which is what makes the module unambiguous.
    """
    chosen = tuner.configs[0]
    tuner.configs = [chosen]
    if hasattr(tuner, "cache"):
        try:
            tuner.cache.clear()
        except AttributeError:
            pass
    return {**dict(chosen.kwargs), "pinned_from": "configs[0] (prune hook not applicable)"}


def _make_run(attention):
    def _run(q, k, v):
        return attention(q, k, v, _SM_SCALE, _CAUSAL)

    return _run


# One dict per variant, so the fingerprint can say which config each was pinned to. A module-level
# FINGERPRINT_CONFIG cannot express four kernels, so this maps name -> config and `_identity` picks
# up the module attribute for whichever run_fn it is looking at.
FINGERPRINT_CONFIG: dict = {}

for _name, _modname, _feature in _VARIANTS:
    _mod = importlib.import_module(f"triton.language.extra.tlx.tutorials.{_modname}")
    _cfgs = getattr(_mod, "configs", [])
    if len(_cfgs) > 1:
        # Pin EVERY autotuner in the module, not just the first. The pipelined+persistent tutorial
        # carries two (_attn_fwd_ws with 42 configs and _attn_bwd_ws with 3); leaving the backward
        # one live meant the runway refused the kernel, which is the check working -- but the fix is
        # to disable all of them, since "which config" must be unambiguous for the whole module.
        _pins = {}
        for _tn, _tuner in vars(_mod).items():
            if type(_tuner).__name__ != "Autotuner" or len(getattr(_tuner, "configs", [])) <= 1:
                continue
            try:
                _pins[_tn] = pin_autotuner(_tuner, prune_args=_PRUNE_ARGS)
            except ValueError:
                # This module's other autotuner is a backward kernel whose prune hook is keyed on
                # arguments a forward-only KernelSpec never supplies, so the hook rejects everything.
                # It is never launched here; pinning it positionally just removes the ambiguity the
                # refusal check (correctly) objects to.
                _pins[_tn] = _force_pin(_tuner)
        FINGERPRINT_CONFIG[_name] = {"pinned": _pins}
    else:
        _c = _cfgs[0]
        FINGERPRINT_CONFIG[_name] = {
            **dict(_c.kwargs),
            "num_warps": _c.num_warps,
            "num_stages": _c.num_stages,
            "pinned_from": f"{_modname}.configs[0] (list already has 1)",
        }
    FINGERPRINT_CONFIG[_name]["features"] = ["tma", "tcgen5", "warp_specialization", _feature]

    register(
        KernelSpec(
            name=_name,
            make_inputs=_inputs,
            run_fn=_make_run(_mod.attention),
            ref_fn=_ref,
            rtol=5e-2,
            desc=f"TLX Blackwell flash attention ({_feature}): TMA + tcgen5 + WS",
        )
    )
