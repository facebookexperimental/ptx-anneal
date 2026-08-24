# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Registry of magnon 3-step e2e smoke kernels.

Append a new kernel by dropping a module in ``kernels/`` that calls
``register(KernelSpec(...))``. The buck harness (``run.py``) and ``e2e.sh``
discover kernels purely through ``KERNELS`` -- no harness edits needed.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class KernelSpec:
    name: str
    make_inputs: Callable[[], tuple]  # () -> args passed to run_fn / ref_fn
    run_fn: Callable  # (*inputs) -> torch.Tensor produced by the TLX kernel
    # (*inputs) -> reference torch.Tensor. None is allowed and falls back to comparing against the
    # kernel's OWN no-ACF output -- see WEAKER_VERDICT below. ref_fn is the hard field to fill for a
    # pasted kernel, so refusing None would just push people into a fake ref_fn, which is worse:
    # a deliberate, labelled fallback beats an unlabelled lie.
    ref_fn: Callable | None = None
    rtol: float = 5e-2  # max allowed max-relative-error vs ref
    desc: str = ""

    @property
    def has_independent_ref(self) -> bool:
        return self.ref_fn is not None


# The single place the weaker-verdict wording lives, so every consumer labels it identically and a
# grep for it finds them all. A self-referential check catches a *miscompile* (the ACF changed the
# math) but not a kernel that was wrong to begin with -- so a green verdict under it is strictly
# weaker evidence, and must never be reported as if it were not.
WEAKER_VERDICT = "WEAKER: no ref_fn; correctness is vs the kernel's own no-ACF output (self-referential)"


KERNELS: dict[str, KernelSpec] = {}


def register(spec: KernelSpec) -> None:
    if spec.name in KERNELS:
        raise ValueError(f"duplicate smoke kernel name: {spec.name}")
    KERNELS[spec.name] = spec
