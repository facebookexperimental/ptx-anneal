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
    ref_fn: Callable  # (*inputs) -> reference torch.Tensor
    rtol: float = 5e-2  # max allowed max-relative-error vs ref
    desc: str = ""


KERNELS: dict[str, KernelSpec] = {}


def register(spec: KernelSpec) -> None:
    if spec.name in KERNELS:
        raise ValueError(f"duplicate smoke kernel name: {spec.name}")
    KERNELS[spec.name] = spec
