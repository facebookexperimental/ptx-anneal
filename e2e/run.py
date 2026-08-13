# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Generic buck harness for the magnon 3-step e2e smoke kernels.

Runs one registered TLX kernel under the buck-built triton-beta so that triton's
magnon collector/consumer -- gated by the ``TRITON_COMPILE_IQ_*`` env vars that
``e2e/e2e.sh`` sets -- sees a real kernel compile. This binary only *runs* the
kernel; collect vs consume is selected entirely by the environment, so ``--mode``
is informational (it just labels the log line).

Normally driven by ``MODE=full e2e/e2e.sh``, which builds this module against a triton
carrying the magnon hooks. How that build happens is site-specific and lives in the
optional hook (``fb/e2e_internal.sh``), not here.

Run it directly against an already-installed magnon-capable triton with:
    python -m e2e.run --kernel ws_gemm --mode collect
The interpreter's ``ptxas`` must be the one the factory tunes with, or the collected PTX
keys to a different store entry and consume will MISS.
"""

import argparse

import torch

from .registry import KERNELS


def _rel_err(out: torch.Tensor, ref: torch.Tensor) -> float:
    denom = ref.abs().max().clamp_min(1e-6)
    return ((out.float() - ref.float()).abs().max() / denom).item()


def main() -> None:
    # Importing the kernels package self-registers every KernelSpec.
    from . import kernels as _kernels  # noqa: F401

    ap = argparse.ArgumentParser(description="magnon e2e smoke kernel runner")
    ap.add_argument("--kernel", required=True, choices=sorted(KERNELS))
    ap.add_argument("--mode", default="bench", choices=["collect", "consume", "bench"])
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    spec = KERNELS[args.kernel]
    inputs = spec.make_inputs()

    out = spec.run_fn(*inputs)
    torch.cuda.synchronize()

    ref = spec.ref_fn(*inputs)
    rel = _rel_err(out, ref)
    if rel > spec.rtol:
        raise SystemExit(
            f"[smoke] FAIL kernel={args.kernel} rel_err={rel:.3e} > rtol={spec.rtol:.3e}"
        )

    # A few extra launches so the compiled+consumed kernel is actually exercised.
    for _ in range(args.iters):
        spec.run_fn(*inputs)
    torch.cuda.synchronize()

    print(f"[smoke] OK kernel={args.kernel} mode={args.mode} rel_err={rel:.3e}")


if __name__ == "__main__":
    main()
