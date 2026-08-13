# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Import each kernel module so it self-registers in the registry.

New kernels: add ``from . import <module>`` below (or rely on the guarded import
if the kernel is heavy / hardware-gated).
"""

import sys

from . import ws_gemm  # noqa: F401

# synth_acf is the register-spilling synthetic kernel: force-apply mode covers the
# collect->factory->consume plumbing, and FORCE_ADMIT=0 + do_bench measures a real
# ptxas-ACF win / fidelity. Guard its import so a broken/unavailable synthetic
# kernel never blocks the ws_gemm smoke path.
try:
    from . import synth_acf  # noqa: F401
except Exception as e:  # pragma: no cover
    print(f"[e2e] synth_acf kernel unavailable: {e}", file=sys.stderr)
