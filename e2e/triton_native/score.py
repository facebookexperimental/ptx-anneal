# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Score ONE ACF by running a registered kernel through **fbtriton** itself.

This is the triton-native channel's scorer. It is invoked exactly the way the PTX-direct one is --
through ``$PTX_ANNEAL_SCORE_CMD``, the argv prefix the CompileIQ adapter appends
``--task/--acf/--ptxas/--warmup/--rep`` to -- and it honours the same contract:

    **the last line of stdout is** ``MS <float>`` **(valid) or** ``INVALID``

Reusing that one mechanism is deliberate. A second scoring protocol would mean the two channels
could not be driven by the same adapter with the same budget, and then a disagreement between them
would be uninterpretable.

What makes this channel *native* is only how the ACF is applied: instead of assembling captured PTX
and relaunching it through the driver, we set

    PTXAS_OPTIONS=--apply-controls=<acf>
    TRITON_ALWAYS_COMPILE=1

**before importing triton**, and let the compiler do what it normally does.

**The frontend is fbtriton specifically, not Triton in general.** ``PTXAS_OPTIONS`` is fbtriton's
spelling of the knob (it backs ``CUDAOptions.ptx_options``); upstream Triton has no such knob. That
distinction is not cosmetic -- on a Triton without it, every candidate would compile *untuned*, run
fine, and be reported as a valid measurement. The search would then "find" wins that are really just
the baseline measured N times, which is the worst failure this repo can produce: green, plausible
and wrong. So the ACF application is **verified, not assumed** (see :func:`verify_acf_applied`), and
the frontend's identity is recorded in the fingerprint.

The ordering is equally load-bearing: ``CUDAOptions.ptx_options`` takes its default from the knob at
*class-definition* time, i.e. when the nvidia backend module is first imported. Once that has
happened the ACF cannot be changed in that process. Hence every module here imports torch/triton
inside functions, and hence one candidate per process -- a consequence, not a design choice.

Usage (normally driven by ``runway.py``, never by hand):

    python -m e2e.triton_native.score --kernel doc_matmul --task doc_matmul \\
        --acf /tmp/x.acf --ptxas /path/ptxas --warmup 100 --rep 1000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping

# Fixed so that inputs are identical in every candidate's process. Without it each subprocess would
# draw different data, and both the correctness check and the ref_fn=None baseline comparison would
# be against a *different problem* -- the candidate would be blamed for the RNG. Recorded in the
# fingerprint as objective.seed.
SEED = 0

# magnon's own consume hook in fbtriton. If it is on, it appends its own `--apply-controls` to the
# ptxas command line and the kernel is compiled with TWO ACFs -- ours and whatever the store had.
# Every measurement after that is meaningless, so this is a hard refusal rather than a warning.
FORBIDDEN_ENV = "TRITON_COMPILE_IQ_APPLY"


def check_env_clean(env: Mapping[str, str] | None = None) -> None:
    """Refuse to run under magnon's consume hook (see :data:`FORBIDDEN_ENV`)."""
    environ: Mapping[str, str] = os.environ if env is None else env
    if environ.get(FORBIDDEN_ENV):
        raise SystemExit(
            f"triton-native: refusing to run with {FORBIDDEN_ENV}={environ[FORBIDDEN_ENV]!r} set. "
            "That is magnon's consume hook: it appends a second --apply-controls, so the kernel "
            f"would be compiled with two ACFs and every measurement would be junk. Unset {FORBIDDEN_ENV}."
        )


def acf_env(acf_path: str | None, ptxas: str | None, raw_opts: str | None = None) -> dict[str, str]:
    """The environment that makes Triton compile this kernel under ``acf_path``.

    Pure and separately testable on purpose -- this is the whole mechanism of the channel, and it
    has to be right before triton is imported, where a mistake is invisible (you just measure the
    untuned kernel and call it a win).

    ``acf_path=None`` is the no-ACF baseline: ``PTXAS_OPTIONS`` is *absent*, not empty, so the
    baseline is compiled exactly as an untouched Triton kernel would be.
    """
    env: dict[str, str] = {}
    if raw_opts:
        # The injection canary: plain ptxas flags (-O0 / -O3) instead of an ACF, down the SAME
        # PTXAS_OPTIONS path. Proves ptxas actually acted on what we sent, which the ACF path cannot
        # show on its own -- an ignored ACF and a no-op ACF look identical.
        env["PTXAS_OPTIONS"] = raw_opts
    elif acf_path:
        # FBTriton's knob is PTXAS_OPTIONS (it backs CUDAOptions.ptx_options), NOT the OAI spelling
        # TRITON_PTXAS_OPTIONS. Getting this wrong silently measures the untuned kernel.
        env["PTXAS_OPTIONS"] = f"--apply-controls={acf_path}"
    # Without this, Triton serves the first candidate's cubin from its cache for every later
    # candidate: the search would score one kernel N times and converge on noise.
    env["TRITON_ALWAYS_COMPILE"] = "1"
    if ptxas:
        # arch >= 100 (Blackwell) reads TRITON_PTXAS_BLACKWELL_PATH, older arches TRITON_PTXAS_PATH.
        # Set both: the factory, the frontend and this channel must all use ONE ptxas, and the
        # discovered >=13.3 one is usually a pip wheel that is not on PATH.
        env["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas
        env["TRITON_PTXAS_PATH"] = ptxas
    return env


def frontend_info() -> dict:
    """Identify the Triton actually in use. Imports triton -- call only after the ACF env is set.

    Recorded in the fingerprint because in this channel the frontend *is* part of the search
    problem: it decides the PTX, and therefore what ptxas is even asked to optimise. Two runs on
    different Triton builds are not the same experiment, and "triton" alone does not say which.
    """
    import triton

    version = getattr(triton, "__version__", "unknown")
    path = os.path.dirname(os.path.abspath(triton.__file__))
    # TLX and the magnon collector are fbtriton-only. Either one present means this is the fork,
    # which is what the channel actually requires -- see verify_acf_applied().
    has_tlx = os.path.isdir(os.path.join(path, "language", "extra", "tlx"))
    has_magnon = os.path.isdir(os.path.join(path, "magnon"))
    return {
        "name": "fbtriton" if (has_tlx or has_magnon or "fb" in version) else "triton",
        "version": version,
        "path": path,
        "has_tlx": has_tlx,
        "has_magnon": has_magnon,
    }


def expected_ptx_options(acf_path: str | None, raw_opts: str | None) -> str | None:
    """What ``CUDAOptions.ptx_options`` must contain for this invocation."""
    if raw_opts:
        return raw_opts
    return f"--apply-controls={acf_path}" if acf_path else None


def verify_acf_applied(acf_path: str | None, raw_opts: str | None = None) -> str | None:
    """Prove the ACF reached the compiler. Refuse to measure anything if it did not.

    Reads the *actual* default of ``CUDAOptions.ptx_options`` -- the value the compiler will use --
    rather than trusting that setting ``PTXAS_OPTIONS`` worked. That distinction is the whole point:
    the three ways this silently breaks (a Triton without the ``PTXAS_OPTIONS`` knob, the upstream
    ``TRITON_PTXAS_OPTIONS`` spelling, or triton imported before the env was set) all fail the same
    way -- the kernel compiles untuned, runs perfectly, and returns a number that looks like a
    result. A search built on that reports fictional wins.

    Returns the observed value so it can be recorded.
    """
    import dataclasses

    try:
        from triton.backends.nvidia.compiler import CUDAOptions
    except ImportError as e:
        raise SystemExit(
            f"triton-native: cannot import triton's nvidia backend ({e}). This channel requires "
            "fbtriton with a CUDA backend."
        ) from None

    fields = {f.name: f.default for f in dataclasses.fields(CUDAOptions)}
    if "ptx_options" not in fields:
        raise SystemExit(
            "triton-native: this Triton's CUDAOptions has no `ptx_options` field, so an ACF can "
            "never be applied. The channel requires **fbtriton** (whose PTXAS_OPTIONS knob backs "
            "that field); upstream Triton does not have it. Refusing to run, because the kernel "
            "would compile untuned and every candidate would be scored as a valid 'win'."
        )
    observed = fields["ptx_options"]
    expected = expected_ptx_options(acf_path, raw_opts)
    if observed != expected:
        raise SystemExit(
            f"triton-native: the ACF did not reach the compiler. CUDAOptions.ptx_options is "
            f"{observed!r}, expected {expected!r}. Either PTXAS_OPTIONS is not this Triton's knob "
            "(upstream spells it TRITON_PTXAS_OPTIONS), or triton was imported before the "
            "environment was set. Refusing to report a measurement of the untuned kernel."
        )
    # A field with no default reads back as dataclasses.MISSING, which cannot equal `expected` and so
    # has already raised above; narrowing here is for the type checker, not a behaviour change.
    return observed if isinstance(observed, str) else None


def _load_spec(kernel: str):
    """Import the kernel registry and return one KernelSpec. Imports torch/triton transitively."""
    from .. import kernels as _kernels  # noqa: F401 - importing the package self-registers every kernel
    from ..registry import KERNELS

    if kernel not in KERNELS:
        raise SystemExit(f"triton-native: unknown kernel {kernel!r} (registered: {sorted(KERNELS) or 'none'})")
    return KERNELS[kernel]


def find_autotuners(fn, depth: int = 2) -> list[str]:
    """``@triton.autotune``'d kernels reachable from ``fn``'s module that are still *unpinned*.

    Why reach past the immediate module: a KernelSpec's ``run_fn`` often just calls a kernel defined
    elsewhere (``ws_gemm`` wraps a tlx tutorial), so the autotuner is one import away.

    **An autotuner narrowed to exactly one config is not autotuning** and is not reported here.
    That is triton's own rule, not a convention we invented: ``Autotuner.run`` only benchmarks when
    ``len(self.configs) > 1``; with one config it uses it directly, with no sweep and no ambiguity
    about which kernel is being measured. Narrowing the list is therefore the supported way to pin a
    tutorial kernel without editing it -- see :func:`e2e.kernels.pin_autotuner`. Refusing those too
    would block 16 of the 19 Blackwell TLX tutorials for no safety gain.

    Detection is by *type name*, not ``isinstance``: this must work without importing triton, so the
    same check can run in a test with a stub. An Autotuner is an Autotuner in either case.
    """
    found: list[str] = []
    seen: set[int] = set()

    def walk_fn(f, level):
        """A callable reaches autotuners two ways: through its module globals, and through its
        closure. The closure matters as much as the globals -- a factory like
        ``_make_run(attention)`` puts the tutorial's entry point in a cell, and its ``__globals__``
        are the *wrapper's* module, where no autotuner lives. Following only globals would report a
        clean bill of health for a fully autotuned kernel, which is the one wrong answer this check
        must never give."""
        if f is None or level > depth:
            return
        walk(getattr(f, "__globals__", None), level)
        for cell in getattr(f, "__closure__", None) or ():
            try:
                v = cell.cell_contents
            except ValueError:  # an empty cell (recursive closure still being built)
                continue
            if callable(v):
                walk_fn(v, level + 1)

    def walk(globals_dict, level):
        if level > depth or globals_dict is None or id(globals_dict) in seen:
            return
        seen.add(id(globals_dict))
        for name, obj in list(globals_dict.items()):
            if type(obj).__name__ == "Autotuner":
                n = len(getattr(obj, "configs", []) or [])
                if n != 1:
                    found.append(f"{globals_dict.get('__name__', '?')}.{name} ({n} configs)")
            elif callable(obj):
                walk_fn(obj, level + 1)

    walk_fn(fn, 1)
    return sorted(set(found))


def refuse_autotune(spec) -> None:
    """Refuse an autotuned kernel. **TODO: support autotune.**

    Two independent problems, either one fatal:

    1. *Which kernel is being tuned?* Autotune picks a config at runtime; the search would tune
       whatever it happened to pick, and the fingerprint's "pinned config" would be a fiction.
    2. *Autotune re-benchmarks under every ACF.* Each candidate would pay a full config sweep, and
       the config could change between candidates -- so the ms we hand the engine would compare two
       different kernels.

    Every TLX tutorial ships ``@triton.autotune``, so a pasted kernel hits this by default. Refusing
    with instructions is the honest outcome; quietly tuning something undefined is not.

    Two forms pass. An autotuner already narrowed to one config (see :func:`find_autotuners`), and a
    module that declares ``AUTOTUNE_BYPASSED = "<why>"`` -- for a wrapper that launches the underlying
    JITFunction directly (``kernel.fn[grid](...)``) and never consults the autotuner at all. The TLX
    WS GEMM does exactly that when handed an explicit config, and its autotuner still lists ~1.2M
    configs that are never used, so refusing on the object's mere presence is a false positive. The
    declaration is required to be explicit and is recorded in the fingerprint, so the claim is
    auditable rather than a silent skip.
    """
    mod = sys.modules.get(getattr(spec.run_fn, "__module__", "") or "")
    bypass = getattr(mod, "AUTOTUNE_BYPASSED", None)
    if bypass:
        return None
    hits = find_autotuners(spec.run_fn)
    if hits:
        raise SystemExit(
            f"triton-native: kernel {spec.name!r} uses @triton.autotune ({', '.join(hits)}). "
            "This is UNSUPPORTED (TODO). Under a search, autotune re-benchmarks every config for "
            "every candidate ACF and may pick a different config each time, so the score compares "
            "different kernels and 'the kernel being tuned' is undefined. Pin it to ONE config -- "
            "either e2e.kernels.pin_autotuner(kernel, ...) to narrow the config list in place (the "
            "supported way to pin a tutorial you do not want to edit), or replace the decorator with "
            "explicit constexpr args + num_warps/num_stages, as e2e/kernels/doc_matmul.py does."
        )


def _identity(spec, inputs) -> dict:
    """Kernel identity for the fingerprint: name, shapes, dtypes, pinned config, oracle strength."""
    import torch

    mod = sys.modules.get(getattr(spec.run_fn, "__module__", "") or "")
    return {
        "name": spec.name,
        "shapes": [list(t.shape) for t in inputs if isinstance(t, torch.Tensor)],
        "dtypes": [str(t.dtype) for t in inputs if isinstance(t, torch.Tensor)],
        # Optional module-level annotation; absent is recorded as null rather than guessed, because a
        # wrong "pinned config" in a fingerprint is worse than an admitted unknown.
        "pinned_config": getattr(mod, "FINGERPRINT_CONFIG", None),
        # Present only when the module claims the autotuner is never consulted; recorded so the
        # claim travels with the result instead of living in a comment.
        "autotune_bypassed": getattr(mod, "AUTOTUNE_BYPASSED", None),
        "has_independent_ref": spec.has_independent_ref,
        "desc": spec.desc,
    }


def _correct(out, ref, oracle: str, atol: float, rtol: float) -> tuple[bool, float]:
    """Apply the correctness oracle. Returns ``(ok, observed_max_rel_err)``."""
    import torch

    denom = ref.abs().max().clamp_min(1e-6)
    rel = ((out.float() - ref.float()).abs().max() / denom).item()
    if oracle == "allclose":
        # NVIDIA's doc example, verbatim: absolute tolerance, no relative term.
        return bool(torch.allclose(out, ref, atol=atol, rtol=0)), rel
    return rel <= rtol, rel


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="e2e.triton_native.score", description="Score one ACF through Triton.")
    ap.add_argument("--kernel", required=True, help="registered kernel name (e2e/kernels/)")
    # The adapter always appends --task. In this channel a "task" is a kernel name, not a directory;
    # cross-check rather than ignore, so a mis-wired SCORE_CMD fails loudly instead of scoring the
    # wrong kernel.
    ap.add_argument("--task", help="must equal --kernel (the adapter's contract field)")
    ap.add_argument("--acf", default="NONE", help="ACF file path, or NONE for the untuned baseline")
    ap.add_argument("--ptxas")
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--rep", type=int, default=1000)
    ap.add_argument("--oracle", choices=["allclose", "relerr"], default="allclose")
    ap.add_argument("--atol", type=float, default=1e-2, help="allclose oracle: absolute tolerance (doc: 1e-2)")
    ap.add_argument("--ref", help="ref_fn=None fallback: compare against this dumped no-ACF output")
    ap.add_argument("--ref-dump", dest="ref_dump", help="write this run's output here (baseline pass)")
    ap.add_argument("--probe-out", dest="probe_out", help="write kernel identity JSON here (baseline pass)")
    ap.add_argument(
        "--ptxas-opts",
        dest="ptxas_opts",
        help="raw PTXAS_OPTIONS instead of an ACF (the -O0/-O3 injection canary); skips correctness",
    )
    args = ap.parse_args(argv)

    if args.task and args.task != args.kernel:
        raise SystemExit(f"triton-native: --task {args.task!r} != --kernel {args.kernel!r} (SCORE_CMD mis-wired)")
    check_env_clean()

    acf = None if args.acf in (None, "NONE") else args.acf
    # THE critical ordering: environment first, triton second. Everything below this line may import
    # triton; nothing above it may.
    os.environ.update(acf_env(acf, args.ptxas, args.ptxas_opts))

    import torch
    import triton

    # Before anything is measured: prove the ACF actually reached the compiler. If it did not, every
    # number below would describe the UNTUNED kernel while looking like a valid candidate score.
    applied = verify_acf_applied(acf, args.ptxas_opts)

    spec = _load_spec(args.kernel)
    refuse_autotune(spec)

    torch.manual_seed(SEED)
    inputs = spec.make_inputs()

    if args.probe_out:
        identity = _identity(spec, inputs)
        identity["frontend"] = frontend_info()
        identity["ptx_options"] = applied
        with open(args.probe_out, "w") as f:
            json.dump(identity, f)

    out = spec.run_fn(*inputs)
    torch.cuda.synchronize()

    if args.ref_dump:
        # The baseline pass doubles as the reference producer for the ref_fn=None fallback. Saved on
        # CPU so a later process can load it without caring which device it lands on.
        torch.save(out.detach().to("cpu"), args.ref_dump)

    # Pick the reference: an independent ref_fn if the spec has one, else the dumped no-ACF output.
    if spec.ref_fn is not None:
        ref = spec.ref_fn(*inputs)
    elif args.ref:
        ref = torch.load(args.ref, map_location=out.device)
    else:
        # The baseline pass with no ref_fn has nothing to compare against yet -- it IS the reference.
        ref = None

    if ref is not None:
        ok, rel = _correct(out, ref, args.oracle, args.atol, spec.rtol)
        if not ok:
            tol = f"atol={args.atol}" if args.oracle == "allclose" else f"rtol={spec.rtol}"
            sys.stderr.write(f"[score] INVALID kernel={args.kernel} oracle={args.oracle} {tol} rel_err={rel:.3e}\n")
            print("INVALID")
            return 0

    ms = triton.testing.do_bench(
        lambda: spec.run_fn(*inputs),
        warmup=args.warmup,
        rep=args.rep,
        return_mode="mean",  # the doc's reduction. warmup/rep are DURATIONS in ms, not iteration counts.
    )
    print(f"MS {float(ms)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001 - any candidate failure is INVALID to the engine, not a crash
        sys.stderr.write(f"[score] {type(e).__name__}: {e}\n")
        print("INVALID")
        raise SystemExit(0) from None
