# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The triton-native runway: drive a CompileIQ search over a kernel run through **fbtriton** itself.

    python -m e2e.triton_native.runway                    # the canary only
    python -m e2e.triton_native.runway --kernel ws_gemm   # canary, then ws_gemm
    python -m e2e.triton_native.runway --kernel ws_gemm --no-canary

**Why this channel exists.** ptx-anneal's shipped path captures a kernel's PTX and relaunches it
through the driver. That is what scales, but it means a red result has two possible causes: the
kernel really breaks under an applied ACF, or our capture/relaunch is wrong. This channel removes
the second possibility -- no PTX capture, no launch spec, no hand-rolled tensormaps, just
``PTXAS_OPTIONS=--apply-controls=<acf>`` and fbtriton doing what it normally does.

The frontend is **fbtriton**, not Triton in general: ``PTXAS_OPTIONS`` is fbtriton's knob (it backs
``CUDAOptions.ptx_options``) and upstream Triton has no equivalent, so on the wrong Triton the ACF
would be ignored and every candidate scored as the untuned kernel. That is checked cheaply here and
verified for real in the scorer.

Its product is a faster *verdict*, not a faster kernel:

    | triton-native | torch-free  | conclusion                                              |
    | fail          | --          | a CompileIQ issue -- reportable to NVIDIA               |
    | pass          | fail        | a Magnon issue: bad PTX repro, or a consumption bug     |
    | pass          | pass, differ| a Magnon issue, the dangerous one -- looks green        |
    | pass          | pass, agree | trustworthy                                             |

That table is only usable if both channels posed the **same search problem**, which is why every run
ends by printing a fingerprint (see :mod:`ptx_anneal.fingerprint`). Diff the two; equal digests mean
the comparison is meaningful, and nothing else does.

**"Pass" here means >= 1 valid candidate**, not a win. The channel's job is to answer "can this
kernel survive an applied ACF at all"; win size is the torch-free channel's question.

**Adhoc and optional by construction.** Pure stdlib: torch, fbtriton and the engine are probed,
never imported, so on a box without them this prints why and exits 2 -- exactly as ``e2e.sh`` does
when its ``MODE=full`` site hook is missing. It lives under ``e2e/`` and is in neither the wheel nor the sdist.

Defaults follow NVIDIA's documented example, because "we followed your doc" is what makes a
CompileIQ bug report credible rather than deflectable. See ``--help`` and README.md for the list.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

# NVIDIA's documented example (compileiq examples/compilers/triton_example/triton_ptx.py). These are
# defaults, not constants: each is exposed as a flag and every one of them is recorded in the
# fingerprint, so a deliberate deviation stays visible instead of becoming folklore.
DOC_GENERATIONS = 5
DOC_POOL = 32  # pydantic-validated to be > 5
# The doc uses start(task_timeout=20). We CANNOT: applying an ACF means setting PTXAS_OPTIONS before
# fbtriton is imported, so every candidate is a fresh process that spends ~8s importing torch+fbtriton
# before it can measure anything. At 20s a slow compile times out, the engine moves on, and (before
# the adapter learned to kill its own child) the abandoned process kept the GPU busy and dragged the
# next candidate over the limit too -- a spiral that reports every candidate invalid. 120s is a
# deviation from the doc, recorded in the fingerprint, and forced by the process-per-candidate model.
DOC_TASK_TIMEOUT = 120  # seconds, per candidate (doc: 20, in-process)
DOC_WARMUP = 100  # do_bench warmup, MILLISECONDS (triton do_bench takes durations, not iters)
DOC_REP = 1000  # do_bench measurement window, MILLISECONDS
DOC_ATOL = 1e-2  # torch.allclose(atol=1e-2, rtol=0)
B200_MAX_SM_MHZ = 1965  # locking clocks is "crucial" per the doc; this is B200's max SM clock

CANARY = "doc_matmul"


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _ensure_repo_on_path() -> str:
    """Put THIS checkout ahead of everything on ``sys.path``, and return it.

    Same reasoning as ``e2e.sh``'s ``HARNESS_PYTHONPATH``: an editable ``ptx_anneal`` install
    pointing at another tree (an fbcode one, say) would otherwise be imported, and this harness
    exists to validate the tree it ships in.
    """
    root = _repo_root()
    if sys.path[:1] != [root]:
        sys.path.insert(0, root)
    return root


def _child_env(base: dict | None = None) -> dict:
    """Environment for a subprocess, with this checkout pinned onto PYTHONPATH (see above)."""
    env = dict(os.environ if base is None else base)
    root = _repo_root()
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def availability() -> tuple[bool, list[str]]:
    """Can this channel run here? Returns ``(ok, reasons_it_cannot)``.

    Probes by *spec lookup*, never by import: importing triton here would defeat the whole
    process-per-candidate design (and importing torch on a box with a half-installed CUDA stack can
    hang). The rest of the repo must be unaffected by this module being unusable.
    """
    import importlib.util

    missing: list[str] = []
    specs = {}
    for mod, why in (
        ("torch", "the kernels are torch tensors"),
        ("triton", "this channel compiles the kernel with fbtriton"),
        ("compileiq", "the search engine (pip install ptx-anneal[compileiq])"),
    ):
        try:
            specs[mod] = importlib.util.find_spec(mod)
        except (ImportError, ValueError):
            specs[mod] = None
        if specs[mod] is None:
            missing.append(f"{mod} -- {why}")

    # The frontend must be FBTRITON, not Triton in general: `PTXAS_OPTIONS` is fbtriton's knob, and
    # on a Triton without it every candidate would compile untuned and still score as valid -- a
    # search reporting wins it never made. Detected here by looking for fbtriton-only subpackages on
    # disk, so the check costs no import. It is only the early warning; the authoritative check is
    # verify_acf_applied() in the scorer, which reads the value the compiler actually received.
    if specs.get("triton") is not None:
        locs = list(getattr(specs["triton"], "submodule_search_locations", None) or [])
        root = locs[0] if locs else ""
        if root and not any(os.path.isdir(os.path.join(root, *p)) for p in (("language", "extra", "tlx"), ("magnon",))):
            missing.append(
                f"fbtriton -- the triton at {root} looks like upstream Triton (no tlx/, no magnon/). "
                "This channel needs fbtriton: PTXAS_OPTIONS is its knob, and without it the ACF is "
                "silently ignored and every candidate scores as the untuned kernel"
            )

    try:
        _ensure_repo_on_path()
        # ptx_anneal is imported off THIS checkout via sys.path (see _ensure_repo_on_path), not from
        # a buck dep -- the e2e target is deliberately tiny and this import is optional by design, so
        # a type checker cannot resolve it statically.
        from ptx_anneal import provision  # pyre-ignore[21]

        ptxas = provision.find_ptxas()
        if not ptxas:
            missing.append("ptxas -- none found (set $PTXAS / $TRITON_PTXAS_BLACKWELL_PATH)")
        else:
            ver = provision.ptxas_version(ptxas)
            need = provision.min_ptxas()
            if not ver or provision._version_tuple(ver) < provision._version_tuple(need):
                missing.append(f"ptxas {ver or '?'} at {ptxas} -- need >= {need} for --apply-controls")
    except ImportError as e:
        missing.append(f"ptx_anneal -- not importable from {_repo_root()} ({e})")

    return (not missing), missing


def _adapter_path() -> str:
    """The engine adapter to drive. Same knob and same bundled default as the shipped CLI.

    Deliberately NOT a second adapter: the two channels must run the same engine with the same
    budget, or their results are not comparable and the decision table above is worthless.
    """
    from ptx_anneal import cli  # pyre-ignore[21]

    return os.environ.get("PTX_ANNEAL_ENGINE_ADAPTER") or cli._bundled_adapter()


def _run_probe(python: str, kernel: str, ptxas: str, work: str, args) -> tuple[float | None, dict, str | None]:
    """The baseline pass: no ACF. Returns ``(baseline_ms, identity, ref_path)``.

    One invocation does three jobs, because all three need the kernel imported and run once:
    refuse autotune, capture kernel identity for the fingerprint, and -- when the spec has no
    ``ref_fn`` -- dump the untuned output to serve as the (weaker) reference for every candidate.
    """
    identity_path = os.path.join(work, f"{kernel}.identity.json")
    ref_path = os.path.join(work, f"{kernel}.ref.pt")
    cmd = [
        python,
        "-m",
        "e2e.triton_native.score",
        "--kernel",
        kernel,
        "--task",
        kernel,
        "--acf",
        "NONE",
        "--ptxas",
        ptxas,
        "--warmup",
        str(args.warmup),
        "--rep",
        str(args.rep),
        "--oracle",
        args.oracle,
        "--atol",
        str(args.atol),
        "--probe-out",
        identity_path,
        "--ref-dump",
        ref_path,
    ]
    # Bound the baseline probe the same way a candidate is bounded, or a hung untuned kernel (bad
    # body, driver stall) blocks the whole run with no way for --task-timeout to interrupt it. The
    # probe does strictly more than a candidate (import + refuse-autotune + identity + ref dump on top
    # of the timed run), so it gets grace on the per-candidate budget rather than the bare value.
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=_repo_root(),
            env=_child_env(),
            timeout=args.task_timeout + 60,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        # text=True makes this a str, but TimeoutExpired.stderr is typed str|bytes|None either way.
        sys.stderr.write(e.stderr.decode(errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or ""))
        raise SystemExit(
            f"triton-native: probe for {kernel!r} exceeded {args.task_timeout + 60:.0f}s and was "
            "killed -- the untuned baseline hung before any ACF was applied (bad kernel body or a "
            "driver stall). Nothing was searched."
        ) from e
    sys.stderr.write(out.stderr)
    if out.returncode != 0:
        # A non-zero exit is a refusal (autotune, unknown kernel, dirty env), not an invalid
        # candidate -- surface it verbatim and stop. This is the message the user needs to act on.
        raise SystemExit(out.stdout.strip() or f"triton-native: probe failed for {kernel!r} (exit {out.returncode})")

    identity = {}
    if os.path.exists(identity_path):
        with open(identity_path) as f:
            identity = json.load(f)

    last = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    baseline_ms = float(last.split()[1]) if last.startswith("MS ") else None
    # The dumped output is only USED as a reference when the spec has no independent ref_fn.
    ref = ref_path if (identity.get("has_independent_ref") is False and os.path.exists(ref_path)) else None
    return baseline_ms, identity, ref


def _time_with_opts(python: str, kernel: str, ptxas: str, opts: str | None, args) -> float | None:
    """Time the kernel with raw PTXAS_OPTIONS (or none). Returns ms, or None if it did not run."""
    cmd = [
        python,
        "-m",
        "e2e.triton_native.score",
        "--kernel",
        kernel,
        "--task",
        kernel,
        "--acf",
        "NONE",
        "--ptxas",
        ptxas,
        "--warmup",
        str(args.warmup),
        "--rep",
        str(args.rep),
        "--oracle",
        args.oracle,
        "--atol",
        str(args.atol),
    ]
    if opts:
        # `--ptxas-opts=-O0`, not two argv entries: a value starting with "-" is parsed as a flag.
        cmd.append(f"--ptxas-opts={opts}")
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=_repo_root(),
            env=_child_env(),
            timeout=args.task_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None
    last = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
    return float(last.split()[1]) if last.startswith("MS ") else None


def injection_canary(python: str, kernel: str, ptxas: str, args) -> tuple[bool, str]:
    """Prove PTXAS_OPTIONS actually changes what ptxas emits, before trusting any ACF score.

    NVIDIA's booster-pack skill makes this a mandatory pre-flight, using the debug pack's O0/O3 ACFs.
    That pack is a release asset we do not have here, so this uses plain ``-O0``/``-O3`` down the
    *same* ``PTXAS_OPTIONS`` path: same env var, same CUDAOptions field, same ptxas command line.

    Why it is worth doing on top of :func:`~e2e.triton_native.score.verify_acf_applied`: that check
    proves the flag *arrived*. This one proves ptxas *acted on it*. An ignored ACF and a genuinely
    no-op ACF produce identical scores, and only a flag with a known-bad effect can tell them apart.

        -O0  must be SLOWER than baseline  -- otherwise the option is not reaching ptxas at all
        -O3  must MATCH baseline           -- the default is already -O3
    """
    base = _time_with_opts(python, kernel, ptxas, None, args)
    o0 = _time_with_opts(python, kernel, ptxas, "-O0", args)
    o3 = _time_with_opts(python, kernel, ptxas, "-O3", args)
    if base is None or o0 is None or o3 is None:
        return False, f"a probe did not run (baseline={base}, -O0={o0}, -O3={o3})"
    if o0 <= base * 1.05:
        return False, (
            f"-O0 did not regress ({o0:.4f}ms vs baseline {base:.4f}ms). PTXAS_OPTIONS is NOT "
            "reaching ptxas, so every ACF score would really be the untuned kernel."
        )
    drift = abs(o3 - base) / base
    note = f"baseline={base:.4f}ms -O0={o0:.4f}ms ({o0 / base:.2f}x) -O3={o3:.4f}ms ({drift * 100:+.1f}%)"
    if drift > 0.05:
        # Not fatal: -O3 is the default, so a gap means measurement noise or a non-default baseline.
        return True, note + "  [WARN: -O3 differs from baseline by >5%]"
    return True, note


def _search(python: str, kernel: str, ptxas: str, ref: str | None, work: str, args) -> dict:
    """Run the engine adapter with SCORE_CMD pointed at this channel's scorer."""
    score_cmd = [
        python,
        "-m",
        "e2e.triton_native.score",
        "--kernel",
        kernel,
        "--oracle",
        args.oracle,
        "--atol",
        str(args.atol),
    ]
    if ref:
        score_cmd += ["--ref", ref]

    result_path = os.path.join(work, f"{kernel}.result.json")
    env = _child_env()
    env["PTX_ANNEAL_SCORE_CMD"] = json.dumps(score_cmd)
    # The adapter requires a task; in this channel the unit of work is a kernel name. The scorer
    # cross-checks it against --kernel, so a mismatch fails loudly rather than scoring the wrong one.
    env["PTX_ANNEAL_TASK"] = kernel
    env["PTX_ANNEAL_PTXAS"] = ptxas
    env["PTX_ANNEAL_RESULT"] = result_path
    env["PTX_ANNEAL_WARMUP"] = str(args.warmup)
    env["PTX_ANNEAL_REP"] = str(args.rep)
    env["CIQ_GENERATIONS"] = str(args.generations)
    env["CIQ_POOL"] = str(args.pool)
    env["CIQ_TASK_TIMEOUT"] = str(args.task_timeout)
    if args.clock_mhz:
        env["CIQ_CLOCK_MHZ"] = str(args.clock_mhz)
    # Doc conformance: derive the search space from the LOCAL ptxas version rather than pinning a
    # catalog tag. A pinned tag is reproducible but is also a deviation NVIDIA can point at, and
    # closing cheap deviations is what buys the right to file a bug. An explicit CIQ_SS_* in the
    # caller's environment still wins -- this only supplies the default.
    env.setdefault("CIQ_SS_VERSION", args.ptxas_version or "")
    if not env.get("CIQ_SS_VERSION"):
        del env["CIQ_SS_VERSION"]
    # Flags beat the environment beats the doc default. An explicit .bin short-circuits the catalog
    # entirely in the adapter, so it is the one knob that makes the others irrelevant.
    if getattr(args, "ss", None):
        env["PTX_ANNEAL_SS"] = os.path.abspath(args.ss)
    if getattr(args, "ss_tag", None):
        env["CIQ_SS_TAG"] = args.ss_tag
    if getattr(args, "ss_variant", None):
        env["CIQ_SS_VARIANT"] = args.ss_variant

    proc = subprocess.run([python, _adapter_path()], env=env, cwd=_repo_root(), check=False)
    if proc.returncode != 0:
        raise SystemExit(f"triton-native: engine adapter exited {proc.returncode} for {kernel!r}")
    with open(result_path) as f:
        return json.load(f)


def _fingerprint(kernel: str, identity: dict, ptxas: str, ver: str | None, res: dict, ref: str | None, args) -> dict:
    from ptx_anneal import fingerprint as fp_mod  # pyre-ignore[21]

    from .score import SEED

    weaker = identity.get("has_independent_ref") is False
    return fp_mod.build(
        channel="triton-native",
        kernel={
            "name": kernel,
            "shapes": identity.get("shapes"),
            "dtypes": identity.get("dtypes"),
            "pinned_config": identity.get("pinned_config"),
            "autotune": False,  # enforced: refuse_autotune() ran before anything was measured
            "autotune_bypassed": identity.get("autotune_bypassed"),
        },
        # The fbtriton that compiled every candidate. In this channel the frontend is a live input to
        # each candidate, so leaving it out would let two different Triton builds produce the same
        # digest. `acf_mechanism` names how the ACF is delivered rather than quoting the probe's own
        # ptx_options (which is null -- the probe is the no-ACF baseline); each candidate verifies
        # its own value at compile time via verify_acf_applied().
        frontend={
            **(identity.get("frontend") or {}),
            "acf_mechanism": "PTXAS_OPTIONS=--apply-controls=<acf>",
            "acf_applied_verified": True,
        },
        ptxas={"path": ptxas, "version": ver, "flags": fp_mod.ptxas_flags()},
        search_space=res.get("search_space") or {},
        engine={**(res.get("engine_info") or {}), "budget": res.get("budget") or {}},
        objective={
            "metric": "min_ms",
            "seed": SEED,
            "timing": {
                "method": "triton.testing.do_bench",
                "warmup": args.warmup,
                "rep": args.rep,
                "return_mode": "mean",
            },
            "correctness": {
                "oracle": ("baseline-output (self-referential)" if weaker else "ref_fn"),
                "strength": "weaker" if weaker else "independent",
                "comparison": args.oracle,
                "atol": args.atol if args.oracle == "allclose" else None,
                "ref_dump": ref,
            },
        },
    )


def _verdict(kernel: str, baseline_ms: float | None, res: dict, identity: dict, args_task_timeout) -> bool:
    """Print the outcome and return whether this kernel PASSED (>= 1 valid candidate)."""
    from ..registry import WEAKER_VERDICT

    valid = int(res.get("valid") or 0)
    evaluated = int(res.get("evaluated") or 0)
    timed_out = int(res.get("timeout") or 0)
    best_ms = res.get("best_ms")
    passed = valid >= 1

    win = ""
    if baseline_ms and isinstance(best_ms, (int, float)):
        win = f" best={best_ms:.4f}ms win={(baseline_ms - best_ms) / baseline_ms * 100:+.2f}%"
    base = f"{baseline_ms:.4f}ms" if baseline_ms else "FAILED"
    print(
        f"triton-native: {'PASS' if passed else 'FAIL'} kernel={kernel} "
        f"valid={valid}/{evaluated} timeout={timed_out} baseline={base}{win}"
    )
    if timed_out:
        # Never at debug level. A timed-out candidate is scope the search silently lost, and enough
        # of them turn "this kernel has no headroom" into a measurement artefact.
        print(
            f"triton-native: {timed_out}/{evaluated} candidates hit --task-timeout "
            f"({args_task_timeout}s) and were killed. Each candidate is a fresh process that imports "
            "torch+fbtriton before it can measure anything, so a timeout tuned for an in-process "
            "objective is too tight here -- raise --task-timeout before reading anything into this."
        )
    if res.get("engine_error"):
        print(f"triton-native: PARTIAL -- the engine did not finish cleanly: {res['engine_error']}")
    if identity.get("has_independent_ref") is False:
        # Never at debug level: a green verdict whose correctness check was self-referential must be
        # as visible as a red one, or the weaker evidence quietly becomes the same as the stronger.
        print(f"triton-native: {WEAKER_VERDICT}")
    if not passed:
        print(
            "triton-native: 0 valid candidates. Per the decision table this is a CompileIQ issue "
            "(the engine / search space / --apply-controls), not a Magnon one -- the PTX-direct "
            "channel was not involved. Confirm by diffing this fingerprint against the torch-free run."
        )
    return passed


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m e2e.triton_native.runway",
        description="Score a registered kernel through fbtriton under ptxas --apply-controls.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="Defaults follow NVIDIA's CompileIQ doc example; deviations are recorded in the fingerprint.",
    )
    p.add_argument("--kernel", action="append", default=[], help="registered kernel (repeatable)")
    p.add_argument("--no-canary", dest="canary", action="store_false", help=f"skip the {CANARY} canary")
    p.add_argument(
        "--no-injection-canary",
        dest="injection_canary",
        action="store_false",
        help="skip the -O0/-O3 pre-flight that proves PTXAS_OPTIONS reaches ptxas",
    )
    p.add_argument("--oracle", choices=["allclose", "relerr"], default="allclose", help="correctness comparison")
    p.add_argument("--atol", type=float, default=DOC_ATOL, help="allclose absolute tolerance")
    p.add_argument("--generations", type=int, default=DOC_GENERATIONS)
    p.add_argument("--pool", type=int, default=DOC_POOL, help="engine pool size (must be > 5)")
    p.add_argument("--task-timeout", dest="task_timeout", type=float, default=DOC_TASK_TIMEOUT)
    p.add_argument("--warmup", type=int, default=DOC_WARMUP, help="do_bench warmup window, MILLISECONDS")
    p.add_argument("--rep", type=int, default=DOC_REP, help="do_bench measurement window, MILLISECONDS")
    p.add_argument(
        "--clock-mhz",
        dest="clock_mhz",
        type=int,
        default=B200_MAX_SM_MHZ,
        help="lock SM clocks around the search (0 = leave clocks alone; the doc calls locking crucial)",
    )
    g = p.add_argument_group(
        "search space",
        "Which catalog the engine searches. An explicit --ss wins; otherwise the version is derived "
        "from the local ptxas, per NVIDIA's example. Whatever is resolved lands in the fingerprint, "
        "so two runs on different catalogs can never be mistaken for the same experiment.",
    )
    g.add_argument("--ss", help="explicit search-space .bin (overrides every catalog selector)")
    g.add_argument("--ss-tag", dest="ss_tag", help="published catalog tag, e.g. search-spaces-2026.05.22")
    g.add_argument("--ss-variant", dest="ss_variant", help="catalog variant (default: the catalog's own)")
    p.add_argument("--work", help="run directory (default: a temp dir)")
    p.add_argument("--acf-out", dest="acf_out", help="write each winning ACF here as <kernel>.acf")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.pool <= 5:
        raise SystemExit(f"triton-native: --pool must be > 5 (engine-validated); got {args.pool}")

    ok, missing = availability()
    if not ok:
        # Mirrors e2e.sh's MODE=full degradation: say precisely what is absent, note that the rest of
        # the repo is unaffected, and exit 2 rather than pretending to have run.
        print("triton-native: unavailable on this box. Missing:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print(
            "This channel is optional and on-demand; the ptx-anneal factory (e2e.sh MODE=tune) is "
            "unaffected and still works here.",
            file=sys.stderr,
        )
        return 2

    _ensure_repo_on_path()
    from ptx_anneal import fingerprint as fp_mod  # pyre-ignore[21]
    from ptx_anneal import provision  # pyre-ignore[21]

    from .score import check_env_clean

    check_env_clean()

    ptxas = os.environ.get("PTXAS") or provision.find_ptxas()
    ver = provision.ptxas_version(ptxas)
    args.ptxas_version = ver

    # Drop an explicitly-named canary only when the canary is ALSO running implicitly -- otherwise
    # `--kernel doc_matmul --no-canary` (the way you tune the canary itself) silently runs nothing.
    named = [k for k in args.kernel if not (args.canary and k == CANARY)]
    kernels = ([CANARY] if args.canary else []) + named
    if not kernels:
        raise SystemExit("triton-native: nothing to run (--no-canary with no --kernel)")

    work = args.work or tempfile.mkdtemp(prefix="triton_native_")
    os.makedirs(work, exist_ok=True)
    print(f"triton-native: ptxas={ptxas} (V{ver or '?'}) kernels={kernels} work={work}")

    all_passed = True
    for kernel in kernels:
        if args.injection_canary:
            ok, note = injection_canary(sys.executable, kernel, ptxas, args)
            print(f"triton-native: injection-canary {'PASS' if ok else 'FAIL'} kernel={kernel} {note}")
            if not ok:
                # Refuse rather than measure: without proven injection every candidate score is the
                # untuned kernel, and the search would report wins it never made.
                all_passed = False
                if kernel == CANARY:
                    return 1
                continue

        baseline_ms, identity, ref = _run_probe(sys.executable, kernel, ptxas, work, args)
        if baseline_ms is None:
            # The UNTUNED kernel already failed -- it did not run, or it did not match its own
            # reference. Searching from here would attribute a pre-existing bug to an ACF, which is
            # exactly the kind of unattributable result this channel exists to prevent.
            print(
                f"triton-native: FAIL kernel={kernel} -- the no-ACF baseline did not produce a valid "
                "result. Nothing was searched: the kernel or its ref_fn is broken before any ACF is "
                "applied (see the [score] line above).",
                file=sys.stderr,
            )
            all_passed = False
            if kernel == CANARY:
                return 1
            continue

        res = _search(sys.executable, kernel, ptxas, ref, work, args)
        passed = _verdict(kernel, baseline_ms, res, identity, args.task_timeout)
        all_passed = all_passed and passed

        if args.acf_out and res.get("acf"):
            os.makedirs(args.acf_out, exist_ok=True)
            dest = os.path.join(args.acf_out, f"{kernel}.acf")
            with open(dest, "wb") as f:
                f.write(bytes.fromhex(res["acf"]))
            print(f"triton-native: wrote {dest}")

        print(fp_mod.render(_fingerprint(kernel, identity, ptxas, ver, res, ref, args)))

        if kernel == CANARY and not passed:
            # The canary is the control. If NVIDIA's own known-good example cannot produce a single
            # valid candidate here, nothing measured afterwards means anything -- stop rather than
            # generate results that will be misread as kernel-specific findings.
            print(
                "triton-native: CANARY FAILED -- the toolchain itself is unhealthy (ptxas / engine / "
                "search space / this box), so any result for the remaining kernels would be "
                "uninterpretable. Fix the canary first.",
                file=sys.stderr,
            )
            return 1

    if not args.work:
        shutil.rmtree(work, ignore_errors=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
