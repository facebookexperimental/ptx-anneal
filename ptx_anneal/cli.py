# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The ``ptx_anneal`` command-line interface -- one binary for the whole ACF factory.

    ptx_anneal --task <task_dir> [--output <store>] [--ss <ss>]
    ptx_anneal doctor

Per-run surface is just the task; the engine, its search space and ptxas are all provisioning, and
each self-provisions by default (pip-installed engine, fetched search-space catalog, discovered
ptxas). ``--ss`` only exists to pin a specific search space.

The harness is engine-free: it assembles/benchmarks candidates with ptxas + the CUDA driver
(``_score``) and admits the winner. The search *engine* (NVIDIA CompileIQ, or an in-house one) is
bring-your-own, driven by a small standalone *adapter* that scores each candidate by calling this
binary's ``_score`` subcommand -- so the harness never imports the engine and can be a frozen binary.

Provisioning (env; flags override):
    PTXAS / TRITON_PTXAS_BLACKWELL_PATH   ptxas to assemble/apply ACFs with
    PTX_ANNEAL_ENGINE_PYTHON              interpreter that has the engine (default: this interpreter,
                                          i.e. install the engine into the same env as ptx_anneal)
    PTX_ANNEAL_ENGINE_ADAPTER             bring-your-own engine adapter (default: bundled CompileIQ)

Version gate: ``ptxas >= MIN_PTXAS`` is required and not overridable -- ``--apply-controls`` (how an
ACF is applied at all) is GA only from that version on, so an older ptxas cannot consume what this
factory produces. A lower ptxas is a hard error, not a fallback.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

from . import provision
from . import task as task_mod
from .store import LocalStore, default_store_root
from .target import load_target

INVALID = float("inf")


def _resolve_ptxas(args) -> str | None:
    # An explicit flag/env is honoured verbatim; everything else goes through the shared discovery
    # path (env -> PATH -> pip-installed nvidia-cuda-nvcc wheel) so `doctor` and `tune` can never
    # disagree about which ptxas this box would use.
    return getattr(args, "ptxas", None) or os.environ.get("PTXAS") or provision.find_ptxas()


def _harness_cmd() -> list[str]:
    """How the adapter re-invokes this binary's ``_score``: the frozen executable, or ``python -m``."""
    if getattr(sys, "frozen", False):
        return [sys.executable]
    return [sys.executable, "-m", "ptx_anneal.cli"]


def _bundled_adapter() -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "adapters", "compileiq_adapter.py")


# -- _score: the per-candidate scorer (engine-free; cuda-python) ----------------------------------
def _cmd_score(args) -> int:
    if args.ptxas:
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = args.ptxas
    from .run import cudapy_worker

    acf = None if args.acf in (None, "NONE") else args.acf
    try:
        ms = cudapy_worker.score(args.task, acf, args.warmup, args.rep, args.bench)
        print(f"MS {float(ms)}")
    except Exception as e:  # noqa: BLE001 - any failure is INVALID to the caller
        sys.stderr.write(f"[_score] {type(e).__name__}: {e}\n")
        print("INVALID")
    return 0


def _score_one(task_dir: str, acf: str | None, warmup: int, rep: int, bench: str = "cudagraph") -> float:
    from .run import cudapy_worker

    try:
        return float(cudapy_worker.score(task_dir, acf, warmup, rep, bench))
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[baseline] {type(e).__name__}: {e}\n")
        return INVALID


# -- tune: baseline -> spawn engine adapter -> admit ----------------------------------------------
def _cmd_tune(args) -> int:
    # --task is enforced by the parser (build_parser), so no check here.
    ptxas = _resolve_ptxas(args)
    if not ptxas:
        raise SystemExit("ptx_anneal: no ptxas (set --ptxas / PTXAS / TRITON_PTXAS_BLACKWELL_PATH, or add to PATH)")
    os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas

    # Engine interpreter: default to THIS interpreter (install the engine into the same env as
    # ptx_anneal). Override with --engine-python / $PTX_ANNEAL_ENGINE_PYTHON only for a separate env
    # (e.g. when the harness is a frozen binary whose own interpreter can't import the engine).
    engine_python = args.engine_python or os.environ.get("PTX_ANNEAL_ENGINE_PYTHON") or sys.executable

    # Hard version gate. An ACF is applied with `ptxas --apply-controls`, GA only from MIN_PTXAS on,
    # so an older ptxas can neither assemble nor consume one -- there is no fallback to offer.
    ver = provision.ptxas_version(ptxas)
    need = provision.min_ptxas()
    if not ver or provision._version_tuple(ver) < provision._version_tuple(need):
        raise SystemExit(
            f"ptx_anneal: ptxas {ver or '?'} < {need}. Applying an ACF (`ptxas --apply-controls`) is "
            f"GA from {need} on; point PTXAS / TRITON_PTXAS_BLACKWELL_PATH at a {need}+ ptxas."
        )
    # The engine stays bring-your-own: the bundled CompileIQ adapter is only the default.
    adapter = args.engine_adapter or os.environ.get("PTX_ANNEAL_ENGINE_ADAPTER") or _bundled_adapter()
    if not os.path.exists(adapter):
        raise SystemExit(f"ptx_anneal: engine adapter not found: {adapter!r}")

    t = task_mod.load(args.task)
    target = load_target(args.target)
    arch, ir_hash = target.arch(t.ir), target.ir_hash(t.ir)

    print(
        f"ptxas={ptxas} (V{ver or '?'}) ; engine-python={engine_python} ; "
        f"adapter={os.path.basename(adapter)} ; task={args.task}"
    )

    # 1) baseline (no ACF), scored in-process by the harness.
    baseline_ms = _score_one(args.task, None, args.warmup, args.rep, args.bench)
    baseline_ok = math.isfinite(baseline_ms)
    print(f"baseline: {f'{baseline_ms:.4f} ms' if baseline_ok else 'FAILED'} (bench={args.bench})")

    # 2) run the engine in its own interpreter via the adapter; it scores candidates through `_score`.
    result_fd, result_path = tempfile.mkstemp(suffix=".json")
    os.close(result_fd)
    env = dict(os.environ)
    env["PTX_ANNEAL_SCORE_CMD"] = json.dumps([*_harness_cmd(), "_score", "--bench", args.bench])
    env["PTX_ANNEAL_TASK"] = os.path.abspath(args.task)
    env["PTX_ANNEAL_PTXAS"] = ptxas
    env["PTX_ANNEAL_RESULT"] = result_path
    env["PTX_ANNEAL_WARMUP"] = str(args.warmup)
    env["PTX_ANNEAL_REP"] = str(args.rep)
    if args.ss:
        env["PTX_ANNEAL_SS"] = os.path.abspath(args.ss)
    try:
        proc = subprocess.run([engine_python, adapter], env=env)
        if proc.returncode != 0:
            raise SystemExit(f"ptx_anneal: engine adapter exited {proc.returncode}")
        with open(result_path) as f:
            res = json.load(f)
    finally:
        if os.path.exists(result_path):
            os.unlink(result_path)

    evaluated = int(res.get("evaluated") or 0)
    valid = res.get("valid")
    best_ms = res.get("best_ms")
    acf_hex = res.get("acf")
    engine_name = os.path.basename(adapter).removesuffix(".py").removesuffix("_adapter")

    # 3) admit. Normally only a finite candidate that BEAT the baseline is admitted. PTX_ANNEAL_FORCE_ADMIT
    #    (default off) admits the best *valid* candidate even without a win -- this is a SMOKE knob: it
    #    guarantees a real (different-from-baseline) ACF lands in the store so the consume/apply path is
    #    always exercised end-to-end, even on kernels/shapes with no reproducible ptxas headroom.
    force_admit = os.environ.get("PTX_ANNEAL_FORCE_ADMIT", "") not in ("", "0")
    best_finite = bool(acf_hex) and isinstance(best_ms, (int, float)) and math.isfinite(best_ms)
    is_win = best_finite and baseline_ok and best_ms < baseline_ms
    if best_finite and baseline_ok and (is_win or force_admit):
        win = (baseline_ms - best_ms) / baseline_ms
        meta = {
            "target": target.name, "arch": arch, "ir_hash": ir_hash, "entry": t.spec.get("entry"),
            "baseline_ms": baseline_ms, "best_ms": best_ms, "search_win": win, "forced": not is_win,
            "evaluated": evaluated, "engine": engine_name, "valid": valid, "ptxas_version": ver,
            # Which search space produced this ACF. The search space is a tuning *input*, and the
            # engine resolves it to "latest" by default, so recording it is what keeps an admitted
            # ACF explainable (and a pinned CIQ_SS_TAG meaningful) after the catalog moves on.
            "search_space": res.get("search_space") or None,
            # Which engine build produced it. `engine` above is just the adapter's name; an engine is
            # an installed library with no path to print, so the adapter reports its own identity.
            "engine_info": res.get("engine_info") or None,
        }
        # Write VERSION-TAGGED ("<ir_hash>.<ptxas>.acf"): an ACF is bound to the ptxas that produced it
        # (a mismatched ptxas rejects it), so scope it by ptxas version -- one kernel keeps a distinct
        # tuned ACF per ptxas, so ACFs from different ptxas versions coexist instead
        # of overwriting each other. The triton-core consumer (magnon.store.acf_path) reads the same
        # version-tagged path using ITS ptxas version; the kernel identity (ir_hash) stays version-free.
        store_path = LocalStore(args.output or default_store_root()).write(
            target.name, arch, ir_hash, bytes.fromhex(acf_hex), meta, toolchain_version=ver or "",
        )
        forced_tag = "" if is_win else " [FORCE_ADMIT: no genuine win; admitted to exercise consume]"
        print(
            f"admitted ACF for {target.name}/{arch}/{ir_hash[:16]} "
            f"(baseline={baseline_ms:.4f}ms best={best_ms:.4f}ms search-win={win * 100:+.2f}%){forced_tag} "
            f"engine={engine_name} evaluated={evaluated} valid={valid} -> {store_path}"
        )
        print("note: the search-time win is noisy; the trustworthy decision is the consumer's A/B vs baseline.")
    else:
        reason = (
            "no baseline" if not baseline_ok
            else "no valid candidate" if not best_finite
            else "no candidate beat the baseline (set PTX_ANNEAL_FORCE_ADMIT=1 to admit best anyway)"
        )
        print(f"no candidate admitted ({reason}); engine={engine_name} evaluated={evaluated} valid={valid}")
    return 0


def _cmd_doctor() -> int:
    print(provision.format_report(provision.report()))
    return 0


# `doctor` and `_score` are intercepted in main() before the parser sees them, so argparse cannot
# advertise them on its own -- spell the real surface out here. `_score` stays undocumented: it is
# the internal per-candidate scorer the adapter calls back into, not a user command.
_USAGE = "ptx_anneal --task <dir> [--output <store>] [options]\n       ptx_anneal doctor"

_EPILOG = """\
knobs with no flag (environment only):
  PTX_ANNEAL_FORCE_ADMIT   admit the best valid candidate even when it did not beat the baseline
                           (smoke knob: guarantees a real ACF lands so consume can be exercised)
  CIQ_POOL / CIQ_GENERATIONS         engine budget (default 8 / 1 -- one batch, no evolution)
  CIQ_SS_TAG / CIQ_SS_VERSION / CIQ_SS_VARIANT   pin the published search-space catalog
  CIQ_SEARCH_SPACES_DIR    offline search-space mirror (no network)

notes:
  The exit code does NOT report admission -- 0 means the run completed, not that an ACF was
  admitted ("no headroom" is an outcome, not an error). To gate on it, match "admitted ACF"
  in the output.

  The ptxas >= 13.3 floor is fixed, not a default: `--apply-controls` is GA only from there
  on, so an older ptxas can neither produce nor consume an ACF. A pinned --ptxas is used
  verbatim even when it is too old -- the whole toolchain must agree on one ptxas.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ptx_anneal",
        usage=_USAGE,
        description="Offline ptxas ACF tuning. The engine, its search space and ptxas all "
        "self-provision, so `--task <dir>` on its own is a complete invocation.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    g = p.add_argument_group("task")
    # Required here rather than checked later: it always was mandatory, and a late SystemExit is a
    # worse error than argparse's usage message.
    g.add_argument("--task", required=True, help="a task directory (kernel.ptx + spec.json)")
    g.add_argument("--output", "--store", dest="output", help="ACF store root (default: ~/.ptx_anneal/store)")

    g = p.add_argument_group("scoring")
    g.add_argument(
        "--bench", choices=["cudagraph", "do_bench"], default="cudagraph",
        help="scoring metric: cudagraph (fast/deterministic, default) or do_bench (L2-flush median, "
        "faithful to the consumer's A/B)",
    )
    g.add_argument("--warmup", type=int, default=25, help="benchmark warmup iters (default: %(default)s)")
    g.add_argument("--rep", type=int, default=50, help="benchmark measured iters (default: %(default)s)")
    g.add_argument("--target", default="ptx", help="target (default: %(default)s)")

    g = p.add_argument_group("provisioning pins", "all optional -- each of these self-provisions")
    g.add_argument("--ptxas", help="ptxas to use (else $PTXAS / $TRITON_PTXAS_BLACKWELL_PATH / PATH / "
                                   "an installed nvidia-cuda-nvcc wheel)")
    g.add_argument("--ss", help="engine search-space path (else the engine fetches its published catalog)")
    g.add_argument(
        "--engine-python", dest="engine_python",
        help="interpreter that has the engine (default: this interpreter / $PTX_ANNEAL_ENGINE_PYTHON)",
    )
    g.add_argument(
        "--engine-adapter", dest="engine_adapter",
        help="bring-your-own engine adapter script (default: bundled CompileIQ / $PTX_ANNEAL_ENGINE_ADAPTER)",
    )
    return p


def _score_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ptx_anneal _score", description="Score one candidate (internal).")
    p.add_argument("--task", required=True)
    p.add_argument("--acf", default="NONE", help="ACF file path, or NONE for the untuned baseline")
    p.add_argument("--ptxas")
    p.add_argument("--warmup", type=int, default=25)
    p.add_argument("--rep", type=int, default=50)
    p.add_argument("--bench", choices=["cudagraph", "do_bench"], default="cudagraph")
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "doctor":
        return _cmd_doctor()
    if argv and argv[0] == "_score":
        return _cmd_score(_score_parser().parse_args(argv[1:]))
    return _cmd_tune(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
