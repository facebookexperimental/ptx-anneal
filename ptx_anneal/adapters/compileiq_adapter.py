# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""NVIDIA CompileIQ engine adapter (standalone).

Run by the ptx_anneal harness under the *engine's* interpreter (``--engine``). Imports ONLY
``compileiq`` + stdlib -- never ``ptx_anneal`` -- so the harness can stay a frozen, engine-free binary
and the engine can live in any interpreter/venv.

Contract (all via env, set by the harness):
    PTX_ANNEAL_SCORE_CMD   JSON argv prefix for the harness scorer, e.g. ["/path/ptx_anneal","_score"]
    PTX_ANNEAL_TASK        task dir (kernel.ptx + spec.json)
    PTX_ANNEAL_PTXAS       ptxas to assemble/apply ACFs with
    PTX_ANNEAL_SS          search-space .bin (OPTIONAL -- unset means "fetch the published catalog")
    PTX_ANNEAL_RESULT      path to write the result JSON to
    PTX_ANNEAL_WARMUP / PTX_ANNEAL_REP   benchmark iters (optional)
    CIQ_GENERATIONS / CIQ_POOL           engine budget (optional)
    CIQ_TASK_TIMEOUT                     per-candidate timeout, seconds (optional; unset = no timeout)
    CIQ_CLOCK_MHZ / CIQ_CLOCK_SUDO       lock GPU clocks around the search (optional; unset = don't touch)
    CIQ_SS_VERSION / CIQ_SS_VARIANT / CIQ_SS_TAG   catalog selector, when PTX_ANNEAL_SS is unset

Per candidate the engine hands a hex ACF; we shell out to `SCORE_CMD _? --task .. --acf f --ptxas ..`
which prints "MS <float>" (valid) or "INVALID". At the end we write
{"best_ms","evaluated","valid","acf","search_space","engine_info","budget"} (acf = winning hex or
null) to PTX_ANNEAL_RESULT.

``budget`` reports what the search ACTUALLY ran with -- not what was requested -- so the caller can
put it in a search-problem fingerprint. Two channels driving this same adapter are only comparable
if their budgets are equal, and "equal" has to be observed rather than assumed.

The timeout and clock knobs default to OFF so the shipped torch-free path behaves exactly as before.
They exist because NVIDIA's documented example uses both (``start(task_timeout=20)`` and
``gpu_benchmark_mode(clock_mhz=...)``, which their doc calls crucial), and "we followed your doc" is
what makes a CompileIQ bug report credible -- so a channel that wants to conform can, through a
recorded knob rather than a fork.

We also log the engine identity and resolved search space to stderr up front: both are *inputs* to
tuning that the harness cannot see (it never imports the engine, and the engine may live in another
interpreter), so an unexplained result is otherwise hard to attribute.
"""

import contextlib
import json
import os
import subprocess
import sys
import tempfile

# Slack between our per-candidate kill and the engine's own task_timeout, so the two cannot fire at
# the same moment and leave a candidate abandoned rather than cleanly recorded as invalid.
_TIMEOUT_GRACE_S = 5


def _env_num(name: str, cast):
    """Read an optional numeric knob. Unset or empty means "not requested", not "zero"."""
    raw = os.environ.get(name, "").strip()
    return cast(raw) if raw else None


def _benchmark_mode(clock_mhz: int | None):
    """Lock the GPU clocks around the search when asked, otherwise do nothing at all.

    Unset is a genuine no-op (``nullcontext``) rather than ``gpu_benchmark_mode(None)``: the latter
    warns and, with ``raise_on_failure=True``, raises. The default must not perturb the existing
    torch-free path.

    ``raise_on_failure=False`` mirrors NVIDIA's example: clock locking needs elevated privileges, and
    a box that cannot lock should still produce a (noisier) result -- the *fact* that it did not lock
    is visible in the fingerprint's ``env.clock_event_reasons``/``sm_clock_mhz``, which is the point.
    """
    if clock_mhz is None:
        return contextlib.nullcontext()
    from compileiq.utils.gpu import gpu_benchmark_mode

    sudo = os.environ.get("CIQ_CLOCK_SUDO", "1") not in ("", "0")
    return gpu_benchmark_mode(clock_mhz=clock_mhz, with_sudo=sudo, raise_on_failure=False)


def _engine_info() -> dict:
    """Identify the engine actually in use: name, version, import path, interpreter.

    The harness cannot work this out for itself -- it must never import the engine, and the adapter
    may be running under a different interpreter (``--engine-python``). And unlike ptxas there is no
    path to print: an installed library is whatever ``import`` resolved to, which on a box with a
    venv, an editable install and a site-packages copy is genuinely ambiguous. So report it here,
    and hand it back so the admitted ACF records which engine produced it.
    """
    import compileiq.ciq

    try:
        from importlib import metadata

        version = metadata.version("compileiq")
    except Exception:  # noqa: BLE001 - version is diagnostics only; never fail a run over it
        version = "unknown"
    # compileiq is a namespace package (__file__ is None), so locate it via a real submodule.
    path = os.path.dirname(os.path.abspath(compileiq.ciq.__file__))
    return {"name": "compileiq", "version": version, "path": path, "python": sys.executable}


def _resolve_search_space(ss: str):
    """Return ``(provider, provenance)`` for the engine's search space.

    An explicit ``PTX_ANNEAL_SS`` always wins -- it is how you pin a dev build or an out-of-band
    asset. With none given, CompileIQ resolves its published catalog for (version, variant, tag) and
    caches the artifact under ``~/.cache/compileiq/<tag>/``, so a plain checkout needs neither a
    CompileIQ clone nor ``--ss``. ``CIQ_SEARCH_SPACES_DIR`` redirects that lookup at an offline
    mirror without changing anything here.

    The provenance dict is handed back to the harness and lands in the ACF's sidecar. That matters
    precisely because the default tag is "latest": the search space is an *input* to tuning, so
    without recording it two ACFs for the same kernel could come from different catalogs with
    nothing on disk saying which.
    """
    from compileiq.search_spaces.compilers import LocalSearchSpaceBin, PtxasSearchSpace

    if ss:
        if not os.path.exists(ss):
            raise SystemExit(f"compileiq_adapter: search space not found: {ss!r} (set --ss / PTX_ANNEAL_SS)")
        return LocalSearchSpaceBin(ss), {"source": "PTX_ANNEAL_SS", "path": os.path.abspath(ss)}

    # Forward version/variant/tag only when set, so the catalog's own default stays the single
    # source of truth instead of being duplicated (and left to rot) here.
    sel = {
        k: v
        for k, v in (
            ("version", os.environ.get("CIQ_SS_VERSION")),
            ("variant", os.environ.get("CIQ_SS_VARIANT")),
            ("tag", os.environ.get("CIQ_SS_TAG")),
        )
        if v
    }
    provider = PtxasSearchSpace(**sel)
    # Resolve up front rather than lazily inside Search(): a fetch failure then surfaces here with a
    # clear message instead of mid-search, and resolution_metadata is populated for the sidecar. The
    # artifact is cached, so the engine's own retrieve() costs nothing.
    provider.retrieve()
    meta = provider.resolution_metadata
    return provider, (meta.as_dict() if meta else {})


def main() -> int:
    score_cmd = json.loads(os.environ["PTX_ANNEAL_SCORE_CMD"])
    task = os.environ["PTX_ANNEAL_TASK"]
    ptxas = os.environ["PTX_ANNEAL_PTXAS"]
    ss = os.environ.get("PTX_ANNEAL_SS") or ""
    result_path = os.environ["PTX_ANNEAL_RESULT"]
    warmup = os.environ.get("PTX_ANNEAL_WARMUP", "25")
    rep = os.environ.get("PTX_ANNEAL_REP", "50")
    gens = int(os.environ.get("CIQ_GENERATIONS", "1"))
    pool = int(os.environ.get("CIQ_POOL", "8"))
    task_timeout = _env_num("CIQ_TASK_TIMEOUT", float)
    clock_mhz = _env_num("CIQ_CLOCK_MHZ", int)

    from compileiq.ciq import Search
    from compileiq.types import INVALID_SCORE, SearchConfiguration

    # Print before resolving the search space, so a fetch failure still tells you which engine tried.
    engine_info = _engine_info()
    print(
        f"engine: {engine_info['name']} {engine_info['version']} from {engine_info['path']} "
        f"(python={engine_info['python']})",
        file=sys.stderr,
        flush=True,
    )

    search_space, ss_meta = _resolve_search_space(ss)
    src = ss_meta.get("source", "?")
    tag = ss_meta.get("resolved_tag")
    print(
        f"search space: {ss_meta.get('filename') or ss_meta.get('path')} [{src}{f' {tag}' if tag else ''}]",
        file=sys.stderr,
        flush=True,
    )

    seen = {"n": 0, "valid": 0, "timeout": 0}

    def objective(acf, *args, **kwargs):
        acf_hex = acf if isinstance(acf, str) else str(acf)
        fd, path = tempfile.mkstemp(suffix=".acf")
        timed_out = False
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(bytes.fromhex(acf_hex))
            cmd = [
                *score_cmd,
                "--task",
                task,
                "--acf",
                path,
                "--ptxas",
                ptxas,
                "--warmup",
                str(warmup),
                "--rep",
                str(rep),
            ]
            try:
                # THE scorer must be bounded here, by us. The engine's own `task_timeout` bounds its
                # worker, not our grandchild process -- so without this a candidate whose kernel hangs
                # (an applied ACF can produce SASS that never returns) survives the engine giving up
                # on it, keeps the GPU busy, and makes every LATER candidate slow enough to time out
                # too. That death spiral is silent: the search just reports everything invalid.
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=task_timeout, check=False)
            except subprocess.TimeoutExpired:
                # subprocess.run has already killed the child by the time this is raised.
                timed_out = True
                out = None
        finally:
            os.unlink(path)
        seen["n"] += 1
        if timed_out:
            seen["timeout"] += 1
        last = "" if out is None or not out.stdout.strip() else out.stdout.strip().splitlines()[-1]
        ok = last.startswith("MS ")
        ms = float(last.split()[1]) if ok else None
        if ok:
            seen["valid"] += 1
        print(
            json.dumps(
                {
                    "kind": "candidate",
                    "index": seen["n"],
                    "ms": ms,
                    "ok": ok,
                    "timeout": timed_out,
                    "acf": acf_hex[:16],
                }
            ),
            file=sys.stderr,
            flush=True,
        )
        return ms if ok else INVALID_SCORE

    cfg = SearchConfiguration(problem_type="min", generations=gens, pool_size=pool)
    # exit_on_failure=False: by default the engine raises if EVERY candidate fails in the first
    # generation. For us that is not an error, it is the single most important verdict there is --
    # "this kernel survives no ACF at all". Letting it raise loses the candidate counts, the search
    # space, the budget and the fingerprint, i.e. exactly the evidence that makes the result
    # attributable. Report it; do not crash on it.
    search = Search(
        objective_function=objective,
        search_space=search_space,
        search_config=cfg,
        exit_on_failure=False,
    )
    # The engine's task_timeout is the OUTER backstop; the per-candidate kill above is the authority.
    # Give the engine a grace margin so it does not preempt us at the same instant -- we want the
    # candidate recorded as a clean INVALID (with its child reaped), not abandoned mid-flight.
    start_kwargs = {} if task_timeout is None else {"task_timeout": task_timeout + _TIMEOUT_GRACE_S}
    engine_error = None
    result = None
    try:
        with _benchmark_mode(clock_mhz):
            result = search.start(**start_kwargs)
    except RuntimeError as e:
        # Belt-and-braces behind exit_on_failure=False. An engine that dies mid-search must still
        # yield a reportable outcome: we already know how many candidates ran, how many were valid
        # and how many timed out, and throwing that away turns an informative red into an
        # unattributable one. Recorded, not swallowed -- the message travels in the payload.
        engine_error = f"{type(e).__name__}: {e}"
        print(f"engine error (results below are partial): {engine_error}", file=sys.stderr, flush=True)

    df = result.get_results() if result is not None else None
    evaluated = len(df) if df is not None else seen["n"]
    best = result.get_best_result() if result is not None else None
    best_ms = best.get("score_1", best.get("score")) if best else None
    params = best.get("params") if best else None
    acf_hex = params if (params and isinstance(best_ms, (int, float))) else None
    payload = {
        "best_ms": float(best_ms) if acf_hex else None,
        "evaluated": evaluated,
        "valid": seen["valid"],
        # Counted and reported, never just logged: a timed-out candidate is scope that was silently
        # dropped from the search, and "we tuned nothing because everything timed out" must be as
        # visible as a failure -- otherwise it reads as "no headroom found".
        "timeout": seen["timeout"],
        "acf": acf_hex,
        # Null on a clean run. Non-null means the numbers above are partial -- the caller must say so
        # rather than present them as a completed search.
        "engine_error": engine_error,
        "search_space": ss_meta,
        "engine_info": engine_info,
        # What the search actually ran with. `seed` is explicitly null, not omitted: CompileIQ's
        # SearchConfiguration exposes no seed, so the search is NOT reproducible run to run. That is
        # a property of the search problem and belongs in the fingerprint rather than being silently
        # absent from it.
        "budget": {
            "generations": gens,
            "pool_size": pool,
            "task_timeout": task_timeout,
            "clock_mhz": clock_mhz,
            "warmup": int(warmup),
            "rep": int(rep),
            "seed": None,
        },
    }
    with open(result_path, "w") as f:
        json.dump(payload, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
