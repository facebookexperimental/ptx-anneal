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
    CIQ_SS_VERSION / CIQ_SS_VARIANT / CIQ_SS_TAG   catalog selector, when PTX_ANNEAL_SS is unset

Per candidate the engine hands a hex ACF; we shell out to `SCORE_CMD _? --task .. --acf f --ptxas ..`
which prints "MS <float>" (valid) or "INVALID". At the end we write
{"best_ms","evaluated","valid","acf","search_space","engine_info"} (acf = winning hex or null) to
PTX_ANNEAL_RESULT.

We also log the engine identity and resolved search space to stderr up front: both are *inputs* to
tuning that the harness cannot see (it never imports the engine, and the engine may live in another
interpreter), so an unexplained result is otherwise hard to attribute.
"""

import json
import os
import subprocess
import sys
import tempfile


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
        f"search space: {ss_meta.get('filename') or ss_meta.get('path')} "
        f"[{src}{f' {tag}' if tag else ''}]",
        file=sys.stderr,
        flush=True,
    )

    seen = {"n": 0, "valid": 0}

    def objective(acf, *args, **kwargs):
        acf_hex = acf if isinstance(acf, str) else str(acf)
        fd, path = tempfile.mkstemp(suffix=".acf")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(bytes.fromhex(acf_hex))
            cmd = [*score_cmd, "--task", task, "--acf", path,
                   "--ptxas", ptxas, "--warmup", str(warmup), "--rep", str(rep)]
            out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        finally:
            os.unlink(path)
        seen["n"] += 1
        last = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
        ok = last.startswith("MS ")
        ms = float(last.split()[1]) if ok else None
        if ok:
            seen["valid"] += 1
        print(
            json.dumps({"kind": "candidate", "index": seen["n"], "ms": ms, "ok": ok, "acf": acf_hex[:16]}),
            file=sys.stderr,
            flush=True,
        )
        return ms if ok else INVALID_SCORE

    cfg = SearchConfiguration(problem_type="min", generations=gens, pool_size=pool)
    result = Search(objective_function=objective, search_space=search_space, search_config=cfg).start()

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
        "acf": acf_hex,
        "search_space": ss_meta,
        "engine_info": engine_info,
    }
    with open(result_path, "w") as f:
        json.dump(payload, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
