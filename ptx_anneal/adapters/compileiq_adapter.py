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
    PTX_ANNEAL_SS          search-space .bin (required for CompileIQ)
    PTX_ANNEAL_RESULT      path to write the result JSON to
    PTX_ANNEAL_WARMUP / PTX_ANNEAL_REP   benchmark iters (optional)
    CIQ_GENERATIONS / CIQ_POOL           engine budget (optional)

Per candidate the engine hands a hex ACF; we shell out to `SCORE_CMD _? --task .. --acf f --ptxas ..`
which prints "MS <float>" (valid) or "INVALID". At the end we write
{"best_ms","evaluated","valid","acf"} (acf = winning hex or null) to PTX_ANNEAL_RESULT.
"""

import json
import os
import subprocess
import sys
import tempfile


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

    if not ss or not os.path.exists(ss):
        raise SystemExit(f"compileiq_adapter: search space not found: {ss!r} (set --ss / PTX_ANNEAL_SS)")

    from compileiq.ciq import Search
    from compileiq.search_spaces.compilers import LocalSearchSpaceBin
    from compileiq.types import INVALID_SCORE, SearchConfiguration

    seen = {"n": 0, "valid": 0}

    def objective(acf, *args, **kwargs):
        acf_hex = acf if isinstance(acf, str) else str(acf)
        fd, path = tempfile.mkstemp(suffix=".acf")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(bytes.fromhex(acf_hex))
            cmd = [*score_cmd, "--task", task, "--acf", path,
                   "--ptxas", ptxas, "--warmup", str(warmup), "--rep", str(rep)]
            out = subprocess.run(cmd, capture_output=True, text=True)
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
    result = Search(objective_function=objective, search_space=LocalSearchSpaceBin(ss), search_config=cfg).start()

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
    }
    with open(result_path, "w") as f:
        json.dump(payload, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
