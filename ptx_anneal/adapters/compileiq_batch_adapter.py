# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""NVIDIA CompileIQ engine adapter that scores each generation as one batch (standalone).

Selected by ``ptx_anneal --batch-scoring``. Like ``compileiq_adapter`` it imports ONLY ``compileiq`` +
stdlib, and it needs CompileIQ >= 1.1 (a custom ``Worker`` receives the whole pool). It reads the same
environment contract as ``compileiq_adapter`` plus:

    PTX_ANNEAL_COMPILE_CMD   JSON argv prefix for the harness ``_compile`` (parallel ptxas over a pool)
    PTX_ANNEAL_SERVER_CMD    JSON argv prefix for the harness ``_score_server`` (one per GPU)
    PTX_ANNEAL_BATCH_RETIME           re-time the untuned baseline every N candidates on each GPU
                                      (default 8; 0 = only at the start of each batch)
    PTX_ANNEAL_BATCH_MAX_FAILURES     benchmark-server failures after which a GPU stays in
                                      per-candidate mode for the rest of the search (default 3)
    PTX_ANNEAL_BATCH_STARTUP_TIMEOUT  seconds a server may take to build inputs and time the
                                      baseline (default 600)

Per generation:

1. Write every ACF and assemble the whole pool in one ``_compile`` call (parallel ptxas, bounded by
   the per-candidate timeout). A failed compile scores INVALID.
2. Cubins already scored -- earlier in the search or earlier in this pool -- reuse that score: the
   same machine code gets one measurement instead of several noisy ones.
3. The remaining cubins go to one long-lived ``_score_server`` per GPU, which validates each against
   the untuned kernel's outputs and times it. Each server re-times the untuned baseline at the start
   of the batch and every PTX_ANNEAL_BATCH_RETIME candidates.
4. Each candidate is reported as ``ms * reference / same_gpu_baseline``, where ``same_gpu_baseline``
   is the median of that GPU's baseline timings so far and ``reference`` is the first baseline of the
   search: still milliseconds, but GPU-to-GPU differences and slow drift no longer look like wins,
   and one outlying baseline cannot rescale every later candidate. The harness re-times the winner
   in its own process before admitting it, so the admitted number is never a rescaled one.

Fallback. If a server does not answer within the per-candidate timeout (a hang) or dies or reports a
poisoned CUDA context (a crash), the adapter kills it and scores the rest of that GPU's share of the
batch the way ``compileiq_adapter`` does: one ``_score`` process, with its own CUDA context, per
candidate. A candidate that crashed the server is re-scored that way, since the crash may have been
caused by an earlier candidate; a candidate that hung is recorded INVALID rather than paying the
timeout twice. The next batch starts a fresh server, unless that GPU has already failed
PTX_ANNEAL_BATCH_MAX_FAILURES times, after which it stays in per-candidate mode.
"""

import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

# The shared helpers live in the standard adapter next to this file. Both are standalone scripts, so
# import it as a sibling module rather than through the ptx_anneal package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compileiq_adapter as base

DEFAULT_RETIME_EVERY = 8
DEFAULT_MAX_FAILURES = 3
DEFAULT_STARTUP_TIMEOUT_S = 600.0
# Compiles normally take well under a second each; this only bounds a pathological ACF.
DEFAULT_COMPILE_TIMEOUT_S = 120.0

_EOF = object()


class _Config:
    """Everything a lane needs to launch scorers, read once from the environment."""

    def __init__(
        self,
        *,
        task,
        ptxas,
        warmup,
        rep,
        score_cmd,
        compile_cmd,
        server_cmd,
        task_timeout,
        retime_every,
        max_failures,
        startup_timeout,
    ):
        self.task = task
        self.ptxas = ptxas
        self.task_timeout = task_timeout
        self.retime_every = retime_every
        self.max_failures = max_failures
        self.startup_timeout = startup_timeout
        self.compile_cmd = compile_cmd
        bench_args = ["--task", task, "--ptxas", ptxas, "--warmup", str(warmup), "--rep", str(rep)]
        self.score_cmd = [*score_cmd, *bench_args]
        self.server_cmd = [*server_cmd, *bench_args]


class _Lane:
    """One GPU: a persistent ``_score_server``, with per-candidate ``_score`` processes as fallback."""

    def __init__(self, gpu, cfg: _Config):
        self.gpu = gpu
        self.cfg = cfg
        self.proc = None
        self.replies = None
        self._next_id = 0
        self.baselines = []  # every untuned timing on this GPU, in order
        self.since_retime = 0
        self.failures = 0
        self.hangs = 0
        self.crashes = 0
        self.starts = 0

    @property
    def baseline_ms(self):
        """Median of this GPU's baseline timings so far (None before the first)."""
        if not self.baselines:
            return None
        ordered = sorted(self.baselines)
        mid = len(ordered) // 2
        return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2

    def _record_baseline(self, ms: float) -> None:
        self.baselines.append(ms)
        self.since_retime = 0
        record = {"kind": "baseline", "gpu": self.label, "ms": ms, "median_ms": self.baseline_ms}
        print(json.dumps(record), file=sys.stderr, flush=True)

    @property
    def label(self) -> str:
        return self.gpu if self.gpu is not None else "default"

    def _env(self):
        return None if self.gpu is None else {**os.environ, "CUDA_VISIBLE_DEVICES": self.gpu}

    def _log(self, msg: str) -> None:
        print(f"[batch gpu={self.label}] {msg}", file=sys.stderr, flush=True)

    # -- persistent server ---------------------------------------------------------------------
    @staticmethod
    def _read(proc, replies) -> None:
        for line in proc.stdout:
            try:
                replies.put(json.loads(line))
            except ValueError:
                print(f"[batch] ignoring non-protocol server output: {line[:200]!r}", file=sys.stderr, flush=True)
        replies.put(_EOF)

    def _get(self, timeout):
        """The next reply, ``_EOF`` if the server exited, or None on timeout."""
        try:
            return self.replies.get(timeout=timeout)
        except queue.Empty:
            return None

    def _start(self) -> bool:
        self.proc = subprocess.Popen(
            self.cfg.server_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self._env(),
        )
        self.replies = queue.Queue()
        threading.Thread(target=self._read, args=(self.proc, self.replies), daemon=True).start()
        msg = self._get(self.cfg.startup_timeout)
        if msg is None or msg is _EOF or not msg.get("ready"):
            why = "timed out" if msg is None else "exited" if msg is _EOF else msg.get("error")
            self._log(f"benchmark server failed to start ({why})")
            self._fail()
            return False
        self.starts += 1
        self._record_baseline(float(msg["baseline_ms"]))
        return True

    def _request(self, op: str, timeout, **fields):
        """Send one request. Returns (status, reply): status is ok, error, hang or crash."""
        rid = self._next_id
        self._next_id += 1
        try:
            self.proc.stdin.write(json.dumps({"op": op, "id": rid, **fields}) + "\n")
            self.proc.stdin.flush()
        except OSError:
            return "crash", None
        while True:
            msg = self._get(timeout)
            if msg is None:
                return "hang", None
            if msg is _EOF:
                return "crash", None
            if msg.get("id") != rid:
                continue  # a late reply to an earlier request
            if msg.get("ok"):
                return "ok", msg
            return ("crash" if msg.get("fatal") else "error"), msg

    def _kill(self) -> None:
        if self.proc is None:
            return
        self.proc.kill()
        self.proc.wait()
        self.proc = None

    def _fail(self) -> None:
        self._kill()
        self.failures += 1
        if self.failures >= self.cfg.max_failures:
            self._log(f"{self.failures} server failures; per-candidate mode for the rest of the search")

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.stdin.write(json.dumps({"op": "exit"}) + "\n")
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
            self.proc = None
        except (OSError, subprocess.TimeoutExpired):
            self._kill()

    def _retime(self) -> bool:
        status, reply = self._request("baseline", self.cfg.task_timeout)
        if status == "ok":
            self._record_baseline(float(reply["ms"]))
            return True
        self._log(f"baseline re-time failed ({status}); falling back to per-candidate mode")
        self._fail()
        return False

    # -- per-candidate fallback ----------------------------------------------------------------
    def _score_isolated(self, cand: dict) -> dict:
        """Score one candidate in its own ``_score`` process (own CUDA context), as compileiq_adapter does."""
        result = {"raw_ms": None, "mode": "isolated", "timeout": False, "baseline_ms": self.baseline_ms}
        try:
            out = subprocess.run(
                [*self.cfg.score_cmd, "--acf", cand["acf_path"]],
                env=self._env(),
                capture_output=True,
                text=True,
                timeout=self.cfg.task_timeout,
                check=False,
            )
            last = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else ""
            if out.returncode == 0 and last.startswith("MS "):
                value = float(last.split()[1])
                if math.isfinite(value) and value > 0:
                    result["raw_ms"] = value
            if result["raw_ms"] is None:
                result["reason"] = (out.stderr.strip().splitlines() or ["INVALID"])[-1][:300]
        except subprocess.TimeoutExpired:
            result["timeout"] = True
            result["reason"] = "per-candidate timeout"
        except (OSError, ValueError, IndexError) as e:
            result["reason"] = f"{type(e).__name__}: {e}"
        return result

    # -- one batch -----------------------------------------------------------------------------
    def score_share(self, work: "queue.Queue", results: dict) -> None:
        """Pull candidates from the shared queue until it is empty, writing ``results[sha]``."""
        server = self.failures < self.cfg.max_failures
        started = False
        if server and self.proc is None:
            server = started = self._start()  # startup already timed the baseline
        if server and not started:
            server = self._retime()
        while True:
            try:
                cand = work.get_nowait()
            except queue.Empty:
                return
            if server and self.cfg.retime_every and self.since_retime >= self.cfg.retime_every:
                server = self._retime()
            if not server:
                results[cand["sha"]] = self._score_isolated(cand)
                continue
            status, reply = self._request("score", self.cfg.task_timeout, cubin=cand["cubin"])
            self.since_retime += 1
            if status == "ok":
                ms = float(reply["ms"])
                results[cand["sha"]] = {"raw_ms": ms, "mode": "server", "baseline_ms": self.baseline_ms}
            elif status == "error":
                reason = reply.get("reason", "")[:300]
                results[cand["sha"]] = {"raw_ms": None, "mode": "server", "reason": reason}
            else:
                # A hang or a crash: drop the server and finish this GPU's share one context per candidate.
                if status == "hang":
                    self.hangs += 1
                else:
                    self.crashes += 1
                if status == "hang":
                    why = "no reply within the per-candidate timeout"
                else:
                    why = (reply or {}).get("reason", "server exited")
                acf = cand["acf_hex"][:16]
                self._log(f"candidate {acf} {status}: {why}; per-candidate mode for the rest of this batch")
                self._fail()
                server = False
                if status == "crash":
                    results[cand["sha"]] = self._score_isolated(cand)
                else:
                    results[cand["sha"]] = {"raw_ms": None, "mode": "server", "timeout": True, "reason": why}


class _BatchObjective:
    """Scores a whole pool per call; the custom Worker hands it each generation's candidates."""

    def __init__(self, cfg: _Config, lanes: list, compile_timeout):
        self.cfg = cfg
        self.lanes = lanes
        self.compile_timeout = compile_timeout
        self.cache = {}  # cubin sha256 -> engine score (None = invalid)
        self.raw_ms = {}  # cubin sha256 -> unscaled milliseconds
        self.acf_sha = {}  # ACF hex -> cubin sha256
        self.reference_ms = None
        self.counts = {"n": 0, "valid": 0, "timeout": 0, "compile_failed": 0, "dedup": 0, "isolated": 0}

    def __call__(self, *args, **kwargs):
        raise RuntimeError("the batch adapter scores whole pools through its Worker, not per candidate")

    def _compile(self, acf_paths: list, out_dir: str) -> list:
        argv = [*self.cfg.compile_cmd, "--task", self.cfg.task, "--ptxas", self.cfg.ptxas, "--out", out_dir]
        if self.compile_timeout:
            argv += ["--timeout", str(self.compile_timeout)]
        try:
            out = subprocess.run(
                [*argv, *acf_paths],
                capture_output=True,
                text=True,
                timeout=max(600.0, 4 * (self.compile_timeout or 0)),
                check=False,
            )
            rows = [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]
            if out.returncode == 0 and len(rows) == len(acf_paths):
                return rows
            reason = (out.stderr.strip().splitlines() or [f"exit {out.returncode}"])[-1]
        except (OSError, ValueError, subprocess.TimeoutExpired) as e:
            reason = f"{type(e).__name__}: {e}"
        print(f"[batch] compile step failed for the whole pool: {reason}", file=sys.stderr, flush=True)
        return [{"acf": p, "cubin": None, "error": reason} for p in acf_paths]

    def _engine_score(self, result: dict):
        raw, base_ms = result.get("raw_ms"), result.get("baseline_ms")
        if raw is None:
            return None
        if self.reference_ms is None or not base_ms:
            return raw
        return raw * self.reference_ms / base_ms

    def score_batch(self, acfs: list) -> list:
        tmp = tempfile.mkdtemp(prefix="ptx_anneal_batch_")
        try:
            cands = []
            for i, acf in enumerate(acfs):
                path = os.path.join(tmp, f"{i}.acf")
                with open(path, "wb") as f:
                    f.write(bytes.fromhex(str(acf)))
                cands.append({"acf_hex": str(acf), "acf_path": path})
            for cand, row in zip(cands, self._compile([c["acf_path"] for c in cands], tmp), strict=True):
                cand.update(cubin=row.get("cubin"), sha=row.get("sha256"), error=row.get("error"))

            fresh = {}
            for cand in cands:
                if cand["cubin"] and cand["sha"] not in self.cache and cand["sha"] not in fresh:
                    fresh[cand["sha"]] = cand
            work = queue.Queue()
            for cand in fresh.values():
                work.put(cand)
            results = {}
            if fresh:
                with ThreadPoolExecutor(max_workers=len(self.lanes)) as pool:
                    list(pool.map(lambda lane: lane.score_share(work, results), self.lanes))
            if self.reference_ms is None:
                self.reference_ms = next((lane.baselines[0] for lane in self.lanes if lane.baselines), None)
            for sha, result in results.items():
                self.cache[sha] = self._engine_score(result)
                self.raw_ms[sha] = result.get("raw_ms")
            for cand in cands:
                if cand["sha"]:
                    self.acf_sha[cand["acf_hex"]] = cand["sha"]

            scores = []
            for cand in cands:
                if not cand["cubin"]:
                    score, info = None, {"mode": "compile_error", "reason": (cand["error"] or "")[:300]}
                    self.counts["compile_failed"] += 1
                elif cand["sha"] in results and fresh.get(cand["sha"]) is cand:
                    info = results[cand["sha"]]
                    score = self.cache[cand["sha"]]
                    self.counts["isolated"] += int(info.get("mode") == "isolated")
                    self.counts["timeout"] += int(bool(info.get("timeout")))
                else:
                    score, info = self.cache.get(cand["sha"]), {"mode": "dedup"}
                    self.counts["dedup"] += 1
                self.counts["n"] += 1
                self.counts["valid"] += int(score is not None)
                scores.append(score)
                print(
                    json.dumps(
                        {
                            "kind": "candidate",
                            "index": self.counts["n"],
                            "ms": score,
                            "raw_ms": info.get("raw_ms"),
                            "ok": score is not None,
                            "timeout": bool(info.get("timeout")),
                            "mode": info.get("mode"),
                            "acf": cand["acf_hex"][:16],
                            **({"reason": info["reason"]} if info.get("reason") and score is None else {}),
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            return scores
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _make_worker(Worker, Score, invalid_score):
    """A CompileIQ Worker that passes each generation's whole pool to the batch objective."""

    class BatchWorker(Worker):
        @classmethod
        def create(cls, cache_folder, normalize, tracker):
            return cls(cache_folder=cache_folder, normalize=normalize, tracker=tracker)

        def run(self, *, function, params_pool, params_ids, num_function_returns=1, **kwargs):
            pool = list(params_pool)
            try:
                values = function.score_batch(pool)
            except Exception as e:  # noqa: BLE001 - one broken batch must not end the search
                print(f"[batch] scoring failed for this generation: {type(e).__name__}: {e}", file=sys.stderr)
                values = [None] * len(pool)
            return [
                Score(
                    score=invalid_score if value is None else value,
                    param_id=param_id,
                    params=params,
                    num_objectives=num_function_returns,
                )
                for value, param_id, params in zip(values, params_ids, pool, strict=True)
            ]

    return BatchWorker


def _require_compileiq_1_1(version: str) -> None:
    m = re.match(r"(\d+)\.(\d+)", version)
    if not m or (int(m.group(1)), int(m.group(2))) < (1, 1):
        raise SystemExit(
            f"compileiq_batch_adapter: needs CompileIQ >= 1.1 (found {version}); point --engine-python at "
            "an interpreter with CompileIQ 1.1, or drop --batch-scoring"
        )


def main() -> int:
    metric = os.environ.get("PTX_ANNEAL_METRIC", "ms")
    if metric != "ms":
        raise SystemExit(f"compileiq_batch_adapter: only runtime (ms) tuning is batched, not {metric!r}")
    missing = [k for k in ("PTX_ANNEAL_COMPILE_CMD", "PTX_ANNEAL_SERVER_CMD") if not os.environ.get(k)]
    if missing:
        unset = ", ".join(missing)
        raise SystemExit(f"compileiq_batch_adapter: {unset} not set; run it through ptx_anneal --batch-scoring")
    task = os.environ["PTX_ANNEAL_TASK"]
    ptxas = os.environ["PTX_ANNEAL_PTXAS"]
    ss = os.environ.get("PTX_ANNEAL_SS") or ""
    result_path = os.environ["PTX_ANNEAL_RESULT"]
    warmup = os.environ.get("PTX_ANNEAL_WARMUP", "25")
    rep = os.environ.get("PTX_ANNEAL_REP", "50")
    gens = int(os.environ.get("CIQ_GENERATIONS", "1"))
    pool = int(os.environ.get("CIQ_POOL", "8"))
    if pool == 0:
        pool = None
    cull = int(os.environ.get("CIQ_CULL", "4"))
    task_timeout = base._task_timeout()
    clock_mhz = base._env_num("CIQ_CLOCK_MHZ", int)
    retime_every = base._env_num("PTX_ANNEAL_BATCH_RETIME", int)
    max_failures = base._env_num("PTX_ANNEAL_BATCH_MAX_FAILURES", int)
    startup_timeout = base._env_num("PTX_ANNEAL_BATCH_STARTUP_TIMEOUT", float)
    try:
        num_workers = base._env_num("PTX_ANNEAL_NUM_WORKERS", int)
        num_workers = 1 if num_workers is None else num_workers
        gpus = base._worker_gpus(num_workers)
    except ValueError as e:
        raise SystemExit(f"compileiq_batch_adapter: {e}") from e

    engine_info = base._engine_info()
    print(
        f"engine: {engine_info['name']} {engine_info['version']} from {engine_info['path']} "
        f"(python={engine_info['python']})",
        file=sys.stderr,
        flush=True,
    )
    _require_compileiq_1_1(engine_info["version"])

    from compileiq.ciq import Search
    from compileiq.types import INVALID_SCORE, SearchConfiguration, Worker
    from compileiq.utils.validation import Score

    search_space, ss_meta = base._resolve_search_space(ss, base._ptxas_version(ptxas))
    tag = ss_meta.get("resolved_tag")
    where = f"{ss_meta.get('source', '?')}{f' {tag}' if tag else ''}"
    print(
        f"search space: {ss_meta.get('filename') or ss_meta.get('path')} [{where}]",
        file=sys.stderr,
        flush=True,
    )

    cfg = _Config(
        task=task,
        ptxas=ptxas,
        warmup=warmup,
        rep=rep,
        score_cmd=json.loads(os.environ["PTX_ANNEAL_SCORE_CMD"]),
        compile_cmd=json.loads(os.environ["PTX_ANNEAL_COMPILE_CMD"]),
        server_cmd=json.loads(os.environ["PTX_ANNEAL_SERVER_CMD"]),
        task_timeout=task_timeout,
        retime_every=DEFAULT_RETIME_EVERY if retime_every is None else max(retime_every, 0),
        max_failures=DEFAULT_MAX_FAILURES if max_failures is None else max(max_failures, 1),
        startup_timeout=startup_timeout or DEFAULT_STARTUP_TIMEOUT_S,
    )
    lanes = [_Lane(gpu, cfg) for gpu in (gpus or [None])]
    compile_timeout = task_timeout or DEFAULT_COMPILE_TIMEOUT_S
    objective = _BatchObjective(cfg, lanes, compile_timeout)
    print(f"scoring: batch; gpus={gpus or 'default CUDA device'}", file=sys.stderr, flush=True)

    search = Search(
        objective_function=objective,
        search_space=search_space,
        search_config=SearchConfiguration(problem_type="min", generations=gens, pool_size=pool, cull_size=cull),
        exit_on_failure=False,
        worker_type=_make_worker(Worker, Score, INVALID_SCORE),
    )
    engine_error = None
    result = None
    try:
        with base._benchmark_mode(clock_mhz):
            result = search.start()
    except RuntimeError as e:
        engine_error = f"{type(e).__name__}: {e}"
        print(f"engine error (results below are partial): {engine_error}", file=sys.stderr, flush=True)
    finally:
        for lane in lanes:
            lane.close()

    counts = objective.counts
    df = result.get_results() if result is not None else None
    best = result.get_best_result() if result is not None else None
    best_score = best.get("score_1", best.get("score")) if best else None
    params = best.get("params") if best else None
    acf_hex = params if (params and isinstance(best_score, (int, float))) else None
    payload = {
        "best_ms": float(best_score) if acf_hex else None,
        # The winner's unscaled time; best_ms is rescaled to the reference baseline.
        "best_raw_ms": objective.raw_ms.get(objective.acf_sha.get(acf_hex)) if acf_hex else None,
        "evaluated": len(df) if df is not None else counts["n"],
        "valid": counts["valid"],
        "timeout": counts["timeout"],
        "acf": acf_hex,
        "engine_error": engine_error,
        "search_space": ss_meta,
        "engine_info": engine_info,
        "batch": {
            "reference_baseline_ms": objective.reference_ms,
            "compile_failed": counts["compile_failed"],
            "dedup": counts["dedup"],
            "isolated": counts["isolated"],
            "server_starts": sum(lane.starts for lane in lanes),
            "server_failures": sum(lane.failures for lane in lanes),
            "hangs": sum(lane.hangs for lane in lanes),
            "crashes": sum(lane.crashes for lane in lanes),
        },
        "budget": {
            "scorer": "batch",
            "num_workers": num_workers,
            "gpus": gpus,
            "generations": gens,
            "pool_size": pool,
            "cull_size": cull,
            "task_timeout": task_timeout,
            "retime_every": cfg.retime_every,
            "max_failures": cfg.max_failures,
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
