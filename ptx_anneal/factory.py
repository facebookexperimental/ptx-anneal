# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The factory orchestrator: search -> validate -> admit -> store.

``tune`` ties the pieces together for one task:

1. Measure the untuned baseline (``runner.score(target, task, None)``).
2. Run the search backend, giving it an ``objective`` that materializes each candidate's bytes to a
   temp file and scores it through the runner (which isolates + validates it).
3. Admit the best candidate **only if it is a finite time that beats the baseline**, writing the
   artifact + provenance to the store. Otherwise admit nothing ("no candidate admitted").

A search-time win is noisy; the trustworthy decision is the A/B the consumer runs against the plain
baseline. The factory's admit gate is just "don't store a candidate that didn't beat baseline here".
"""

from __future__ import annotations

import math
import os
import tempfile

from .log.base import Logger, NullLogger
from .run.base import INVALID


def tune(
    task,
    *,
    target,
    store,
    runner,
    search,
    config: dict | None = None,
    timeout: float | None = None,
    warmup: int = 100,
    rep: int = 200,
    logger: Logger | None = None,
) -> dict:
    """Tune one task and (maybe) admit a winning artifact to the store. Returns a result summary.

    ``logger`` receives orchestration-level signal (baseline/search timings, the admit decision,
    search win); it defaults to :class:`~ptx_anneal.log.base.NullLogger`. Per-candidate scoring runs
    inside the runner's isolation boundary (a subprocess), so it is not logged from here.
    """
    config = config or {}
    logger = logger or NullLogger()
    arch = target.arch(task.ir)
    ir_hash = target.ir_hash(task.ir)
    if task.spec.get("arch") not in (None, arch):
        raise ValueError(f"spec arch {task.spec.get('arch')!r} != PTX arch {arch!r}")
    if task.ir_hash != ir_hash:
        raise ValueError(f"spec ir_hash {task.ir_hash!r} != computed {ir_hash!r}")

    # 1) baseline.
    with logger.timing("baseline"):
        baseline_ms = runner.score(target, task, None, timeout=timeout, warmup=warmup, rep=rep)
    baseline_ok = math.isfinite(baseline_ms) and baseline_ms != INVALID
    logger.event(
        "baseline",
        target=target.name,
        arch=arch,
        ir_hash=ir_hash,
        baseline_ms=baseline_ms if baseline_ok else None,
        ok=baseline_ok,
    )

    # 2) search. objective: candidate bytes -> ms (or INVALID), scored through the isolating runner.
    # Each call scores exactly one candidate; emit a per-candidate event so the run log shows the
    # result of every candidate (ok=False == failed validation or run error inside the runner).
    scored = {"i": 0}

    def objective(artifact_bytes: bytes) -> float:
        fd, path = tempfile.mkstemp(suffix=".acf")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(artifact_bytes)
            ms = runner.score(target, task, path, timeout=timeout, warmup=warmup, rep=rep)
            scored["i"] += 1
            ok = math.isfinite(ms) and ms != INVALID
            logger.event(
                "candidate",
                index=scored["i"],
                ms=ms if ok else None,
                ok=ok,
                acf=artifact_bytes.hex()[:16],
                bytes=len(artifact_bytes),
            )
            return ms
        finally:
            os.unlink(path)

    with logger.timing("search"):
        result = search.search(task, objective, config=config)
    logger.metric("evaluated", result.evaluated, "count")

    # 3) admit iff a finite candidate beat the baseline.
    admitted = False
    store_path = None
    best_ms = result.score
    win = None
    if result.artifact is not None and math.isfinite(best_ms) and baseline_ok and best_ms < baseline_ms:
        win = (baseline_ms - best_ms) / baseline_ms
        meta = {
            "target": target.name,
            "arch": arch,
            "ir_hash": ir_hash,
            "entry": task.spec.get("entry"),
            "baseline_ms": baseline_ms,
            "best_ms": best_ms,
            "search_win": win,
            "evaluated": result.evaluated,
            **result.meta,
        }
        store_path = store.write(target.name, arch, ir_hash, result.artifact, meta)
        admitted = True
        logger.metric("search_win", win, "ratio")
        logger.event(
            "admitted",
            target=target.name,
            arch=arch,
            ir_hash=ir_hash,
            baseline_ms=baseline_ms,
            best_ms=best_ms,
            win=win,
            engine=result.meta.get("engine"),
            store_path=store_path,
        )
    else:
        reason = "no baseline" if not baseline_ok else "no candidate beat the baseline"
        logger.event(
            "no_candidate",
            target=target.name,
            arch=arch,
            ir_hash=ir_hash,
            reason=reason,
            engine=result.meta.get("engine"),
            evaluated=result.evaluated,
        )

    return {
        "admitted": admitted,
        "target": target.name,
        "arch": arch,
        "ir_hash": ir_hash,
        "baseline_ms": None if not baseline_ok else baseline_ms,
        "best_ms": best_ms if math.isfinite(best_ms) else None,
        "search_win": win,
        "evaluated": result.evaluated,
        "engine": result.meta.get("engine"),
        "store_path": store_path,
    }
