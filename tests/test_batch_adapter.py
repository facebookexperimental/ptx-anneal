# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Batch CompileIQ adapter: pooled compile, dedup, per-GPU servers and the per-candidate fallback.

Everything here runs without a GPU, torch or the engine. The harness commands are small fake
scripts: an ACF's bytes stand for what the candidate does (``OK <ms>``, ``BAD``, ``HANG``, ``CRASH``,
``FATAL``, ``NOCOMPILE``), and the fake ``_compile`` copies each ACF into its cubin, so identical ACFs
compile to identical cubins.
"""

import importlib.util
import sys
import textwrap
import types

import pytest

ADAPTER = "ptx_anneal.adapters.compileiq_batch_adapter"


def _load_adapter():
    sys.modules.pop(ADAPTER, None)
    spec = importlib.util.find_spec(ADAPTER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FAKE_COMPILE = """
import hashlib, json, os, shutil, sys
args = sys.argv[1:]
out = args[args.index("--out") + 1]
acfs = [a for i, a in enumerate(args) if i > 0 and not a.startswith("--") and not args[i - 1].startswith("--")]
for i, acf in enumerate(acfs):
    data = open(acf, "rb").read()
    if data.startswith(b"NOCOMPILE"):
        print(json.dumps({"acf": acf, "cubin": None, "error": "ptxas failed"}))
        continue
    path = os.path.join(out, f"{i}.cubin")
    shutil.copy(acf, path)
    print(json.dumps({"acf": acf, "cubin": path, "sha256": hashlib.sha256(data).hexdigest()}))
"""

# Baselines come from FAKE_BASELINES ("2.0,4.0" = 2.0 at startup, 4.0 for later re-times). Every
# request is appended to FAKE_LOG with the GPU it ran on.
FAKE_SERVER = """
import json, os, sys, time
log = open(os.environ["FAKE_LOG"], "a")
gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "default")
baselines = [float(b) for b in os.environ.get("FAKE_BASELINES", "2.0").split(",")]
if os.environ.get("FAKE_SERVER_BROKEN"):
    print(json.dumps({"ready": False, "error": "no inputs"}), flush=True)
    sys.exit(1)
print(json.dumps({"ready": True, "baseline_ms": baselines[0], "baseline_sha256": "b"}), flush=True)
retimes = 0
for line in sys.stdin:
    req = json.loads(line)
    if req["op"] == "exit":
        break
    if req["op"] == "baseline":
        retimes += 1
        log.write(f"baseline {gpu}\\n"); log.flush()
        print(json.dumps({"id": req["id"], "ok": True, "ms": baselines[min(retimes, len(baselines) - 1)]}), flush=True)
        continue
    body = open(req["cubin"]).read()
    log.write(f"score {gpu} {body}\\n"); log.flush()
    time.sleep(float(os.environ.get("FAKE_SERVER_DELAY", "0")))
    if body.startswith("OK"):
        print(json.dumps({"id": req["id"], "ok": True, "ms": float(body.split()[1])}), flush=True)
    elif body == "BAD":
        print(json.dumps({"id": req["id"], "ok": False, "reason": "validation: diverged", "fatal": False}), flush=True)
    elif body == "HANG":
        time.sleep(60)
    elif body == "CRASH":
        sys.exit(3)
    elif body == "FATAL":
        print(json.dumps({"id": req["id"], "ok": False, "reason": "illegal address", "fatal": True}), flush=True)
        sys.exit(3)
"""

# The per-candidate scorer, with its own "CUDA context": a candidate that crashed the server is fine
# here (2.5 ms), modelling a crash caused by an earlier candidate's corruption.
FAKE_SCORE = """
import os, sys, time
args = sys.argv[1:]
body = open(args[args.index("--acf") + 1]).read()
open(os.environ["FAKE_LOG"], "a").write(f"isolated {os.environ.get('CUDA_VISIBLE_DEVICES', 'default')} {body}\\n")
if body.startswith("OK"):
    print("MS", body.split()[1])
elif body in ("CRASH", "FATAL"):
    print("MS 2.5")
elif body == "HANG":
    time.sleep(60)
else:
    print("INVALID")
"""


@pytest.fixture
def harness(tmp_path, monkeypatch):
    scripts = {}
    for name, body in (("compile", FAKE_COMPILE), ("server", FAKE_SERVER), ("score", FAKE_SCORE)):
        path = tmp_path / f"fake_{name}.py"
        path.write_text(textwrap.dedent(body))
        scripts[name] = [sys.executable, str(path)]
    log = tmp_path / "requests.log"
    log.write_text("")
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    return types.SimpleNamespace(scripts=scripts, log=log)


def _objective(mod, harness, *, gpus=(None,), task_timeout=10.0, retime_every=0, max_failures=3):
    cfg = mod._Config(
        task="/task",
        ptxas="/ptxas",
        warmup=1,
        rep=2,
        score_cmd=harness.scripts["score"],
        compile_cmd=harness.scripts["compile"],
        server_cmd=harness.scripts["server"],
        task_timeout=task_timeout,
        retime_every=retime_every,
        max_failures=max_failures,
        startup_timeout=30.0,
    )
    lanes = [mod._Lane(gpu, cfg) for gpu in gpus]
    return mod._BatchObjective(cfg, lanes, compile_timeout=30.0), lanes


def _hex(*bodies):
    return [body.encode().hex() for body in bodies]


def _log(harness):
    return harness.log.read_text().splitlines()


def test_pool_is_deduplicated_and_compile_failures_are_invalid(harness):
    mod = _load_adapter()
    obj, lanes = _objective(mod, harness)
    try:
        scores = obj.score_batch(_hex("OK 1.0", "OK 1.0", "OK 3.0", "NOCOMPILE", "BAD"))
        assert scores == [1.0, 1.0, 3.0, None, None]
        assert [line for line in _log(harness) if line.startswith("score")] == [
            "score default OK 1.0",
            "score default OK 3.0",
            "score default BAD",
        ]
        assert obj.counts["dedup"] == 1 and obj.counts["compile_failed"] == 1

        # Identical machine code in a later generation reuses the measurement.
        assert obj.score_batch(_hex("OK 3.0")) == [3.0]
        assert sum(line.startswith("score") for line in _log(harness)) == 3
        assert obj.counts["dedup"] == 2
    finally:
        for lane in lanes:
            lane.close()


def test_scores_are_normalized_to_the_median_same_gpu_baseline(harness, monkeypatch):
    # Baselines: 2.0 ms at startup, then 4.0 ms at each batch's re-time. A 2.0 ms candidate is scaled by
    # reference (2.0) / median of the GPU's baselines so far: 2/3 after one re-time, 2/4 after two.
    monkeypatch.setenv("FAKE_BASELINES", "2.0,4.0,4.0")
    mod = _load_adapter()
    obj, lanes = _objective(mod, harness)
    try:
        assert obj.score_batch(_hex("OK 1.0")) == [1.0]
        assert obj.reference_ms == 2.0
        assert obj.score_batch(_hex("OK 2.0")) == [pytest.approx(2.0 * 2.0 / 3.0)]
        assert obj.score_batch(_hex("OK 2.5")) == [pytest.approx(2.5 * 2.0 / 4.0)]
        assert lanes[0].baselines == [2.0, 4.0, 4.0]
        # The unscaled time stays available for the winner.
        assert obj.raw_ms[obj.acf_sha[b"OK 2.0".hex()]] == 2.0
    finally:
        for lane in lanes:
            lane.close()


def test_baseline_is_retimed_every_n_candidates(harness):
    mod = _load_adapter()
    obj, lanes = _objective(mod, harness, retime_every=2)
    try:
        obj.score_batch(_hex("OK 1.0", "OK 1.1", "OK 1.2", "OK 1.3", "OK 1.4"))
        kinds = [line.split()[0] for line in _log(harness)]
        assert kinds == ["score", "score", "baseline", "score", "score", "baseline", "score"]
    finally:
        for lane in lanes:
            lane.close()


@pytest.mark.parametrize("failure", ["CRASH", "FATAL"])
def test_crash_rescores_the_candidate_alone_and_finishes_the_batch_isolated(harness, failure):
    mod = _load_adapter()
    obj, lanes = _objective(mod, harness)
    try:
        assert obj.score_batch(_hex("OK 1.0", failure, "OK 2.0")) == [1.0, 2.5, 2.0]
        assert _log(harness) == [
            "score default OK 1.0",
            f"score default {failure}",
            f"isolated default {failure}",
            "isolated default OK 2.0",
        ]
        assert lanes[0].crashes == 1 and lanes[0].failures == 1 and obj.counts["isolated"] == 2

        # The next generation gets a fresh server.
        assert obj.score_batch(_hex("OK 4.0")) == [4.0]
        assert _log(harness)[-1] == "score default OK 4.0" and lanes[0].starts == 2
    finally:
        for lane in lanes:
            lane.close()


def test_hang_is_invalid_without_paying_the_timeout_twice(harness):
    mod = _load_adapter()
    obj, lanes = _objective(mod, harness, task_timeout=1.0)
    try:
        assert obj.score_batch(_hex("HANG", "OK 1.5")) == [None, 1.5]
        assert _log(harness) == ["score default HANG", "isolated default OK 1.5"]
        assert lanes[0].hangs == 1 and obj.counts["timeout"] == 1
    finally:
        for lane in lanes:
            lane.close()


def test_gpu_stays_isolated_after_max_failures(harness):
    mod = _load_adapter()
    obj, lanes = _objective(mod, harness, max_failures=1)
    try:
        obj.score_batch(_hex("CRASH"))
        assert obj.score_batch(_hex("OK 1.0", "OK 2.0")) == [1.0, 2.0]
        assert _log(harness)[-2:] == ["isolated default OK 1.0", "isolated default OK 2.0"]
        assert lanes[0].starts == 1
    finally:
        for lane in lanes:
            lane.close()


def test_failed_server_startup_falls_back_to_isolated(harness, monkeypatch):
    monkeypatch.setenv("FAKE_SERVER_BROKEN", "1")
    mod = _load_adapter()
    obj, lanes = _objective(mod, harness)
    try:
        assert obj.score_batch(_hex("OK 1.0", "BAD")) == [1.0, None]
        assert _log(harness) == ["isolated default OK 1.0", "isolated default BAD"]
        assert lanes[0].starts == 0 and lanes[0].failures == 1
    finally:
        for lane in lanes:
            lane.close()


def test_gpus_share_one_pool_and_score_each_cubin_once(harness, monkeypatch):
    monkeypatch.setenv("FAKE_SERVER_DELAY", "0.05")
    mod = _load_adapter()
    obj, lanes = _objective(mod, harness, gpus=("0", "1"))
    try:
        bodies = [f"OK {1 + i / 10}" for i in range(12)]
        assert obj.score_batch(_hex(*bodies)) == [float(b.split()[1]) for b in bodies]
        scored = [line.split(maxsplit=2) for line in _log(harness) if line.startswith("score")]
        assert sorted(body for _, _, body in scored) == sorted(bodies)
        assert {gpu for _, gpu, _ in scored} == {"0", "1"}
    finally:
        for lane in lanes:
            lane.close()


def test_worker_maps_scores_to_candidate_ids():
    mod = _load_adapter()

    class Worker:
        def __init__(self, cache_folder, normalize, tracker):
            pass

    class Score:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    worker = mod._make_worker(Worker, Score, invalid_score=-1).create("/cache", False, None)
    batch = types.SimpleNamespace(score_batch=lambda pool: [1.0 if p == "aa" else None for p in pool])
    scores = worker.run(function=batch, params_pool=["aa", "bb"], params_ids=[7, 8])
    assert [(s.param_id, s.score, s.params) for s in scores] == [(7, 1.0, "aa"), (8, -1, "bb")]

    def broken(pool):
        raise RuntimeError("boom")

    scores = worker.run(function=types.SimpleNamespace(score_batch=broken), params_pool=["aa"], params_ids=[9])
    assert [(s.param_id, s.score) for s in scores] == [(9, -1)]


@pytest.mark.parametrize("version", ["1.0.2", "0.9", "unknown"])
def test_requires_compileiq_1_1(version):
    with pytest.raises(SystemExit, match="CompileIQ >= 1.1"):
        _load_adapter()._require_compileiq_1_1(version)


def test_accepts_compileiq_1_1_prereleases():
    _load_adapter()._require_compileiq_1_1("1.1.0b2")


def test_batch_adapter_never_imports_ptx_anneal():
    """The engine-free boundary holds for the batch adapter too."""
    with open(importlib.util.find_spec(ADAPTER).origin) as f:
        body = f.read()
    assert "import ptx_anneal" not in body and "from ptx_anneal" not in body


def test_main_rejects_spill_metric_and_missing_harness_commands(monkeypatch):
    mod = _load_adapter()
    monkeypatch.setenv("PTX_ANNEAL_METRIC", "spill_bytes")
    with pytest.raises(SystemExit, match="only runtime"):
        mod.main()
    monkeypatch.setenv("PTX_ANNEAL_METRIC", "ms")
    monkeypatch.delenv("PTX_ANNEAL_COMPILE_CMD", raising=False)
    monkeypatch.delenv("PTX_ANNEAL_SERVER_CMD", raising=False)
    with pytest.raises(SystemExit, match="--batch-scoring"):
        mod.main()
