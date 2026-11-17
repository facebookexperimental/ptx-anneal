# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Harness side of --batch-scoring: the pooled ``_compile``, the ``_score_server`` protocol, and the
CLI wiring that hands both to the batch adapter. Only the last test needs a GPU (skipped in CI)."""

import hashlib
import io
import json
import os
import stat
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ptx_anneal import batch_score, cli, provision

SAMPLE_TASK = str(Path(__file__).resolve().parents[1] / "sample_tasks" / "sample_task")

# A stand-in ptxas: the cubin is "cubin:" + the ACF bytes; an ACF saying SLEEP or FAIL misbehaves.
FAKE_PTXAS = """#!/usr/bin/env python3
import sys, time
args = sys.argv[1:]
acf = next((a.split("=", 1)[1] for a in args if a.startswith("--apply-controls=")), None)
body = open(acf).read() if acf else "baseline"
if body == "SLEEP":
    time.sleep(30)
if body == "FAIL":
    sys.stderr.write("ptxas fatal: bad control\\n")
    sys.exit(255)
assert any(a.startswith("-arch=") for a in args) and "--regAllocOptLevel=2" in args
open(args[args.index("-o") + 1], "w").write("cubin:" + body)
"""


@pytest.fixture
def fake_ptxas(tmp_path):
    path = tmp_path / "ptxas"
    path.write_text(FAKE_PTXAS)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def _acfs(tmp_path, *bodies):
    paths = []
    for i, body in enumerate(bodies):
        path = tmp_path / f"{i}.acf"
        path.write_text(body)
        paths.append(str(path))
    return paths


def test_compile_batch_keeps_order_hashes_cubins_and_reports_failures(tmp_path, fake_ptxas):
    acfs = _acfs(tmp_path, "a", "FAIL", "a", "SLEEP")
    rows = batch_score.compile_batch(SAMPLE_TASK, fake_ptxas, acfs, str(tmp_path / "out"), jobs=4, timeout=2)
    assert [row["acf"] for row in rows] == acfs
    assert Path(rows[0]["cubin"]).read_text() == "cubin:a"
    assert rows[0]["sha256"] == rows[2]["sha256"] == hashlib.sha256(b"cubin:a").hexdigest()
    assert rows[0]["cubin"] != rows[2]["cubin"]
    assert rows[1]["cubin"] is None and "bad control" in rows[1]["error"]
    assert rows[3]["cubin"] is None and "timed out" in rows[3]["error"]


def test_compile_subcommand_prints_one_json_line_per_acf(tmp_path, fake_ptxas, capsys):
    acfs = _acfs(tmp_path, "x", "y")
    argv = ["_compile", "--task", SAMPLE_TASK, "--ptxas", fake_ptxas, "--out", str(tmp_path / "out"), *acfs]
    assert cli.main(argv) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [Path(row["cubin"]).read_text() for row in rows] == ["cubin:x", "cubin:y"]


def test_batch_scoring_selects_the_bundled_batch_adapter(monkeypatch):
    monkeypatch.delenv("PTX_ANNEAL_ENGINE_ADAPTER", raising=False)
    args = cli.build_parser().parse_args(["--task", "/t", "--batch-scoring", "--engine-python", "/py"])
    python, adapter = cli._engine_command(args)
    assert python == "/py" and adapter == cli._bundled_adapter("compileiq_batch_adapter.py")
    assert os.path.exists(adapter)

    args = cli.build_parser().parse_args(["--task", "/t", "--batch-scoring", "--engine-adapter", "/a.py"])
    with pytest.raises(SystemExit, match="drop --engine-adapter"):
        cli._engine_command(args)


def test_run_engine_hands_batch_commands_to_adapters(monkeypatch, tmp_path):
    seen = {}

    def run(cmd, **kwargs):
        seen["env"] = kwargs["env"]
        Path(kwargs["env"]["PTX_ANNEAL_RESULT"]).write_text("{}")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    args = SimpleNamespace(bench="do_bench", ss=None, task=str(tmp_path), warmup=5, rep=10)
    cli._run_engine(args, "/ptxas", "/py", "/adapter.py")
    harness = cli._harness_cmd()
    assert json.loads(seen["env"]["PTX_ANNEAL_COMPILE_CMD"]) == [*harness, "_compile"]
    assert json.loads(seen["env"]["PTX_ANNEAL_SERVER_CMD"]) == [*harness, "_score_server", "--bench", "do_bench"]


def test_score_server_reports_setup_failure_as_not_ready(tmp_path):
    out = io.StringIO()
    code = batch_score.serve(
        str(tmp_path / "missing"), "/ptxas", warmup=1, rep=1, bench="cudagraph", stdin=io.StringIO(""), stdout=out
    )
    assert code == 1
    reply = json.loads(out.getvalue())
    assert reply["ready"] is False and reply["error"]


def _gpu_ptxas():
    torch = pytest.importorskip("torch")
    pytest.importorskip("cuda.bindings")
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU")
    ptxas = os.environ.get("PTXAS") or provision.find_ptxas()
    if not ptxas:
        pytest.skip("needs a ptxas >= 13.3")
    return ptxas


def test_score_server_end_to_end_on_gpu(tmp_path):
    ptxas = _gpu_ptxas()
    from ptx_anneal import ptx_launch, task

    t = task.load(SAMPLE_TASK)
    good = tmp_path / "good.cubin"
    good.write_bytes(ptx_launch.ptxas_compile(t.ir, ptxas, arch=t.spec["arch"]))
    garbage = tmp_path / "garbage.cubin"
    garbage.write_bytes(b"not a cubin")
    requests = "".join(
        json.dumps(r) + "\n"
        for r in (
            {"op": "score", "id": 1, "cubin": str(good)},
            {"op": "baseline", "id": 2},
            {"op": "score", "id": 3, "cubin": str(garbage)},
            {"op": "score", "id": 4, "cubin": str(good)},
            {"op": "exit"},
        )
    )
    out = io.StringIO()
    code = batch_score.serve(
        SAMPLE_TASK, ptxas, warmup=2, rep=10, bench="cudagraph", stdin=io.StringIO(requests), stdout=out
    )
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert code == 0
    assert replies[0]["ready"] and replies[0]["baseline_ms"] > 0
    assert replies[1] == {"id": 1, "ok": True, "ms": replies[1]["ms"]} and replies[1]["ms"] > 0
    assert replies[2]["id"] == 2 and replies[2]["ok"]
    # A cubin that does not load is an ordinary invalid candidate; the context survives it.
    assert replies[3]["id"] == 3 and not replies[3]["ok"] and replies[3]["fatal"] is False
    assert replies[4]["id"] == 4 and replies[4]["ok"]
