# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Manifold publication through the tuning CLI, without network, engine or GPU dependencies."""

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from ptx_anneal import cli, task

SAMPLE_TASK = Path(__file__).resolve().parents[1] / "sample_tasks" / "sample_task"
REMOTE = "manifold://bucket/tree/task"
DESTINATION = "bucket/tree/task/compileiq_20261014123456"


@pytest.fixture
def tuning(monkeypatch, tmp_path):
    state = SimpleNamespace(
        calls=[],
        uploads={},
        fail=None,
        timeout=False,
        result={
            "acf": "00abff",
            "best_ms": 0.5,
            "evaluated": 8,
            "valid": 8,
            "engine_info": {"name": "compileiq", "version": "test", "path": "/engine", "python": "/python"},
            "search_space": {"resolved_tag": "test-catalog", "sha256": "abcd", "variant": "default", "source": "test"},
        },
        store=tmp_path / "store",
    )

    def now(tz):
        assert tz is timezone.utc
        return datetime(2026, 10, 14, 12, 34, 56, tzinfo=tz)

    monkeypatch.setattr(task, "datetime", SimpleNamespace(now=now))
    monkeypatch.setattr(task.shutil, "which", lambda name: "/tools/manifold" if name == "manifold" else None)
    monkeypatch.setattr(cli, "_require_ptxas", lambda args: ("/ptxas", "13.3"))
    monkeypatch.setattr(cli, "_score_one", lambda *args: 1.0)
    monkeypatch.setattr(cli, "_fingerprint", lambda *args: {"digest": "test-fingerprint"})
    monkeypatch.delenv("PTX_ANNEAL_FORCE_ADMIT", raising=False)
    monkeypatch.delenv("PTX_ANNEAL_ENGINE_ADAPTER", raising=False)

    def run(argv, **kwargs):
        state.calls.append(argv)
        if argv[0] != "/tools/manifold":
            # The engine adapter's subprocess contract: consume a local task and emit result JSON.
            env = kwargs["env"]
            assert task.load(env["PTX_ANNEAL_TASK"]).entry == "add_kernel"
            Path(env["PTX_ANNEAL_RESULT"]).write_text(json.dumps(state.result))
        else:
            operation = argv[1]
            if state.fail == operation or (state.fail == "sidecar" and argv[-1].endswith(".acf.json")):
                if state.timeout:
                    raise subprocess.TimeoutExpired(argv, 120)
                return SimpleNamespace(returncode=1, stderr="permission denied or destination exists", stdout="")
            if operation == "get":
                shutil.copyfile(SAMPLE_TASK / Path(argv[2]).name, argv[3])
            elif operation == "mkdir":
                assert argv == ["/tools/manifold", "mkdir", DESTINATION]
            elif operation == "put":
                assert kwargs["check"] is False
                state.uploads[argv[3]] = Path(argv[2]).read_bytes()
            else:
                pytest.fail(f"unexpected Manifold command: {argv}")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(task.subprocess, "run", run)
    return state


@pytest.mark.parametrize("source", [REMOTE, REMOTE + "/"])
def test_tuning_uploads_acf_and_provenance_to_original_task(tuning, source, capsys):
    assert cli.main(["--task", source, "--output", str(tuning.store), "--upload-manifold"]) == 0
    acf = next(tuning.store.rglob("*.acf"))
    remote_acf = f"{DESTINATION}/{acf.name}"
    assert tuning.uploads[remote_acf] == acf.read_bytes() == bytes.fromhex("00abff")
    assert tuning.uploads[remote_acf + ".json"] == Path(str(acf) + ".json").read_bytes()
    meta = json.loads(tuning.uploads[remote_acf + ".json"])
    assert meta["ptxas_version"] == "13.3"
    assert meta["engine_info"] == tuning.result["engine_info"]
    assert meta["search_space"] == tuning.result["search_space"]
    assert len(tuning.uploads) == 2
    assert f"uploaded ACF and provenance -> manifold://{remote_acf}" in capsys.readouterr().out


@pytest.mark.parametrize("outcome", ["slower", "invalid", "no_baseline"])
def test_no_admission_does_not_publish(tuning, monkeypatch, capsys, outcome):
    if outcome == "slower":
        tuning.result["best_ms"] = 2.0
    elif outcome == "invalid":
        tuning.result["acf"] = None
    else:
        monkeypatch.setattr(cli, "_score_one", lambda *args: float("inf"))
    assert cli.main(["--task", REMOTE, "--output", str(tuning.store), "--upload-manifold"]) == 0
    assert not tuning.uploads
    assert not any(argv[1] == "mkdir" for argv in tuning.calls)
    assert "Manifold upload skipped: no ACF admitted." in capsys.readouterr().out


def test_force_admitted_acf_can_be_uploaded(tuning, monkeypatch):
    tuning.result["best_ms"] = 2.0
    monkeypatch.setenv("PTX_ANNEAL_FORCE_ADMIT", "1")
    assert cli.main(["--task", REMOTE, "--output", str(tuning.store), "--upload-manifold"]) == 0
    sidecar = next(data for path, data in tuning.uploads.items() if path.endswith(".json"))
    assert json.loads(sidecar)["forced"] is True


def test_upload_is_opt_in(tuning):
    assert cli.main(["--task", REMOTE, "--output", str(tuning.store)]) == 0
    assert list(tuning.store.rglob("*.acf"))
    assert not tuning.uploads
    assert not any(argv[1] == "mkdir" for argv in tuning.calls)


@pytest.mark.parametrize("failure", ["mkdir", "put", "sidecar", "timeout"])
def test_upload_failure_is_nonzero_and_keeps_local_result(tuning, capsys, failure):
    tuning.fail = "put" if failure == "timeout" else failure
    tuning.timeout = failure == "timeout"
    assert cli.main(["--task", REMOTE, "--output", str(tuning.store), "--upload-manifold"]) == 1
    acf = next(tuning.store.rglob("*.acf"))
    assert acf.read_bytes() == bytes.fromhex("00abff")
    assert Path(str(acf) + ".json").is_file()
    captured = capsys.readouterr()
    assert f"Local ACF retained at {acf}" in captured.err
    assert "uploaded ACF and provenance" not in captured.out
    assert "test-fingerprint" in captured.out
    if failure == "mkdir":
        assert not any(argv[1] == "put" for argv in tuning.calls)


@pytest.mark.parametrize("source", ["/local/task", "https://bucket/tree/task"])
def test_upload_rejects_non_manifold_tasks_before_io(tuning, source, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--task", source, "--upload-manifold"])
    assert error.value.code == 2
    assert "--upload-manifold requires --task manifold://" in capsys.readouterr().err
    assert tuning.calls == []


@pytest.mark.parametrize("command", ["replay", "doctor", "_score"])
def test_non_tuning_commands_reject_upload_option(tuning, command):
    args = [] if command == "doctor" else ["--task", REMOTE]
    with pytest.raises(SystemExit) as error:
        cli.main([command, *args, "--upload-manifold"])
    assert error.value.code == 2
    assert tuning.calls == []


def test_missing_provenance_fails_before_remote_writes(tuning, tmp_path):
    acf = tmp_path / "test.acf"
    acf.write_bytes(b"acf")
    with pytest.raises(task.TaskError, match="missing local file"):
        task.upload_acf(REMOTE, str(acf))
    assert tuning.calls == []
