# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the ACF tuning-task schema (ptx_anneal.task)."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from ptx_anneal import cli, task

SAMPLE_TASK = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_tasks", "sample_task")


def _spec() -> dict:
    """A fresh copy of the sample spec, so each test can mutate it freely."""
    with open(os.path.join(SAMPLE_TASK, "spec.json")) as f:
        return json.load(f)


def test_load_sample_task():
    t = task.load(SAMPLE_TASK)
    assert t.entry == "add_kernel"
    assert t.arch == "sm_100a"
    assert t.spec_version == 1
    assert t.ir_hash == "6cb1f53d5136f5b8e3e3c25727534f6d006f6564d26b0e0b51cfd4826e254000"
    assert ".visible .entry add_kernel" in t.ir
    assert len(t.spec["tensors"]) == 3
    assert t.spec["args"][-1] == {"null": True}


def test_spec_version_absent_is_v1():
    spec = _spec()
    spec.pop("spec_version", None)
    task.validate_spec(spec)  # must not raise; legacy tasks have no spec_version


def test_legacy_ptx_sha256_alias():
    spec = _spec()
    spec.pop("ir_hash", None)  # only the legacy alias remains
    assert task.spec_ir_hash(spec) == spec["ptx_sha256"]
    task.validate_spec(spec)


def test_missing_required_field_rejected():
    spec = _spec()
    spec.pop("entry")
    with pytest.raises(task.TaskError):
        task.validate_spec(spec)


def test_unsupported_spec_version_rejected():
    spec = _spec()
    spec["spec_version"] = 999
    with pytest.raises(task.TaskError):
        task.validate_spec(spec)


def test_missing_files_rejected(tmp_path):
    with pytest.raises(task.TaskError):
        task.load(str(tmp_path))  # empty dir: no spec.json


def test_default_task_root_env(monkeypatch):
    monkeypatch.setenv(task.TASK_DIR_ENV, "/tmp/ciq_tasks_xyz")
    assert task.default_task_root() == "/tmp/ciq_tasks_xyz"
    monkeypatch.delenv(task.TASK_DIR_ENV, raising=False)
    assert task.default_task_root().endswith("/.compile_iq/tasks")


@pytest.fixture
def manifold_download(monkeypatch):
    calls = []
    monkeypatch.setattr(task.shutil, "which", lambda name: "/tools/manifold")

    def download(argv, **kwargs):
        assert argv[:2] == ["/tools/manifold", "get"]
        assert kwargs["check"] is False and kwargs["timeout"] > 0
        calls.append(argv)
        shutil.copyfile(Path(SAMPLE_TASK) / Path(argv[2]).name, argv[3])
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(task.subprocess, "run", download)
    return calls


@pytest.mark.parametrize("suffix", ["", "/"])
def test_remote_task_download_and_cleanup(manifold_download, suffix):
    with task.local_task_dir("manifold://bucket/tree/task" + suffix) as local_dir:
        assert task.load(local_dir).entry == "add_kernel"
        assert [call[2] for call in manifold_download] == [
            "bucket/tree/task/spec.json", "bucket/tree/task/kernel.ptx"
        ]
    assert not Path(local_dir).exists()


def test_local_task_needs_no_manifold(monkeypatch):
    def unexpected_lookup(name):
        pytest.fail("local tasks must not look up a download tool")

    monkeypatch.setattr(task.shutil, "which", unexpected_lookup)
    with task.local_task_dir(Path(SAMPLE_TASK)) as local_dir:
        assert local_dir == SAMPLE_TASK
    assert Path(local_dir).is_dir()


def test_remote_task_cleanup_on_caller_failure(manifold_download):
    with (
        pytest.raises(RuntimeError, match="replay failed"),
        task.local_task_dir("manifold://bucket/tree/task") as local_dir,
    ):
        assert Path(local_dir).is_dir()
        raise RuntimeError("replay failed")
    assert not Path(local_dir).exists()


@pytest.mark.parametrize("failure", ["nonzero", "missing_file", "timeout"])
def test_failed_download_cleans_partial_task(monkeypatch, manifold_download, failure):
    successful_download = task.subprocess.run
    destinations = []

    def download(argv, **kwargs):
        destinations.append(Path(argv[3]))
        if argv[2].endswith("/spec.json"):
            return successful_download(argv, **kwargs)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 120)
        return SimpleNamespace(returncode=1 if failure == "nonzero" else 0, stderr="download failed", stdout="")

    monkeypatch.setattr(task.subprocess, "run", download)
    with pytest.raises(task.TaskError, match="failed to download manifold://bucket/tree/task/kernel.ptx"):
        with task.local_task_dir("manifold://bucket/tree/task"):
            pytest.fail("a partial download must not be passed to the scorer")
    assert len(destinations) == 2
    assert not destinations[0].parent.exists()


def test_missing_manifold_cli_is_actionable(monkeypatch):
    monkeypatch.setattr(task.shutil, "which", lambda name: None)
    with pytest.raises(task.TaskError, match="'manifold' CLI on PATH"):
        with task.local_task_dir("manifold://bucket/tree/task"):
            pytest.fail("missing downloader must fail")


@pytest.mark.parametrize("url", [
    "https://bucket/task", "manifold:///tree/task", "manifold://bucket",
    "manifold://bucket/tree/task?query=yes",
])
def test_invalid_task_url_rejected(url):
    with pytest.raises(task.TaskError):
        with task.local_task_dir(url):
            pytest.fail("invalid URL must fail")


@pytest.mark.parametrize("command", ["replay", "tune"])
def test_cli_keeps_download_for_whole_run(monkeypatch, manifold_download, command):
    used_dirs = []

    def run(args):
        used_dirs.append(args.task)
        assert task.load(args.task).entry == "add_kernel"
        return 0

    monkeypatch.setattr(cli, "_cmd_" + command, run)
    argv = ["replay"] if command == "replay" else []
    assert cli.main([*argv, "--task", "manifold://bucket/tree/task"]) == 0
    assert len(manifold_download) == 2
    assert len(used_dirs) == 1 and not Path(used_dirs[0]).exists()


@pytest.mark.parametrize("command", ["replay", "tune"])
@pytest.mark.parametrize("filename", ["spec.json", "spec.json/", "kernel.ptx"])
def test_cli_rejects_manifold_file_urls(manifold_download, capsys, command, filename):
    argv = ["replay"] if command == "replay" else []
    assert cli.main([*argv, "--task", f"manifold://bucket/tree/task/{filename}"]) == 1
    assert "must name a directory, not a file; use manifold://bucket/tree/task" in capsys.readouterr().err
    assert manifold_download == []


def test_cli_reports_download_error_without_starting_replay(monkeypatch, capsys):
    monkeypatch.setattr(task.shutil, "which", lambda name: None)
    monkeypatch.setattr(cli, "_cmd_replay", lambda args: pytest.fail("replay started without task files"))
    assert cli.main(["replay", "--task", "manifold://bucket/tree/task"]) == 1
    assert "'manifold' CLI on PATH" in capsys.readouterr().err
