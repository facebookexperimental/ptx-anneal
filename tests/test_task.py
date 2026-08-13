# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the ACF tuning-task schema (ptx_anneal.task)."""

import json
import os

import pytest

from ptx_anneal import task

SAMPLE_TASK = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_tasks", "sample_task")


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
    spec = json.load(open(os.path.join(SAMPLE_TASK, "spec.json")))
    spec.pop("spec_version", None)
    task.validate_spec(spec)  # must not raise; legacy tasks have no spec_version


def test_legacy_ptx_sha256_alias():
    spec = json.load(open(os.path.join(SAMPLE_TASK, "spec.json")))
    spec.pop("ir_hash", None)  # only the legacy alias remains
    assert task.spec_ir_hash(spec) == spec["ptx_sha256"]
    task.validate_spec(spec)


def test_missing_required_field_rejected():
    spec = json.load(open(os.path.join(SAMPLE_TASK, "spec.json")))
    spec.pop("entry")
    with pytest.raises(task.TaskError):
        task.validate_spec(spec)


def test_unsupported_spec_version_rejected():
    spec = json.load(open(os.path.join(SAMPLE_TASK, "spec.json")))
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
