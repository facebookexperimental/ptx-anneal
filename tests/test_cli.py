# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CLI tests: argument parsing, the _score subcommand, adapter wiring, and doctor formatting (no GPU)."""

import os

import pytest

from ptx_anneal import cli, provision


def test_tune_parser_defaults():
    # engine is provisioning (env), not a required flag; per-run surface is just --ss/--task.
    a = cli.build_parser().parse_args(["--ss", "/s.bin", "--task", "/t", "--output", "/o"])
    assert a.ss == "/s.bin" and a.task == "/t" and a.output == "/o"
    assert a.target == "ptx" and a.engine_python is None and a.engine_adapter is None


def test_engine_overrides_parse():
    a = cli.build_parser().parse_args(["--engine-python", "/py", "--engine-adapter", "/a.py", "--task", "/t"])
    assert a.engine_python == "/py" and a.engine_adapter == "/a.py"


def test_output_aliases_store():
    assert cli.build_parser().parse_args(["--task", "/t", "--store", "/o"]).output == "/o"


def test_unknown_flag_rejected():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--bogus"])


def test_task_is_required():
    # The parser owns this, not a late SystemExit from _cmd_tune -- so the user gets a usage message.
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--output", "/o"])


def test_score_parser():
    a = cli._score_parser().parse_args(["--task", "/t", "--acf", "/a.acf", "--ptxas", "/p"])
    assert a.task == "/t" and a.acf == "/a.acf" and a.ptxas == "/p"
    assert cli._score_parser().parse_args(["--task", "/t"]).acf == "NONE"  # default = untuned baseline


def test_harness_cmd_invokes_this_module():
    # dev (not frozen): the adapter re-invokes `python -m ptx_anneal.cli` for scoring.
    assert "ptx_anneal.cli" in cli._harness_cmd()


def test_bundled_compileiq_adapter_exists():
    p = cli._bundled_adapter()
    assert p.endswith(os.path.join("adapters", "compileiq_adapter.py")) and os.path.exists(p)


def test_format_report_no_gpu():
    rep = {
        "ptxas_path": None,
        "ptxas_version": None,
        "ptxas_min": "13.3",
        "ptxas_ok": False,
        "engines": [],
        "gpu": {"cuda_python": False, "torch": False, "gpu": False, "detail": ""},
    }
    text = provision.format_report(rep)
    assert "ptxas: not found" in text
    assert "real tuning ready: no" in text
