# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The search-problem fingerprint (CPU-only; nvidia-smi is stubbed, never required).

The fingerprint's whole value is that two channels can be diffed, so what is tested here is exactly
that: it covers every input that can move a result, it is canonical (equal problems render
identically), and its digest actually changes when an input changes. A fingerprint that quietly
ignores a field would make two different search problems look like the same one -- which is worse
than having no fingerprint at all.
"""

import pytest

import ptx_anneal.fingerprint as fp

_KERNEL = {"name": "k", "shapes": [[512, 512]], "dtypes": ["torch.float16"], "pinned_config": {"BLOCK_M": 128}}
_PTXAS = {"path": "/usr/bin/ptxas", "version": "13.3", "flags": []}
_SS = {"resolved_tag": "search-spaces-2026.05.22", "sha256": "deadbeef", "variant": "default"}
_ENGINE = {"name": "compileiq", "version": "1.0", "budget": {"generations": 5, "pool_size": 32}}
_OBJ = {"metric": "min_ms", "timing": {"method": "do_bench", "warmup": 100, "rep": 1000}}
_ENV = {"gpu": "NVIDIA B200", "cuda_version": "13.0", "sm_clock_mhz": 1965}


def _build(**over):
    kw = {
        "channel": "triton-native",
        "kernel": _KERNEL,
        "frontend": {"name": "fbtriton", "version": "3.8.0+fb"},
        "ptxas": _PTXAS,
        "search_space": _SS,
        "engine": _ENGINE,
        "objective": _OBJ,
        "env": _ENV,
    }
    kw.update(over)
    return fp.build(**kw)


def test_covers_every_search_problem_input():
    """§3 of the design principles enumerates what must be in here; none of it may be dropped."""
    got = _build()
    assert set(got) == {
        "channel",
        "kernel",
        "frontend",
        "ptxas",
        "search_space",
        "engine",
        "objective",
        "env",
        "digest",
    }
    # ptxas FLAGS, not just the version: the same binary with different flags is a different compiler.
    assert "flags" in got["ptxas"]
    # The RESOLVED tag. "We both asked for latest" is not agreement.
    assert got["search_space"]["resolved_tag"]


def test_is_canonical_and_order_independent():
    """Two equal problems must render byte-identical, whatever order the dicts were built in."""
    fields = {"name": "k", "shapes": [[512, 512]], "dtypes": ["torch.float16"], "pinned_config": {"BLOCK_M": 128}}
    a = _build(kernel=dict(fields.items()))
    b = _build(kernel=dict(reversed(list(fields.items()))))
    assert fp.canonical(a) == fp.canonical(b)
    assert a["digest"] == b["digest"]


def test_digest_excludes_itself():
    """Otherwise the digest would depend on the digest and never be reproducible."""
    got = _build()
    assert "digest" not in fp.canonical(got)
    assert fp.digest(got) == got["digest"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("channel", "ptx-direct"),
        ("kernel", {**_KERNEL, "pinned_config": {"BLOCK_M": 64}}),
        ("ptxas", {**_PTXAS, "version": "13.4"}),
        ("ptxas", {**_PTXAS, "flags": ["-O2"]}),
        ("search_space", {**_SS, "resolved_tag": "search-spaces-2026.06.01"}),
        ("engine", {**_ENGINE, "budget": {"generations": 1, "pool_size": 8}}),
        ("objective", {**_OBJ, "timing": {"method": "do_bench", "warmup": 25, "rep": 50}}),
        ("env", {**_ENV, "sm_clock_mhz": 1200}),
        # The frontend is a live input in the triton-native channel: two fbtriton builds are two
        # different experiments, and the digest has to say so.
        ("frontend", {"name": "fbtriton", "version": "3.9.0+fb"}),
    ],
)
def test_any_changed_input_changes_the_digest(field, value):
    """The point of the digest: a silent divergence between channels is impossible."""
    assert _build()["digest"] != _build(**{field: value})["digest"]


def test_frontend_defaults_to_empty_not_absent():
    """The PTX-direct channel has no frontend at tune time (it replays a frozen PTX). Recording {}
    keeps the key present so a diff against triton-native shows the difference rather than hiding it."""
    got = _build(frontend=None)
    assert got["frontend"] == {}
    assert got["digest"] != _build(frontend={"name": "fbtriton", "version": "3.8.0+fb"})["digest"]


def test_render_carries_the_digest_and_is_diffable():
    text = fp.render(_build())
    assert _build()["digest"] in text
    # Indented + sorted, so `diff <(chanA) <(chanB)` points at the field that differs, not the line.
    assert '\n  "channel": "triton-native"' in text


def test_ptxas_flags_excludes_the_candidate_acf(monkeypatch):
    """The ACF is the candidate under test, not a flag -- including it would make every candidate a
    different search problem and the digest would never match anything."""
    monkeypatch.setenv("PTXAS_OPTIONS", "--apply-controls=/tmp/x.acf -O3")
    assert fp.ptxas_flags() == ["-O3"]


def test_ptxas_flags_empty_when_unset(monkeypatch):
    monkeypatch.delenv("PTXAS_OPTIONS", raising=False)
    assert fp.ptxas_flags() == []


def test_gpu_env_degrades_without_nvidia_smi(monkeypatch):
    """No GPU tooling is a recordable state, not a crash: the fingerprint still renders."""
    monkeypatch.setattr(fp.shutil, "which", lambda _: None)
    env = fp.gpu_env()
    assert env["gpu"] is None and env["cuda_version"] is None
    assert "sm_clock_mhz" in env  # the key is present-but-null, so a diff shows "unknown", not absent


def test_gpu_env_parses_nvidia_smi(monkeypatch):
    monkeypatch.setattr(fp.shutil, "which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(fp, "_driver_cuda_version", lambda: "13.0")
    monkeypatch.setattr(
        fp,
        "_smi",
        lambda query, index: (
            ["NVIDIA B200", "580.82.07", "1965", "1965", "1965", "Enabled"]
            if "name" in query
            else ["0x0000000000000001"]
        ),
    )
    env = fp.gpu_env(0)
    assert env["gpu"] == "NVIDIA B200"
    assert env["sm_clock_mhz"] == 1965 and env["max_sm_clock_mhz"] == 1965
    assert env["clock_event_reasons"] == "0x0000000000000001"


def test_gpu_env_follows_cuda_visible_devices(monkeypatch):
    """The fingerprint must describe the GPU that was measured, not device 0."""
    seen = {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,4")
    monkeypatch.setattr(fp, "_driver_cuda_version", lambda: None)
    monkeypatch.setattr(fp, "_smi", lambda query, index: seen.setdefault("index", index) and None)
    fp.gpu_env()
    assert seen["index"] == "3"


def test_ptxas_version_of_none_is_none_not_a_crash():
    """ "No ptxas found" is a reportable state, like any other unusable path. It used to raise a
    TypeError from inside subprocess, which took out callers that had already handled a None return."""
    from ptx_anneal import provision

    assert provision.ptxas_version(None) is None
    assert provision.ptxas_version("") is None
