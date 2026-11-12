# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the pluggable candidate validator."""

import pytest

from ptx_anneal.validate import (
    DEFAULT_REL_TOL,
    REL_TOL_ENV,
    VALIDATOR_ENV,
    RelTolValidator,
    ValidationError,
    Validator,
    load_validator,
)


# Illustrates a custom validator (subclassing the ABC); also exercised directly below.
class AlwaysReject(Validator):
    def check(self, deviations):
        raise ValidationError("nope")


def test_reltol_accepts_within_tolerance():
    RelTolValidator(1e-2).check([0.0, 1e-3, 1e-2])  # no raise


def test_reltol_rejects_over_tolerance():
    with pytest.raises(ValidationError):
        RelTolValidator(1e-2).check([2e-2])


def test_reltol_rejects_nan_and_inf():
    with pytest.raises(ValidationError):
        RelTolValidator(1e-2).check([float("nan")])
    with pytest.raises(ValidationError):
        RelTolValidator(1e-2).check([float("inf")])


def test_reltol_empty_is_accepted():
    RelTolValidator(1e-2).check([])  # no buffers compared -> trivially fine


def test_custom_validator_subclass():
    with pytest.raises(ValidationError):
        AlwaysReject().check([0.0])


def test_load_validator_default(monkeypatch):
    monkeypatch.delenv(VALIDATOR_ENV, raising=False)
    monkeypatch.delenv(REL_TOL_ENV, raising=False)
    v = load_validator()
    assert isinstance(v, RelTolValidator) and v.rel_tol == DEFAULT_REL_TOL


def test_load_validator_rel_tol_env(monkeypatch):
    monkeypatch.delenv(VALIDATOR_ENV, raising=False)
    monkeypatch.setenv(REL_TOL_ENV, "5e-3")
    assert load_validator().rel_tol == 5e-3


def test_load_validator_import_path(monkeypatch):
    # The import-path branch resolves the named class and ignores REL_TOL_ENV (it constructs with no
    # args) -> default tol proves the import-path branch ran rather than the rel-tol-env default.
    monkeypatch.setenv(VALIDATOR_ENV, "ptx_anneal.validate:RelTolValidator")
    monkeypatch.setenv(REL_TOL_ENV, "5e-3")
    v = load_validator()
    assert isinstance(v, RelTolValidator) and v.rel_tol == DEFAULT_REL_TOL


def test_load_validator_bad_import_path(monkeypatch):
    monkeypatch.setenv(VALIDATOR_ENV, "ptx_anneal.validate:DoesNotExist")
    with pytest.raises(RuntimeError):
        load_validator()


# --- per-buffer relative deviation (torch scorer + numpy worker); optional deps, skipped in CI ---
def _torch_deviation():
    torch = pytest.importorskip("torch")
    from ptx_anneal.target.ptx import relative_deviation

    return torch.tensor, relative_deviation


def _numpy_deviation():
    np = pytest.importorskip("numpy")
    pytest.importorskip("cuda.bindings")
    from ptx_anneal.run.cudapy_worker import relative_deviation

    return np.array, relative_deviation


@pytest.mark.parametrize("make", [_torch_deviation, _numpy_deviation])
def test_deviation_ignores_matching_nonfinite(make):
    arr, dev = make()
    nan, inf = float("nan"), float("inf")
    ref = arr([1.0, nan, inf, -inf, 4.0])
    assert dev(ref, ref) == 0.0  # a NaN-producing baseline must accept itself
    assert dev(ref, arr([1.0, nan, inf, -inf, 4.2])) == pytest.approx(0.05)


_NAN, _INF = float("nan"), float("inf")


@pytest.mark.parametrize("make", [_torch_deviation, _numpy_deviation])
@pytest.mark.parametrize("cur", [[1.0, 2.0, 3.0], [1.0, _NAN, _INF], [_NAN, 2.0, -_INF]])
def test_deviation_rejects_changed_nonfinite_pattern(make, cur):
    arr, dev = make()
    ref = arr([1.0, float("nan"), float("-inf")])
    assert dev(ref, arr(cur)) == float("inf")
    with pytest.raises(ValidationError):
        RelTolValidator().check([dev(ref, arr(cur))])


@pytest.mark.parametrize("make", [_torch_deviation, _numpy_deviation])
def test_deviation_all_nonfinite_buffer(make):
    arr, dev = make()
    ref = arr([float("nan"), float("inf")])
    assert dev(ref, ref) == 0.0
