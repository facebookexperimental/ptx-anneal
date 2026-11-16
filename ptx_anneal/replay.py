# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Standalone PTX replay, including BF16, strides and TMA.

Tensors with a captured input pickle in the task directory are loaded from it; the rest are
synthesized. Only the captured PTX is compiled. Torch allocates inputs and Triton encodes host TMA
descriptors; neither a frontend kernel nor an optimization engine is invoked. GPU dependencies are lazy.

:func:`replay` benchmarks the untuned PTX. :func:`replay_acf` A/Bs an ACF against it: both cubins run
once on identical fresh inputs, every buffer is compared (the same self-referential oracle tuning uses),
and then each is timed with the same method.
"""

from __future__ import annotations

from dataclasses import dataclass

from .task import Task


@dataclass
class AcfReport:
    """The outcome of :func:`replay_acf`. ``numerics_error`` is ``None`` iff the validator accepted."""

    baseline_ms: float
    acf_ms: float
    deviations: list[float]
    numerics_error: str | None = None
    validator: str = ""

    @property
    def numerics_ok(self) -> bool:
        return self.numerics_error is None

    @property
    def speedup(self) -> float:
        return self.baseline_ms / self.acf_ms

    @property
    def win(self) -> float:
        """Fractional time saved by the ACF (positive is faster)."""
        return (self.baseline_ms - self.acf_ms) / self.baseline_ms


def _check_bench_args(warmup: int, rep: int, bench: str) -> None:
    if warmup < 0 or rep < 1:
        raise ValueError("warmup must be nonnegative and rep must be positive")
    if bench not in ("cudagraph", "do_bench"):
        raise ValueError(f"unknown benchmark method: {bench}")


def _import_torch():
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("replay requires GPU dependencies: pip install 'ptx-anneal[harness,cudapy]'") from e
    return torch


class _Launched:
    """A loaded kernel that has run once on fresh inputs; owns everything its launch args reference."""

    def __init__(self, task: Task, ptxas: str, acf_path: str | None):
        torch = _import_torch()
        from . import ptx_launch as L

        spec = task.spec
        compile_kw = {"acf_path": acf_path} if acf_path else {}
        cubin = L.ptxas_compile(task.ir, ptxas, arch=task.arch, **compile_kw)
        self.kernel = L.load_cubin(cubin, task.entry, spec["shared"], cluster=spec.get("cluster"))
        try:
            self.tensors = L.alloc_tensors(spec, seed=0, task_dir=task.dir)
            # The tensors, descriptor objects and scratch must outlive graph capture and its replays.
            self._tensormaps, addresses = L.build_tensormaps(spec, self.tensors)
            self._scratch = L.alloc_global_scratch(spec)
            self.args = L.kernel_args_from_spec(spec, self.tensors, addresses, self._scratch)
            # The benchmark uses a nonblocking stream; finish torch's input initialization first.
            torch.cuda.synchronize()
            self.kernel.launch(spec["grid"], spec["block"], self.args)
            torch.cuda.synchronize()
        except BaseException:
            self.kernel.unload()
            raise

    def bench(self, spec: dict, warmup: int, rep: int, bench: str) -> float:
        from . import ptx_launch as L

        if bench == "do_bench":
            ms = L.bench_ms_do_bench(self.kernel, spec["grid"], spec["block"], self.args, warmup=warmup, reps=rep)
        else:
            ms = L.bench_ms_cudagraph(
                self.kernel, spec["grid"], spec["block"], self.args, warmup=warmup, reps=max(rep // 10, 3)
            )
        return float(ms)

    def unload(self) -> None:
        self.kernel.unload()


def replay(task: Task, ptxas: str, *, warmup: int = 25, rep: int = 50, bench: str = "cudagraph") -> float:
    """Return per-launch milliseconds. This is a benchmark, not an output-correctness check."""
    _check_bench_args(warmup, rep, bench)
    run = _Launched(task, ptxas, None)
    try:
        return run.bench(task.spec, warmup, rep, bench)
    finally:
        run.unload()


def replay_acf(
    task: Task, ptxas: str, acf_path: str, *, warmup: int = 25, rep: int = 50, bench: str = "cudagraph"
) -> AcfReport:
    """A/B one ACF against the untuned PTX: numerics first, then timing.

    Both kernels launch once on separately allocated but identical inputs, and every buffer is compared
    *before* either is benchmarked -- a benchmark re-launches the kernel, so in-place/accumulating
    outputs are no longer comparable afterwards. A numerics failure is reported, not raised: the
    timings are still measured so the report is complete. Compilation and launch failures raise.
    """
    _check_bench_args(warmup, rep, bench)
    from .target.ptx import relative_deviation
    from .validate import ValidationError, load_validator

    spec = task.spec
    validator = load_validator()
    base = _Launched(task, ptxas, None)
    try:
        cand = _Launched(task, ptxas, acf_path)
        try:
            deviations = [relative_deviation(r, c) for r, c in zip(base.tensors, cand.tensors, strict=True)]
            error = None
            try:
                validator.check(deviations)
            except ValidationError as e:
                error = str(e)
            # Timed back to back with the same method; the baseline goes first, as in tuning.
            baseline_ms = base.bench(spec, warmup, rep, bench)
            acf_ms = cand.bench(spec, warmup, rep, bench)
        finally:
            cand.unload()
    finally:
        base.unload()
    desc = type(validator).__name__
    if hasattr(validator, "rel_tol"):
        desc += f"(rel_tol={validator.rel_tol:g})"
    return AcfReport(baseline_ms, acf_ms, deviations, error, desc)
