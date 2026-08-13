# ptx-anneal e2e validation

Adhoc end-to-end validation of the ptx-anneal factory on real kernels. `e2e.sh` runs the factory
(search → score → admit an ACF) on a pre-captured task from `../sample_tasks/`, using **CompileIQ**
with **ptxas ≥ 13.3** (the GA `--apply-controls` path).

## ⚠️ Dev/test only — NEVER packaged

This folder needs **torch + triton** and is for **adhoc validation only**. It is deliberately
excluded from every package build:

- **wheel** — ships only the `ptx_anneal` package (`pyproject.toml`
  `[tool.hatch.build.targets.wheel] packages = ["ptx_anneal"]`), so `e2e/` is never in it.
- **sdist** — `e2e` is not in the sdist `include` list and is in its `exclude` list.

If you add a build target/manifest, keep `e2e/` out of it.

## Usage

```bash
./e2e.sh                                   # factory-only smoke on ../sample_tasks/sample_task
TASK=../sample_tasks/<name> ./e2e.sh       # a different captured task
MODE=full ./e2e.sh                         # 3-step collect -> factory -> consume (synth_acf)
MODE=full ./e2e.sh synth_acf ws_gemm       # ... on specific registered kernels
```

`MODE=tune` (default) runs the **factory only** — search → score → admit; no frontend, no
collect/consume. Env: `GPU`, `BENCH` (`cudagraph`|`do_bench`), `FORCE_ADMIT`, `PTXAS` (ptxas ≥ 13.3),
`SS` / `$COMPILE_IQ_SEARCH_SPACE_BIN` (CompileIQ search space), `PYTHON` (env with the factory deps),
`STORE`.

**`MODE=tune` is the portable one.** It is pure Python + `ptxas` + the engine, so it must keep
working from a plain git checkout with no build system. Do not add a build-system dependency to it.

`MODE=full` runs the whole flow on a registered kernel (`kernels/`): collect a task from a real
compile, tune it, then consume the stored ACF and let the launch-time free-win A/B decide. That needs
a **frontend** — a triton built with the magnon collector/consumer — and building one is
site-specific, so `run_full` lives in the optional hook `../fb/e2e_internal.sh` that `e2e.sh` sources
when present. Without the hook, `MODE=full` reports that it is unavailable and `MODE=tune` still
works. The hook adds its own env (`BUCK_CUDA`, `NVCC_ARCH`, …); `NO_ADMIT_OK` (kernels allowed to
admit nothing) and `FORCE_APPLY` (skip the A/B — plumbing gate only) apply there.

> **Collect, factory and consume must all use the same `ptxas`.** Triton derives the emitted PTX ISA
> version from the ptxas version, so a different ptxas produces different PTX and therefore a
> different store key — every lookup would MISS and silently fall back to plain. Keeping the frontend
> on the factory's ptxas is the hook's job.

There is a single toolchain — CompileIQ + ptxas ≥ 13.3 — so there is nothing to select. `PTXAS`
comes from `$PTXAS` or `PATH` (the hook may point it at an in-repo build).

## Adding a kernel

Drop a module in `kernels/` that calls `register(KernelSpec(...))` (see `registry.py`). No harness
edits needed.

## Sample tasks

Pre-captured `kernel.ptx` + `spec.json` fixtures live in `../sample_tasks/`. A task is a frozen
`kernel.ptx`, so the factory can replay it under any ptxas ≥ 13.3 — but the ACF it produces is only
consumable by a frontend whose own ptxas re-emits that same PTX (see the note above).
