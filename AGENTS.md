# AGENTS.md — guide for coding agents working on ptx-anneal

This file orients automated coding agents. Humans: see `README.md`.

## What this repo is

`ptx-anneal` is an offline **ptxas tuning framework** (the "ACF factory") built on NVIDIA CompileIQ. 
It searches `ptxas` advanced controls for a compiled GPU kernel, validates the
result is numerically correct against the untuned baseline, and caches the
winning tuning artifact (an **ACF**, Advanced Control File) in a
content-addressed store. It is the orchestration **harness only** — the
optimization engine is bring-your-own (see below).

PTX in, ACF out

## Hard rules

- **Must run from a plain git checkout.** `ptx_anneal` (the ACF factory) and the
  `MODE=tune` validation are what we share with the NVIDIA CompileIQ team, so they
  must work with nothing but Python, `ptxas` and the engine — no build system, no
  fbcode paths, no `fb/`. This is the constraint most likely to be broken by a
  well-meaning change; check it before adding any dependency.
- **Pure Python, no build system beyond `hatchling`,** under `ptx_anneal/` and
  `e2e/*.sh`. Anything needing Buck lives in `fb/` (site scaffolding, stripped) —
  today the `MODE=full` frontend orchestration. `e2e/BUCK` is the one Buck file
  outside `fb/`: it only *defines* the frontend target from `e2e/` sources and is
  inert in a git checkout.
- **Never vendor the engine.** The CompileIQ engine is bring-your-own and must
  never be committed, downloaded into the tree, or imported directly. The harness
  reaches it only by spawning an adapter script (`ptx_anneal/adapters/`) in the
  engine's own interpreter, so nothing here imports it.
- **No internal-only code in the public tree.** Inhouse APIs/services live either
  out-of-tree (registered via `entry_points`) or in the stripped `fb/`. Nothing
  outside `fb/` may import or require them; `fb/` is optional by construction.
- **Every source file carries the MIT license header.**
- **Leave changes uncommitted** unless explicitly asked to commit.

## Layout

- `ptx_anneal/target/` — `Target` ABC + the `ptx` target (the only v1 target;
  other targets such as AMD `gcn` are future registered plugins).
- `ptx_anneal/store/` — `Store` ABC + `LocalStore` (filesystem). Keys are
  `(target, arch, ir_hash)`.
- `ptx_anneal/run/` — `Runner` ABC + `LocalRunner` (single host).
- `ptx_anneal/search/` — `SearchBackend` ABC + the entry-point registry.
- `ptx_anneal/log/` — `Logger` ABC + `NullLogger` (default) / `StderrLogger`.
- `ptx_anneal/factory.py` — the library orchestrator (search → validate → admit →
  store). Library API only — the CLI does not call it; see "Two tuning paths".
- `ptx_anneal/task.py` — the task (`kernel.ptx` + `spec.json`) loader.
- `ptx_anneal/validate.py` — `Validator` ABC + `RelTolValidator`.
- `ptx_anneal/ptx_launch.py` — the PTX-direct driver-launch benchmarking harness.
- `ptx_anneal/adapters/` — engine adapters (standalone scripts run in the engine's
  own interpreter); `compileiq_adapter.py` is the bundled default.
- `ptx_anneal/provision.py` / `cli.py` — toolchain detection (`ptx-anneal doctor`)
  and the CLI (`doctor`, `tune`).
- `e2e/` — the adhoc validation harness (`e2e.sh`, `kernels/`); never packaged.
  `MODE=tune` needs only the factory's deps; `kernels/` + `run.py` need torch+triton.
- `fb/` — site scaffolding, ShipIt-stripped and absent from git checkouts. Today:
  `e2e_internal.sh`, the buck-driven `MODE=full` frontend. Optional by construction.
- `sample_tasks/` — pre-captured `kernel.ptx` + `spec.json` fixtures (torch-free data).
- `tests/`, `skills/`.

## One toolchain

`ptxas >= 13.3` + CompileIQ, everywhere. The floor is fixed (`provision.MIN_PTXAS`),
not a default to relax: `--apply-controls` is GA only from 13.3, and the frontend
derives its emitted PTX ISA version from *its* ptxas version — so mixing versions
across collect/factory/consume changes the PTX, changes the store key, and silently
MISSes. Do not add a version-override knob or a fallback path for an older ptxas.

## Two tuning paths (know which one you are editing)

There are currently two, and they do not share code:

1. **`ptx_anneal tune` (the CLI, what ships).** `cli._cmd_tune` scores the baseline
   by calling `run/cudapy_worker.py:score` directly, spawns the engine adapter as a
   subprocess, and admits through `LocalStore`. It imports only `provision`, `task`,
   `store` and `target` — **not** `factory`, `SearchBackend` or any `Runner`.
2. **`factory.tune` (the library DI API).** Takes `Store`/`Runner`/`SearchBackend`/
   `Logger` implementations as arguments. Exercised by `tests/` only.

So `factory.tune`, `load_backend`, `LocalRunner`, `CudaPyRunner` and the `Logger`
implementations are currently reachable from tests and embedders, not from the CLI.
`provision.report()` (i.e. `doctor`) inspects the `ptx_anneal.search_backends`
registry that path (2) uses, which is why it can report "no engine" on a setup where
path (1) tunes fine. Do not "fix" a doc or a report to claim the CLI uses the ABCs —
either wire the CLI through `factory.tune`, or keep the distinction explicit.

## Conventions

- `Target`/`SearchBackend`/`Runner`/`Store`/`Logger` are ABCs with local/null
  implementations shipped here; out-of-tree ones are injected by constructing them
  and passing them to `factory.tune` (no registry needed for runner/store/logger).
- Interop env vars from a frontend (e.g. fbtriton) — `TRITON_COMPILE_IQ_*`,
  `COMPILE_IQ_STORE`, `COMPILE_IQ_TASK_DIR` — are an external contract; do not
  rename them. ptx-anneal's own knobs use the `PTX_ANNEAL_*` prefix.

## Agent skills

Task-specific playbooks live in `skills/` as plain markdown (provisioning,
factory search, ACF store). They are vendor-agnostic and usable by any agent.
