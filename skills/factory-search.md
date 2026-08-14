# Skill: Running a factory search (tuning)

Use this to tune a kernel — drive the engine over `ptxas` controls, validate the
result, and admit the winner to the store.

## Inputs

A **tuning task**: a fixed compiled IR (for the `ptx` target, a `kernel.ptx`) plus
a launch `spec` describing how to assemble and launch it. A frontend produces
these (e.g. fbtriton's collection hook writes one per compiled kernel).

## Run

```bash
ptx-anneal --task <task_dir> --output <store_dir>
```

`--task` is the only required argument. There is no `tune` subcommand — tuning is the
default action (`doctor` and the internal `_score` are the only subcommands), and
`--store` is an alias for `--output`. `ptx-anneal --help` lists both forms, the
option groups, and the environment-only knobs.

Nothing else is required: the engine comes from `pip install ptx-anneal[compileiq]`,
it fetches its own search space, and `ptxas` is discovered (env → `PATH` → the
`nvidia-cuda-nvcc` wheel). See `skills/provisioning.md`.

What happens (PTX-direct — no frontend recompilation in the loop):
1. The engine proposes candidate controls.
2. Each candidate is **assembled** with `ptxas --apply-controls` and the cubin is
   launched via the driver API on real-shaped inputs.
3. Each candidate is **validated** against the untuned (no-ACF) baseline: per-buffer
   relative deviation is handed to the pluggable validator, which rejects the
   candidate on failure (an ACF changes codegen, never the math).
4. The fastest *correct* candidate is **admitted** and its artifact written to the
   store under `(target, arch, ir_hash)`, tagged by ptxas version.

## Reading the banner

The first lines identify every input that can change the outcome:

```
ptxas=<path> (V13.3) ; engine-python=<python> ; adapter=compileiq_adapter.py ; task=<dir>
baseline: 0.0010 ms (bench=cudagraph)
engine: compileiq 1.0.0.dev1 from <site-packages>/compileiq (python=<python>)
search space: ptxas13.3_search_space.bin [cache search-spaces-2026.05.22]
```

`engine:` and `search space:` are printed by the *adapter*, not the harness — the
harness never imports the engine and the engine may run under another interpreter,
so only the adapter can report them. A bring-your-own adapter should print its own.

## Knobs

Per-run flags: `--task`, `--output`/`--store`, `--bench` (`cudagraph`|`do_bench`),
`--warmup`, `--rep`, `--target`, and the pins `--ptxas`, `--ss`, `--engine-python`,
`--engine-adapter`.

Environment:

| Var | Effect |
|-----|--------|
| `PTXAS` / `TRITON_PTXAS_BLACKWELL_PATH` | pin ptxas (used **verbatim**, even if below the floor) |
| `PTX_ANNEAL_FORCE_ADMIT` | admit the best valid candidate even without a win (smoke knob) |
| `PTX_ANNEAL_ENGINE_PYTHON` / `PTX_ANNEAL_ENGINE_ADAPTER` | engine interpreter / adapter script |
| `PTX_ANNEAL_WORKER_PYTHON` | interpreter for the per-candidate benchmark (`CudaPyRunner`) |
| `CIQ_POOL` / `CIQ_GENERATIONS` | engine budget (default 8 / 1 — one batch, no evolution) |
| `CIQ_SS_TAG` / `CIQ_SS_VERSION` / `CIQ_SS_VARIANT` | pin the published search-space catalog |
| `CIQ_SEARCH_SPACES_DIR` | offline search-space mirror (no network) |

The `CIQ_*` knobs belong to the engine and are read by the adapter, not the harness.
The minimum ptxas version is **not** a knob: it is fixed at `provision.MIN_PTXAS`
(13.3). Applying an artifact (`ptxas --apply-controls`) is GA only from there on,
and the frontend's ptxas decides the PTX ISA version it emits — so a lower floor
just yields artifacts whose key nothing will ever match.

## Reading results

- **The exit code does not report admission.** `ptx-anneal` returns 0 whether or not
  an ACF was admitted — "no headroom on this kernel" is an outcome, not an error. To
  gate on it, match `admitted ACF` in the output (this is what `e2e/e2e.sh` does).
- "no candidate admitted" → either every candidate failed validation/wedged, or
  none beat the baseline. Check the per-candidate log; widen the search
  (`CIQ_GENERATIONS`, `CIQ_POOL`) or verify the task's spec/inputs.
- A search-time "win" is measured in the search harness and is noisy — at the default
  budget it is best-of-8 sampling, not an evolutionary search, and repeat runs on the
  same task vary by a few percent. The trustworthy decision is made when the artifact
  is consumed (an A/B vs. the plain baseline). Treat search-time numbers as a ranking
  signal, not a promise.
- Every admitted ACF records its inputs — ptxas version, engine build, and the exact
  search space (`resolved_tag` + `sha256`) — in the `.acf.json` sidecar. Start there
  when a result is surprising or won't reproduce. See `skills/acf-store.md`.

## Isolation / safety

Mind which of the two tuning paths you are reasoning about (see `AGENTS.md`):

- **`factory.tune` (library DI API).** `CudaPyRunner.score` benchmarks each candidate
  in its own subprocess with a `timeout`, so a candidate that hangs or wedges the GPU
  is reaped and scored `INVALID`.
- **`ptx-anneal --task` (the CLI, what ships).** Each candidate is still scored in a
  separate `_score` subprocess, and any exception there is caught and reported as
  `INVALID` — but that subprocess is spawned **without a timeout**, and `ptxas_compile`
  has none either. A candidate that truly hangs will hang the run rather than being
  reaped. Correctness is checked before timing in both paths.
