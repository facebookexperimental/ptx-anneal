# PTX-anneal

Offline ptxas tuning framework (ACF factory) for faster GPU kernels.

`ptx-anneal` searches ACF (Advanced Control File, encrypted `ptxas` advanced controls) for a compiled GPU kernel, verifies
the result is numerically correct against the untuned baseline, and caches the winning
ACF in a content-addressed store. Applying a cached ACF at
runtime yields faster SASS with no change to your kernel source or on-disk compilation cache.

`ptx-anneal` is the orchestration harness only. The optimization engine
([NVIDIA CompileIQ](https://github.com/nvidia/compileiq)) is **bring-your-own** and
provisioned separately; `ptx-anneal` treats it as a black box behind a plugin interface.

## How it works — a complete local flow with [fbtriton](https://github.com/facebookexperimental/triton)

Step 1. Task collection: fbtriton's JIT hook dumps each compiled PTX as an ACF tuning task.
```bash
TRITON_COMPILE_IQ_COLLECT=1 COMPILE_IQ_TASK_DIR=$TASKS python your_workload.py
```

Step 2. ACF factory: Search for the ACF whose ptxas-assembled SASS runs fastest.
```bash
# one engine-free binary. Install the search engine (e.g. NVIDIA CompileIQ) into the same env as
# ptx_anneal; ptxas (>= 13.3, GA --apply-controls) comes from $PTXAS / $TRITON_PTXAS_BLACKWELL_PATH / PATH.
# Engine + ptxas are provisioning (env), so the per-run command is just the task + search space:
PTXAS=<ptxas-13.3> ptx_anneal --ss <search_space> --task $TASKS/<task> --output $STORE
```
Run `ptx_anneal doctor` first to check the toolchain.

Step 3. ACF consumption: On an ACF-store hit, re-assemble the kernel with the ACF (ptxas --apply-controls) and run the tuned SASS.
```bash
TRITON_COMPILE_IQ_APPLY=1 COMPILE_IQ_STORE=$STORE python your_workload.py
```

## How the CLI runs a search

`ptx_anneal --task ...` is local and single-host: it scores the untuned baseline in-process,
then spawns the **engine adapter** as a subprocess and lets the engine drive the search, scoring
each candidate by calling this same binary back as `ptx_anneal _score`. The winner is admitted to a
local filesystem store. The harness never imports the engine, so it can be a frozen binary and the
engine can live in its own interpreter.

```
ptx_anneal tune ──> baseline score (in-process)
              └───> $PTX_ANNEAL_ENGINE_ADAPTER (subprocess, engine's own interpreter)
                        └──> ptx_anneal _score <candidate>   (one call per candidate)
                    admit winner ──> LocalStore
```

## Extending ptx-anneal

Two seams are wired into the CLI:

| Seam | How | Ships here |
|------|-----|-----------|
| Search engine | `--engine-adapter` / `$PTX_ANNEAL_ENGINE_ADAPTER` — a standalone script run in the engine's interpreter (see `ptx_anneal/adapters/`) | `compileiq_adapter.py` (bundled default) |
| Per-IR backend | `--target`, resolved from built-ins + the `ptx_anneal.targets` entry point | `PtxTarget` (`ptx`) |

Env knobs: `PTX_ANNEAL_ENGINE_PYTHON` (interpreter that has the engine) and
`PTX_ANNEAL_WORKER_PYTHON` (interpreter for the per-candidate benchmark, when it differs).

There is also a **library-level** dependency-injection API — `factory.tune(task, target=, store=,
runner=, search=, logger=)` over the `Store` / `Runner` / `SearchBackend` / `Logger` ABCs, with
`LocalStore`, `LocalRunner`, `CudaPyRunner`, `BaselineSearch` and `NullLogger`/`StderrLogger`
shipped. Use it to embed ptx-anneal in your own orchestration (e.g. a remote store or fleet
runner). Note the CLI does **not** currently go through it — it uses the adapter path above — so
these ABCs are exercised by the test suite rather than by `ptx_anneal tune`.

The `ptxas >= 13.3` floor is fixed and not overridable — see [Requirements](#requirements).

## Install

```bash
pip3 install ptx-anneal   # once published; until then, install from a checkout: pip3 install .
```

## Requirements

- `ptxas` >= 13.3 (from the CUDA Toolkit).
- An optimization engine — NVIDIA CompileIQ — installed separately. Run
  `ptx-anneal doctor` to check your toolchain and print provisioning instructions.
- A frontend that emits ACF tasks and can apply an ACF (e.g. fbtriton).

> **Use one `ptxas` for all three steps.** The 13.3 floor is a hard requirement, not a default:
> `--apply-controls` is GA only from 13.3 on, *and* the frontend derives the PTX ISA version it emits
> from its own `ptxas` version. Collecting under a different `ptxas` therefore produces different PTX,
> a different store key, and a guaranteed MISS at consume — which fails open to the untuned kernel, so
> it looks healthy while delivering nothing.

## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out, and our
[Code of Conduct](CODE_OF_CONDUCT.md).

## License

`ptx-anneal` is MIT licensed, as found in the [LICENSE](LICENSE) file.
