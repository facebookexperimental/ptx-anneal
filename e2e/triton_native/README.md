# The triton-native channel

An **optional, on-demand** runway for scoring a kernel under `ptxas --apply-controls`
*through [fbtriton](https://github.com/facebookexperimental/triton) itself* — no PTX capture, no
launch spec, no hand-rolled tensormaps.

> **The frontend is fbtriton, not Triton in general.** `PTXAS_OPTIONS` is fbtriton's knob (it backs
> `CUDAOptions.ptx_options`); upstream Triton has no equivalent — it spells a different knob,
> `TRITON_PTXAS_OPTIONS`. On the wrong Triton the ACF is silently ignored, every candidate compiles
> **untuned**, and the search reports wins it never made. So the runway checks for fbtriton up front
> (tlx/ and magnon/ on disk, no import needed) and the scorer *verifies* per candidate that
> `CUDAOptions.ptx_options` really is our `--apply-controls=<acf>` before measuring anything.

```bash
python -m e2e.triton_native.runway                       # the canary only
python -m e2e.triton_native.runway --kernel ws_gemm      # canary, then ws_gemm
MODE=native ../e2e.sh ws_gemm                            # same, through e2e.sh
```

## Why it exists

`ptx-anneal` proper captures a kernel's PTX and relaunches it through the CUDA driver. That is the
channel that scales, and it is the one we ship. But it means a red result has two possible causes:
the kernel really does break under an applied ACF, or our capture-and-relaunch is wrong.

This channel eliminates the second possibility. It sets

```
PTXAS_OPTIONS=--apply-controls=<acf>
TRITON_ALWAYS_COMPILE=1
```

and lets fbtriton compile and launch the kernel exactly as it normally would. **Its product is a
faster verdict, not a faster kernel** — its success metric is time-to-answer.

| triton-native | PTX-direct | conclusion |
|---|---|---|
| fail | — | a **CompileIQ** issue — reproducible with zero ptx-anneal code, so reportable upstream |
| pass | fail | a **ptx-anneal** issue: a wrong PTX repro, or a consumption bug |
| pass | pass, numbers disagree | a **ptx-anneal** issue, the dangerous one — it looks green |
| pass | pass, numbers agree | trustworthy |

**"Pass" means ≥ 1 valid candidate**, not a win. Win size is the PTX-direct channel's question.

## The fingerprint

That table is only usable if both channels posed the **same search problem**. So every run of either
channel ends by printing a fingerprint covering kernel identity, ptxas path/version/**flags**, the
**resolved** search space, the **frontend** (which fbtriton, and the verified ACF mechanism), engine
and budget, the objective (metric, timing method *and budget*, correctness oracle *and tolerance*)
and the environment (GPU, CUDA, driver, clock state).

The frontend field is why `frontend: {}` on the PTX-direct side is meaningful rather than missing:
that channel replays a frozen `kernel.ptx`, so no Triton is an input at tune time.

```bash
python -m e2e.triton_native.runway --kernel k  > native.log
ptx_anneal --task ../../sample_tasks/<k>       > direct.log
diff <(sed -n '/fingerprint/,$p' native.log) <(sed -n '/fingerprint/,$p' direct.log)
```

Equal digests mean the comparison is meaningful. Nothing else does. Execution mechanism and timing
method are pluggable *knobs*, not a fork — the channels may differ, as long as the fingerprint says
so.

## Conformance with NVIDIA's documented example

Defaults are taken from CompileIQ's `examples/compilers/triton_example/triton_ptx.py`, because
"we followed your doc" is what makes an upstream bug report actionable rather than deflectable.

| knob | default | flag |
|---|---|---|
| search space | `PtxasSearchSpace(version=<local ptxas version>)` | `--ss` / `--ss-tag` / `--ss-variant` |
| correctness | `torch.allclose(atol=1e-2, rtol=0)` | `--oracle relerr` uses the spec's `rtol` |
| timing | `do_bench(warmup=100, rep=1000, return_mode="mean")` — **milliseconds, not iterations** | `--warmup/--rep` |
| budget | `generations=5`, `pool_size=32` (must be > 5) | `--generations/--pool` |
| per-candidate timeout | **120 s**, not the doc's 20 s — see below | `--task-timeout` |
| clocks | `gpu_benchmark_mode(clock_mhz=1965)` — B200 max SM clock; the doc calls locking crucial | `--clock-mhz 0` to leave alone |

Known, deliberate deviations — all visible in the fingerprint:

- **The config is pinned, not autotuned.** See below.
- **The ACF arrives via `PTXAS_OPTIONS`**, not the doc's per-launch
  `ptx_options=f"--apply-controls={acf}"`. Same knob underneath (`PTXAS_OPTIONS` backs
  `CUDAOptions.ptx_options`); using the env form means a pasted kernel needs no edit.
- **Large-K fp16 GEMMs need `--oracle relerr`.** The doc's `atol=1e-2` is calibrated for its
  512×512×512 example; a K=1152 fp16 accumulation will not meet it for reasons that have nothing to
  do with the ACF.
- **`task_timeout` is 120 s, not the doc's 20 s.** The doc runs its objective in-process; we cannot,
  because `PTXAS_OPTIONS` must be set before triton is imported. Every candidate is therefore a fresh
  process that spends ~8 s importing torch+fbtriton before it can measure anything, and at 20 s a
  slow compile times out for no interesting reason.

## Swapping the search space

The catalog is a tuning **input**, so it is selectable and always recorded in the fingerprint
(`resolved_tag`, `sha256`, `source`) — two runs on different catalogs can never be mistaken for the
same experiment.

```bash
--ss /path/ptxas13.3_search_space.bin     # an explicit .bin; wins over every selector
--ss-tag search-spaces-2026.09.01         # a published catalog tag
--ss-variant att                          # a non-default variant
```

Precedence is flag → environment (`PTX_ANNEAL_SS`, `CIQ_SS_TAG`/`CIQ_SS_VERSION`/`CIQ_SS_VARIANT`) →
the doc default (version derived from the local ptxas). Downloaded catalogs live in
`~/.cache/compileiq/<tag>/` as a `manifest.json` plus `<sha12>_ptxas<ver>_search_space.bin`; point
`CIQ_SEARCH_SPACES_DIR` at a mirror to work offline. A missing `--ss` is a hard error, never a silent
fallback to a different catalog.

**Match the catalog to your ptxas.** The bin is keyed by compiler version
(`ptxas13.3_search_space.bin`); pairing a 13.3 catalog with a different ptxas is not a supported
combination, and the fingerprint records both so a mismatch is visible after the fact.

## Adding a kernel

Drop a file in `../kernels/` that calls `register(KernelSpec(...))`. Four fields:

```python
register(KernelSpec(
    name="my_kernel",
    make_inputs=lambda: (torch.randn(...),),   # () -> args
    run_fn=my_kernel_wrapper,                  # (*inputs) -> torch.Tensor
    ref_fn=lambda x: torch.some_reference(x),  # (*inputs) -> torch.Tensor, or None
    rtol=5e-2,
))
```

Optionally set a module-level `FINGERPRINT_CONFIG = {...}` naming the pinned config; it goes into
the fingerprint's kernel identity. Absent, it is recorded as `null` rather than guessed.

### `ref_fn` is the hard field

`ref_fn=None` is allowed. The runway then compares each candidate against the kernel's **own no-ACF
output**, and labels the verdict:

```
triton-native: WEAKER: no ref_fn; correctness is vs the kernel's own no-ACF output (self-referential)
```

That check catches a miscompile — an ACF changing the math — but not a kernel that was already
wrong. It is a fine deliberate fallback and a dangerous silent default, so it is never silent.

### Autotune is not supported yet (TODO)

An `@triton.autotune`'d kernel is **refused** with instructions. Two independent reasons:

1. Autotune picks a config at runtime, so "the kernel being tuned" is undefined and the
   fingerprint's pinned config would be a fiction.
2. Autotune re-benchmarks every config under every candidate ACF, so the score can compare two
   different kernels.

Every TLX tutorial ships `@triton.autotune`, so a pasted kernel hits this by default. Pin one config
— replace the decorator with explicit constexpr args plus `num_warps`/`num_stages`. `../kernels/doc_matmul.py`
is a worked example of exactly that transformation.

## The canary

`doc_matmul` is NVIDIA's doc-example matmul, transcribed with its config pinned: plain Triton, no
TMA and no warp specialization (it does use tcgen5 — on sm_100 `tl.dot` lowers to it). Recorded as
scoring 6/6 valid with an ~11.5% win on CUDA 13.3. The runway runs it
**first by default** (`--no-canary` to skip) and stops if it fails, because a toolchain that cannot
tune the known-good example makes every later result uninterpretable.

## Portability

Pure stdlib in `runway.py`: torch, fbtriton and the engine are *probed*, never imported. Without them
the runway prints what is missing and exits 2, and the rest of the repo is unaffected — the same way
`e2e.sh` degrades `MODE=full` to `MODE=tune` when its site hook is absent.

`score.py` is the only module that imports torch/fbtriton, and only after the ACF environment is
set — `PTXAS_OPTIONS` is read when the nvidia backend is imported, so it is one candidate per
process, necessarily.

## Not packaged

This lives under `e2e/`, which is excluded from the wheel (`packages = ["ptx_anneal"]`) and from the
sdist (`exclude = ["e2e", "e2e/**"]`). It adds no dependency to the shipped wheel.
