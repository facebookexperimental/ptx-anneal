# Skill: Running a factory search (tuning)

Use this to tune a kernel — drive the engine over `ptxas` controls, validate the
result, and admit the winner to the store.

## Inputs

A **tuning task**: a fixed compiled IR (for the `ptx` target, a `kernel.ptx`) plus
a launch `spec` describing how to assemble and launch it. A frontend produces
these (e.g. fbtriton's collection hook writes one per compiled kernel).

## Run

```bash
ptx-anneal tune <task_dir> --store <store_dir>
```

What happens (PTX-direct — no frontend recompilation in the loop):
1. The engine proposes candidate controls.
2. Each candidate is **assembled** with `ptxas --apply-controls` and the cubin is
   launched via the driver API on real-shaped inputs.
3. Each candidate is **validated** for numerical correctness against the untuned
   (no-artifact) baseline; failing candidates are rejected.
4. The fastest *correct* candidate is **admitted** and its artifact written to the
   content-addressed store under `(target, arch, ir_hash)`.

## Tuning knobs (ptx-anneal's own, `PTX_ANNEAL_*`)

- `PTX_ANNEAL_SEARCH_*` — engine/search depth knobs (generations, pool, per-
  candidate timeout, etc.), passed through to the selected `SearchBackend`.

The minimum ptxas version is **not** a knob: it is fixed at `provision.MIN_PTXAS`
(13.3). Applying an artifact (`ptxas --apply-controls`) is GA only from there on,
and the frontend's ptxas decides the PTX ISA version it emits — so a lower floor
just yields artifacts whose key nothing will ever match.

## Reading results

- "no candidate admitted" → either every candidate failed validation/wedged, or
  none beat the baseline. Check the per-candidate log; widen the search or verify
  the task's spec/inputs.
- A search-time "win" is measured in the search harness and can be noisy; the
  trustworthy decision is made when the artifact is consumed (an A/B vs. the
  plain baseline). Treat search-time numbers as a ranking signal, not a promise.

## Isolation / safety

Each candidate is benchmarked in its own subprocess with a timeout so a candidate
that hangs or crashes the GPU is killed and scored invalid — it never wedges the
driver. Correctness is checked before timing.
