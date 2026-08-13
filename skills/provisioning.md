# Skill: Provisioning the ptx-anneal toolchain

Use this when `ptx-anneal` can't run a search, reports a missing/old `ptxas`, or
can't find an optimization engine. Goal: get a working tuning toolchain.

## What ptx-anneal needs

1. **`ptxas` >= 13.3** (from the CUDA Toolkit) — required to *assemble* tuning
   artifacts (`ptxas --apply-controls`). Older `ptxas` cannot apply an ACF.
2. **An optimization engine** (bring-your-own; e.g. NVIDIA CompileIQ) — the
   search engine that proposes candidate controls. Not shipped with ptx-anneal.
3. **A frontend** that emits tuning tasks and can apply an artifact (e.g.
   fbtriton via `TRITON_COMPILE_IQ_COLLECT` / `TRITON_COMPILE_IQ_APPLY`).

## Steps

1. **Check the toolchain:**
   ```bash
   ptx-anneal doctor
   ```
   It reports the detected `ptxas` path + version, whether it is new enough, and
   whether an engine is discoverable, then prints the next provisioning step.

2. **If `ptxas` is missing or < 13.3:** install/point at a newer CUDA Toolkit.
   Set `TRITON_PTXAS_BLACKWELL_PATH` (or your frontend's equivalent) to the
   `ptxas` you want ptx-anneal to use, and re-run `ptx-anneal doctor`.

3. **If the engine is missing:** `doctor` prints the public install command for
   the engine it knows about. The engine is provisioned **separately** and is
   never downloaded into this repo.

4. **Verify end to end:** run the smoke flow (see `skills/factory-search.md`).

## Diagnosing

- "cannot apply --apply-controls / Invalid compiler controls file" → version
  mismatch: the artifact was produced for a different `ptxas`. Match versions.
- Engine import/registration errors → the engine plugin isn't installed or its
  `entry_point` isn't registered. Confirm `ptx-anneal doctor` lists it.
- No GPU visible → the search/benchmark harness needs a GPU; `doctor` and the
  store/ABC layer work without one, but `tune` does not.
