# Skill: Provisioning the ptx-anneal toolchain

Use this when `ptx-anneal` can't run a search, reports a missing/old `ptxas`, or
can't find an optimization engine. Goal: get a working tuning toolchain.

## What ptx-anneal needs

1. **`ptxas` >= 13.3** — required to *assemble* tuning artifacts
   (`ptxas --apply-controls`). Older `ptxas` cannot apply an ACF. From the CUDA
   Toolkit, or `pip install nvidia-cuda-nvcc`; ptx-anneal looks in site-packages
   for the wheel's `ptxas` because it is not on `PATH`.
2. **An optimization engine** (bring-your-own; e.g. NVIDIA CompileIQ) — proposes
   candidate controls. Installed, never vendored:
   `pip install ptx-anneal[compileiq]`. It is an extra rather than a base
   dependency because it narrows the support matrix (Python `>=3.11,<3.14`,
   glibc >= 2.34) while the rest of ptx-anneal installs anywhere.
3. **A search space** — the engine's own concern, not ptx-anneal's. CompileIQ
   fetches its published catalog and caches it under `~/.cache/compileiq/<tag>/`,
   so there is normally nothing to provision. Override with `--ss` (an explicit
   `.bin`) or `CIQ_SEARCH_SPACES_DIR` (offline mirror with `manifest.json`).
4. **A frontend** that emits tuning tasks and can apply an artifact (e.g.
   fbtriton via `TRITON_COMPILE_IQ_COLLECT` / `TRITON_COMPILE_IQ_APPLY`).

## Steps

1. **Check the toolchain:**
   ```bash
   ptx-anneal doctor      # also listed in `ptx-anneal --help`
   ```
   It reports the detected `ptxas` path + version, whether it is new enough, and
   whether an engine is discoverable, then prints the next provisioning step.

2. **If `ptxas` is missing or < 13.3:** install a newer CUDA Toolkit, or
   `pip install nvidia-cuda-nvcc`. Set `TRITON_PTXAS_BLACKWELL_PATH` (or your
   frontend's equivalent) to the `ptxas` you want ptx-anneal to use, and re-run
   `ptx-anneal doctor`. An explicitly pointed-at `ptxas` is used **verbatim**,
   even if it is too old — the whole toolchain must agree on one `ptxas`, so
   ptx-anneal never silently substitutes a different one.

3. **If the engine is missing:** `pip install ptx-anneal[compileiq]` (or
   `pip install compileiq` into the same env). `doctor` prints the public install
   command for the engine it knows about. The engine is always **installed**, never
   downloaded into this repo, and the harness never imports it — it is reached
   through an adapter subprocess, so it may live in a different interpreter
   (`--engine-python`).

   Note `doctor` reports the `ptx_anneal.search_backends` *plugin registry*, so it
   can say "no engine" on a box where `tune` works fine — `tune` reaches the
   engine through the adapter subprocess, which needs no plugin registration.

4. **If the search space can't be fetched** (air-gapped, blocked egress): mirror
   the catalog (`manifest.json` + its `.bin`s) and set `CIQ_SEARCH_SPACES_DIR`,
   or pass a single `--ss <bin>`. Pin `CIQ_SS_TAG` when a run must be
   reproducible; the resolved tag and sha256 are recorded in each ACF's
   `.acf.json` either way.

5. **Verify end to end:** run the smoke flow (see `skills/factory-search.md`).

## Diagnosing

- "cannot apply --apply-controls / Invalid compiler controls file" → version
  mismatch: the artifact was produced for a different `ptxas`. Match versions.
- Engine import/registration errors → the engine plugin isn't installed or its
  `entry_point` isn't registered. Confirm `ptx-anneal doctor` lists it.
- No GPU visible → the search/benchmark harness needs a GPU; `doctor` and the
  store/ABC layer work without one, but `tune` does not.
