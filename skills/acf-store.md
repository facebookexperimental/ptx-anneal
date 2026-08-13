# Skill: Inspecting and managing the tuning-artifact store

Use this to find, inspect, or clear cached tuning artifacts (ACFs) — e.g. when a
consume run unexpectedly misses, or after re-tuning.

## Layout

The store is content-addressed by `(target, arch, ir_hash)`. For the `ptx`
target the `ir_hash` is `sha256` of the normalized PTX (line/debug info stripped
so cosmetic differences don't change the key). A local store is a directory tree:

```
<store>/<arch>/<ir_hash>.acf            # the tuning artifact (controls)
<store>/<arch>/<ir_hash>.acf.json       # provenance sidecar (engine, version, win)
<store>/<arch>/<ir_hash>.<ptxas>.acf.cubin   # optional: assembled cubin cache
```

## Commands

```bash
ptx-anneal store ls  --store <store_dir>          # list cached artifacts
ptx-anneal store get --store <store_dir> <ir_hash>  # show one artifact's provenance
```

## Common questions

- **Why did consume miss?** The consumer computes the key from the *post-compile*
  IR. If the kernel changed (different PTX → different `ir_hash`) the key won't
  match. Confirm the `ir_hash` of the current kernel equals a stored one.
- **Version mismatch.** An artifact is only valid for the `ptxas` that produced it,
  so both the `.acf` and the assembled-cubin cache are tagged by ptxas version. Note
  the frontend's ptxas also decides the PTX ISA version it emits, so a version
  mismatch usually shows up one step earlier, as an `ir_hash` that never matches.
- **Re-tuning / staleness.** Writing a new `.acf` for the same key should
  invalidate any derived `.acf.cubin` for that key, or later runs may serve the
  stale assembly. When in doubt, delete the derived `.acf.cubin` files.

## Keying note

`ir_hash` subsumes autotune-config and dtype/alignment specialization (they
change the IR) but **not** runtime shape (M, N, K) when the kernel is
shape-agnostic — so one artifact can transfer across shapes of the same kernel.
