# meta_workload_example_2 — low-headroom kernels from a production recommendation-model training step

This directory holds **14 kernel tasks** where `ptx_anneal` found little or no ptxas tuning headroom: 10 whose best ACF
won less than 3% in the search, and 4 where no candidate beat the untuned baseline. Use it to check that a search
engine or search space does not report false wins, and to study kernels that ptxas already compiles well.

## The workload

- **Model and run:** the same workload as `meta_workload_example_1`, a multi-task ads ranking model trained locally on
  4× B200 GPUs (`sm_100a`) for 200 batches with PyTorch 2 (Inductor) on Meta's Triton fork, ptxas 13.3.
- **Selection:** these 14 come from a tuning run over the 53 largest kernels of that workload (the 51 kernels of 100 µs
  or more standalone, plus the two TLX block-attention kernels). The other 39 kernels won 3% or more.
- **Kernels:** mostly Inductor LayerNorm-backward reductions and pointwise layout/`bmm` copies, plus the TLX
  `_block_attention` forward kernel.

## How the ACFs were produced

- **Engine:** CompileIQ 1.0.2 with the published `ptxas13.3_search_space.bin` catalog.
- **Budget:** pool 32, 20 generations, cull 24, which came to about 580 candidates per kernel.
- **Scoring:** the CUDA-graph benchmark (warmup 25, rep 50). Each candidate's outputs were checked against the untuned
  baseline. Inputs are synthetic except where the task carries captured-input pickles.
- **Admission:** an ACF is admitted only when the re-measured best beats the baseline.

## Results

| Kernel | Task | Tuning baseline (µs) | Best (µs) | Search win | Saved / launch (µs) | Training median (µs) |
|---|---|---|---|---|---|---|
| `triton_poi_fused_add_div_expand_mul_sigmoid_unsqueeze_73` | `29842a9d58ca2c45` | 145.1 | 140.8 | 2.97% | 4.3 | 146.6 |
| `triton_per_fused_addcmul_mul_native_layer_norm_native_layer_norm_backward_53` | `dc4c06ca209d10fe` | 221.7 | 215.3 | 2.92% | 6.5 | 217.6 |
| `triton_red_fused__unsafe_view_clone_native_layer_norm_transpose_view_14` | `c018fdd3865ba4a4` | 218.9 | 213.7 | 2.34% | 5.1 | 216.1 |
| `triton_per_fused_addcmul_mul_native_layer_norm_native_layer_norm_backward_47` | `8e93fc6c4084b3e8` | 222.2 | 217.7 | 2.03% | 4.5 | 216.4 |
| `triton_red_fused__unsafe_view_clone_native_layer_norm_transpose_view_13` | `9715a183c669a743` | 218.6 | 214.2 | 2.01% | 4.4 | 247.7 |
| `_block_attention` | `4bb6c1c96def3eef` | 93.7 | 92.0 | 1.79% | 1.7 | 351.6 |
| `triton_poi_fused__unsafe_view_bmm_clone_permute_squeeze_view_3` | `f0a1f6435b41ef80` | 190.3 | 187.3 | 1.54% | 2.9 | 211.9 |
| `triton_poi_fused__unsafe_view_constant_pad_nd_mul_sigmoid_view_3` | `7e245438704c1442` | 343.0 | 338.3 | 1.38% | 4.7 | 337.0 |
| `triton_poi_fused_bmm_clone_permute_unsqueeze_view_3` | `f5d30d1aa8dae33d` | 201.8 | 199.5 | 1.12% | 2.3 | 191.0 |
| `triton_poi_fused__unsafe_view_constant_pad_nd_mul_sigmoid_view_4` | `496aed07c61c2b7b` | 343.5 | 340.0 | 1.03% | 3.5 | 377.8 |
| `triton_per_fused_mul_native_layer_norm_native_layer_norm_backward_39` | `2002bde47a2f5d98` | 271.6 | - | no win | - | 269.1 |
| `triton_per_fused_addcmul_mul_native_layer_norm_native_layer_norm_backward_22` | `80e35b95cb6901cf` | 221.6 | - | no win | - | 216.9 |
| `triton_per_fused_mul_native_layer_norm_native_layer_norm_backward_43` | `ad0b1cf3d9d3bb71` | 272.1 | - | no win | - | 269.9 |
| `triton_per_fused_mul_native_layer_norm_native_layer_norm_backward_44` | `ec6db5c5c7a7f26b` | 352.7 | - | no win | - | 268.4 |

- **Tuning baseline / Best:** the kernel's standalone time without / with its ACF, measured during the search.
- **Saved / launch:** tuning baseline minus best.
- **Training median:** the kernel's median duration in the training run's Kineto traces (all 4 ranks, steps 101–103),
  matched by kernel name, grid and block.
- **No win:** every candidate compiled and passed validation (about 580 per kernel), but none was faster than the
  baseline, so no ACF is shipped.

Wins this small are close to benchmark noise. In training, Triton re-times each ACF against the plain kernel on its
first launch and keeps the faster; for several of these kernels it keeps the plain one.

## Directory layout

```
meta_workload_example_2/
  <task id = first 16 hex of ir_hash>/
    kernel.ptx                 the frozen PTX to assemble (never recompiled)
    spec.json                  source-free launch spec (entry, arch, shared, block, grid, tensors, args)
    launch_metadata.json       compiler launch metadata (Level 0 schema)
    launcher.c                 the compiler-generated C launcher (Level 1)
    <ir_hash>.13.3.acf         tuned ACF (the 10 tasks with a win only)
    <ir_hash>.13.3.acf.json    its provenance: baseline/best ms, search win, budget, engine, search space
    <pickle_sha256>.pkl        captured input values (_block_attention only)
```

## `_block_attention` needs its captured inputs

`4bb6c1c96def3eef` (`_block_attention`) ships three captured-input pickles: the jagged sequence offsets (`int32[129]`),
the per-CTA work schedule (`int64[149]`) and the per-sequence block offsets (`int32[129]`). They describe only the
batch structure of the captured launch (182,034 tokens in 128 sequences), not feature values. Without them, replay
fills these tensors with zeros, every sequence is empty, and the kernel takes about 4 µs instead of about 94 µs.
Loading them requires a `ptx_anneal` that reads `<pickle_sha256>.pkl` captured inputs; replay then reports
`inputs=captured 3/7`.

Its training median (351.6 µs) is not comparable to its tuning baseline: the kernel is persistent (always 148 CTAs),
so the trace match pools launches of every batch size, and the captured launch is smaller than a typical training batch.
Its cost scales linearly with tokens; on a 646,989-token batch the same kernel takes about 345 µs standalone, within 2%
of the training median.

## Reproducing

```bash
# baseline timing of one task (no ACF)
ptx_anneal replay --task benchmarks/meta_workload_example_2/<task> --ptxas <ptxas 13.3>

# assemble with the shipped ACF
ptxas -arch=sm_100a --apply-controls=<task>/<ir_hash>.13.3.acf <task>/kernel.ptx -o out.cubin
```

Pin `--ptxas` to a **13.3** ptxas: an ACF is bound to the ptxas version that produced it.
