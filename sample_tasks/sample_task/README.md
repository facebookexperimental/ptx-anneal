# sample_task — an example ptx-anneal ACF tuning task

A **task** is the unit of work the factory consumes. It is a directory with exactly two files:

    sample_task/
      kernel.ptx    the fixed PTX to tune (assembled as-is; never recompiled)
      spec.json     the source-free launch description (entry/arch/shared/block/grid/tensors/args/ir_hash)

This sample is a naive fp32 vector-add (`add_kernel`). See `ptx_anneal/task.py` for the full
`spec.json` schema; the store key is `(target, arch, ir_hash)` where `ir_hash = sha256(normalize_ptx(ptx))`,
so one tuned ACF transfers across runtime shapes of the same kernel.

## How a task is generated

A frontend compiles a kernel once, then captures its PTX + a launch spec (it never dumps source).
A task is produced by: compile via Triton → take `h.asm["ptx"]` and `h.metadata` → `build_spec(...)`
(which applies the frontend's `constexpr`/`equal_to_1` specialization and appends the two trailing
null scratch params) → set `ir_hash`. Point `TRITON_PTXAS_BLACKWELL_PATH` at the SAME ptxas the
factory will tune with (≥ 13.3 to assemble/apply ACFs) so the PTX ISA matches.

Complete example emitter (a Triton matmul; adapt the kernel/args for your own task):

```python
# Usage: python emit.py --ptxas /path/to/ptxas --work-dir <DIR> [--shape M N K] [--check]
import argparse, json, os
import torch, triton
import triton.language as tl
from ptx_anneal import ptx_launch as L
from ptx_anneal.target.ptx import ptx_ir_hash


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def compile_and_spec(M, N, K, ptxas, dtype=torch.float16, device="cuda"):
    BLOCK_M = BLOCK_N = 64
    BLOCK_K = 32
    torch.manual_seed(0)
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)
    c = torch.empty((M, N), device=device, dtype=dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    h = matmul_kernel[grid](
        a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    torch.cuda.synchronize()
    ptx = h.asm["ptx"]
    ds = str(dtype).replace("torch.", "")
    ordered = [
        ("tensor", {"shape": [M, K], "dtype": ds, "strides": list(a.stride())}),
        ("tensor", {"shape": [K, N], "dtype": ds, "strides": list(b.stride())}),
        ("tensor", {"shape": [M, N], "dtype": ds, "strides": list(c.stride())}),
        ("scalar", M), ("scalar", N), ("scalar", K),
        ("scalar", a.stride(0)), ("scalar", a.stride(1)),
        ("scalar", b.stride(0)), ("scalar", b.stride(1)),
        ("scalar", c.stride(0)), ("scalar", c.stride(1)),
        ("constexpr",), ("constexpr",), ("constexpr",),  # BLOCK_M, BLOCK_N, BLOCK_K
    ]
    spec = L.build_spec(ptx, h.metadata, tuple(grid), ordered, ptxas)
    spec["ir_hash"] = ptx_ir_hash(ptx)  # store key (sha256 of normalized PTX)
    return ptx, spec, a, b, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ptxas", default=os.environ.get("TRITON_PTXAS_BLACKWELL_PATH", "ptxas"))
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--shape", type=int, nargs=3, default=[256, 256, 256], metavar=("M", "N", "K"))
    ap.add_argument("--check", action="store_true", help="verify baseline driver-launch == torch")
    a = ap.parse_args()
    M, N, K = a.shape
    os.makedirs(a.work_dir, exist_ok=True)
    ptx, spec, ta, tb, tc = compile_and_spec(M, N, K, a.ptxas)
    with open(os.path.join(a.work_dir, "kernel.ptx"), "w") as f:
        f.write(ptx)
    with open(os.path.join(a.work_dir, "spec.json"), "w") as f:
        json.dump(spec, f, indent=2)
    if a.check:
        cubin = L.ptxas_compile(ptx, a.ptxas, arch=spec["arch"])
        k = L.load_cubin(cubin, spec["entry"], spec["shared"])
        out = torch.empty((M, N), device="cuda", dtype=torch.float16)
        ka = L.kernel_args_from_spec(spec, [ta, tb, out])
        k.launch(spec["grid"], spec["block"], ka)
        torch.cuda.synchronize()
        ref = torch.matmul(ta.float(), tb.float())
        rel = (out.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-9)
        assert rel < 1e-2, "baseline driver launch INCORRECT"


if __name__ == "__main__":
    main()
```
