# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Batched candidate scoring, driven by the batch CompileIQ adapter.

The standard adapter starts one ``_score`` process per candidate, and each process re-imports torch,
creates a CUDA context, allocates the inputs, and recompiles and re-runs the untuned kernel before it
times the candidate. Batched scoring keeps everything a candidate does not change alive across a
whole search:

* :func:`compile_batch` assembles a generation's ACFs in parallel, with the scorer's own ptxas flags.
* :func:`serve` is a long-lived benchmark process for one GPU. It builds the inputs, the untuned
  kernel's reference outputs and its timing once, then validates and times each cubin it is sent,
  restoring the inputs before every candidate.

``serve`` speaks JSON lines on stdin/stdout. A candidate that poisons the CUDA context (for example
an illegal address) makes it report ``fatal`` and exit; a hang is caught by the adapter's timeout.
Either way the adapter falls back to per-candidate ``_score`` processes, each with its own context.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor


def compile_batch(
    task_dir: str, ptxas: str, acf_paths: list[str], out_dir: str, *, jobs: int | None = None, timeout=None
) -> list[dict]:
    """Assemble every ACF against the task's PTX, in parallel. One result per ACF, in order:
    ``{"acf", "cubin", "sha256"}`` on success, ``{"acf", "cubin": None, "error"}`` on failure."""
    from . import ptx_launch as L
    from . import task as task_mod

    task = task_mod.load(task_dir)
    os.makedirs(out_dir, exist_ok=True)

    def one(index: int, acf: str) -> dict:
        try:
            cubin = L.ptxas_compile(task.ir, ptxas, arch=task.spec["arch"], acf_path=acf, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"acf": acf, "cubin": None, "error": f"ptxas timed out after {timeout}s"}
        except (RuntimeError, OSError) as e:
            return {"acf": acf, "cubin": None, "error": str(e)[:2000]}
        path = os.path.join(out_dir, f"{index}.cubin")
        with open(path, "wb") as f:
            f.write(cubin)
        return {"acf": acf, "cubin": path, "sha256": hashlib.sha256(cubin).hexdigest()}

    with ThreadPoolExecutor(max_workers=max(1, jobs or min(32, os.cpu_count() or 1))) as pool:
        return list(pool.map(one, range(len(acf_paths)), acf_paths))


def _cuda_usable() -> bool:
    """False once the context is poisoned: after an illegal address every later CUDA call fails."""
    import torch

    try:
        torch.cuda.synchronize()
        torch.ones(1, device="cuda").add_(1).item()
        return True
    except Exception:  # noqa: BLE001 - any failure here means the context cannot be trusted
        return False


def _protocol_stream():
    """The original stdout, reserved for protocol lines. File descriptor 1 is pointed at stderr so a
    stray print from a library cannot corrupt the protocol."""
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    sys.stdout.flush()
    os.dup2(2, 1)
    return proto


def serve(task_dir: str, ptxas: str, *, warmup: int, rep: int, bench: str, stdin=None, stdout=None) -> int:
    """Run the benchmark server for one GPU until stdin closes or an ``exit`` request arrives.

    Requests (one JSON object per line):
        {"op": "score", "id": ..., "cubin": "<path>"}   validate against the untuned outputs, then time
        {"op": "baseline", "id": ...}                    re-time the untuned kernel
        {"op": "exit"}
    Replies echo ``id``: ``{"ok": true, "ms": ...}`` or ``{"ok": false, "reason": ..., "fatal": bool}``.
    The first line is ``{"ready": true, "baseline_ms", "baseline_sha256"}`` once setup has finished, or
    ``{"ready": false, "error"}``.
    """
    proto = stdout or _protocol_stream()
    stdin = stdin or sys.stdin

    def emit(obj: dict) -> None:
        proto.write(json.dumps(obj) + "\n")
        proto.flush()

    try:
        import torch

        from . import ptx_launch as L
        from . import task as task_mod
        from .target.ptx import bench_ms, relative_deviation
        from .validate import ValidationError, load_validator

        task = task_mod.load(task_dir)
        spec = task.spec
        validator = load_validator()
        grid, block = spec["grid"], spec["block"]

        def load(cubin: bytes):
            return L.load_cubin(cubin, spec["entry"], spec["shared"], cluster=spec.get("cluster"))

        baseline_cubin = L.ptxas_compile(task.ir, ptxas, arch=spec["arch"])
        baseline = load(baseline_cubin)
        # One set of inputs for the whole search. Every candidate starts from `pristine` (kernels may
        # update their inputs in place), and the kernel arguments -- pointers, tensormaps, scratch --
        # stay valid because the buffers never move.
        tensors = L.alloc_tensors(spec, seed=0, task_dir=task.dir)
        pristine = [t.clone() for t in tensors]
        tensormaps, addrs = L.build_tensormaps(spec, tensors)
        scratch = L.alloc_global_scratch(spec)
        kernel_args = L.kernel_args_from_spec(spec, tensors, addrs, scratch)

        def restore() -> None:
            for t, p in zip(tensors, pristine, strict=True):
                t.copy_(p)
            torch.cuda.synchronize()

        def run(kernel) -> None:
            restore()
            kernel.launch(grid, block, kernel_args)
            torch.cuda.synchronize()

        def time_kernel(kernel) -> float:
            return float(bench_ms(kernel, spec, kernel_args, bench=bench, warmup=warmup, rep=rep))

        run(baseline)
        reference = [t.clone() for t in tensors]
        baseline_ms = time_kernel(baseline)
    except Exception as e:  # noqa: BLE001 - report setup failures to the adapter instead of crashing
        emit({"ready": False, "error": f"{type(e).__name__}: {e}"})
        return 1
    emit({"ready": True, "baseline_ms": baseline_ms, "baseline_sha256": hashlib.sha256(baseline_cubin).hexdigest()})

    for line in stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        op, rid = request.get("op"), request.get("id")
        if op == "exit":
            break
        if op == "baseline":
            try:
                run(baseline)
                emit({"id": rid, "ok": True, "ms": time_kernel(baseline)})
            except Exception as e:  # noqa: BLE001
                emit({"id": rid, "ok": False, "reason": f"baseline: {type(e).__name__}: {e}", "fatal": True})
                return 3
            continue
        if op != "score":
            emit({"id": rid, "ok": False, "reason": f"unknown op {op!r}", "fatal": False})
            continue
        kernel = None
        try:
            with open(request["cubin"], "rb") as f:
                kernel = load(f.read())
            run(kernel)
            validator.check([relative_deviation(r, t) for r, t in zip(reference, tensors, strict=True)])
            emit({"id": rid, "ok": True, "ms": time_kernel(kernel)})
        except ValidationError as e:
            emit({"id": rid, "ok": False, "reason": f"validation: {e}", "fatal": False})
        except Exception as e:  # noqa: BLE001 - classify by whether the context survived
            fatal = not _cuda_usable()
            emit({"id": rid, "ok": False, "reason": f"{type(e).__name__}: {e}", "fatal": fatal})
            if fatal:
                return 3
        finally:
            if kernel is not None:
                try:
                    kernel.unload()
                except Exception as e:  # noqa: BLE001 - unloading from a failed context can also fail
                    print(f"[_score_server] unload failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
    del tensormaps, scratch  # referenced by kernel_args until here
    return 0
