# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Engine-free CLI scoring, selecting the tensor implementation from the captured spec.

Ordinary contiguous tasks keep the cuda-python/numpy path. BF16, TMA and strided tensors use
the torch-based PTX target, which validates candidates against untuned output before timing.
GPU dependencies are imported only after selecting a scorer.
"""

from . import task as task_mod

_NUMPY_DTYPES = {"float32", "float16", "int32", "int64", "int8"}


def needs_torch(spec: dict) -> bool:
    if spec.get("tensordescs") or any("tma" in arg for arg in spec["args"]):
        return True
    for tensor in spec["tensors"]:
        if tensor["dtype"] not in _NUMPY_DTYPES:
            return True
        shape, strides = tensor["shape"], tensor.get("strides")
        if strides is None:
            continue
        if len(shape) != len(strides):
            return True
        expected = 1
        for size, stride in reversed(list(zip(shape, strides, strict=True))):
            if size > 1 and stride != expected:
                return True
            expected *= size
    return False


def score(task_dir, acf_path, warmup, rep, bench="cudagraph", *, ptxas=None):
    task = task_mod.load(task_dir)
    if needs_torch(task.spec):
        from .target.ptx import PtxTarget

        return PtxTarget().score_one(task, acf_path, warmup=warmup, rep=rep, bench=bench, ptxas=ptxas)
    from .run import cudapy_worker

    return cudapy_worker.score(task_dir, acf_path, warmup, rep, bench, ptxas=ptxas)
