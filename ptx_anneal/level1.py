# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Level 1 launcher adoption (PROTOTYPE): drive the compiler-generated launcher
(``asm["launcher_src"]`` + ``nvidia/backend/launch.h``) from a captured Level 0 schema, instead of
the hand-rolled PTX-direct launcher in ptx_launch.py.

The generated ``triton_launch_<kernel>()`` takes a typed ``<kernel>_args_t`` struct (the kernel args
in order -- pointers as ``CUdeviceptr``, scalars as their C types) and does everything else itself:
param packing, TMA ``CUtensorMap`` construction (incl. auto-TMA recipes), cluster attrs, and the
``cuLaunchKernelEx`` call, all via launch.h. So the ONLY Python<->C ABI surface ptx-anneal needs is
mapping each schema arg type to a ctypes field type -- and a ctypes.Structure follows the same
platform ABI as the C compiler, so the layout matches ``args_t`` field-for-field.

This eliminates the arg-list duplication build_spec/ _build_kernel_params carry (PTX parse,
specialization, TMA ABI, param packing) -- the compiler's own launcher owns all of it. Torch-free
(pure ctypes/stdlib), matching the P0 constraint.

Scope (prototype): the args_t translation + value packing (below). Compiling launcher.c against
launch.h and dlopen/launching triton_launch_<kernel>() is a GPU/toolchain follow-up.
"""

import ctypes

# Mirror of the nvidia backend's make_launcher_src `_TYPE_TO_C` (compiler.py), but Triton type ->
# ctypes. Kept in sync with that map: it is the single source of truth for the args_t field types.
_SCALAR_CTYPES = {
    "i1": ctypes.c_int8,
    "i8": ctypes.c_int8,
    "i16": ctypes.c_int16,
    "i32": ctypes.c_int32,
    "i64": ctypes.c_int64,
    "u1": ctypes.c_uint8,
    "u8": ctypes.c_uint8,
    "u16": ctypes.c_uint16,
    "u32": ctypes.c_uint32,
    "u64": ctypes.c_uint64,
    "fp16": ctypes.c_uint16,  # half carried in a 16-bit slot
    "bf16": ctypes.c_uint16,
    "fp32": ctypes.c_float,
    "f32": ctypes.c_float,
    "fp64": ctypes.c_double,
}

# CUdeviceptr is an unsigned 64-bit device address handle on LP64; a by-value CUtensorMap is 128B.
_CUdeviceptr = ctypes.c_uint64
_CUtensorMap = ctypes.c_byte * 128


def field_ctype(triton_ty: str):
    """Map a schema arg ``type`` string to the ctypes type of its ``args_t`` field, mirroring
    make_launcher_src._c_type (Triton type -> C type): pointers and host tensordescs are passed as a
    device pointer, ``nvTmaDesc`` is a by-value 128B tensormap, scalars use _SCALAR_CTYPES."""
    if triton_ty.startswith("*"):
        return _CUdeviceptr
    if triton_ty.startswith("tensordesc"):
        return _CUdeviceptr  # host-side descriptor: passed as its base pointer
    if triton_ty == "nvTmaDesc":
        return _CUtensorMap
    c = _SCALAR_CTYPES.get(triton_ty)
    if c is None:
        raise NotImplementedError(f"unmapped Triton type {triton_ty!r} (add it to _SCALAR_CTYPES)")
    return c


def build_args_struct(schema: dict):
    """Return a ``ctypes.Structure`` subclass mirroring the generated ``<kernel>_args_t``: one field
    per schema arg (constants already excluded), in order, at C-natural alignment. A zero-arg kernel
    gets a single ``char _unused`` field, matching the generated struct (an empty struct is a GCC
    extension, not valid C)."""
    args = schema.get("args") or []
    fields = []
    for i, a in enumerate(args):
        name = a.get("name") or f"arg{i}"
        fields.append((name, field_ctype(str(a.get("type", "")))))
    if not fields:
        fields = [("_unused", ctypes.c_char)]
    name = (schema.get("entry_name") or "kernel").replace(".", "_")
    return type(f"{name}_args_t", (ctypes.Structure,), {"_fields_": fields})


def pack_args(struct_cls, schema: dict, values):
    """Fill a fresh ``args_t`` instance from ordered runtime ``values`` (aligned 1:1 with
    ``schema["args"]`` in post-specialization order): an int device address for pointer / host
    tensordesc args, or a python int/float for scalars. Returns the populated instance.

    By-value ``nvTmaDesc`` (128B) fields are not supported here -- those belong to the host-side
    TensorDescriptor path, a follow-up; auto-TMA needs only pointer + scalar fields (the launcher
    builds the CUtensorMap from them)."""
    args = schema.get("args") or []
    if len(values) != len(args):
        raise ValueError(f"{len(values)} values != {len(args)} schema args")
    obj = struct_cls()
    for a, v in zip(args, values, strict=True):
        name = a.get("name")
        if field_ctype(str(a.get("type", ""))) is _CUtensorMap:
            raise NotImplementedError(f"by-value nvTmaDesc arg {name!r} not supported (host-TMA follow-up)")
        setattr(obj, name, v)
    return obj
