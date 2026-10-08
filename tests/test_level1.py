# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the Level 1 args_t translation core (schema -> ctypes.Structure + value packing; no GPU).

These lock the one Python<->C ABI surface of the Level 1 launcher path: that a ctypes args_t built
from a Level 0 schema matches the C struct the generated launcher expects (field types, natural
alignment) and round-trips runtime values."""

import ctypes

import pytest

from ptx_anneal.level1 import _CUdeviceptr, build_args_struct, field_ctype, pack_args


def _schema(args, entry_name="k"):
    return {"entry_name": entry_name, "args": args}


# --- type mapping (mirror of the nvidia backend _TYPE_TO_C) --------------------------------------
def test_field_ctype_pointer_and_tensordesc_are_device_ptr():
    assert field_ctype("*fp16") is _CUdeviceptr
    assert field_ctype("*i8") is _CUdeviceptr
    assert field_ctype("tensordesc<fp16>") is _CUdeviceptr


def test_field_ctype_scalars():
    assert field_ctype("i32") is ctypes.c_int32
    assert field_ctype("i64") is ctypes.c_int64
    assert field_ctype("fp32") is ctypes.c_float
    assert field_ctype("fp64") is ctypes.c_double
    assert field_ctype("fp16") is ctypes.c_uint16  # half rides a 16-bit slot


def test_field_ctype_nvtmadesc_is_128_bytes():
    ty = field_ctype("nvTmaDesc")
    assert ctypes.sizeof(ty) == 128


def test_field_ctype_unknown_raises():
    with pytest.raises(NotImplementedError):
        field_ctype("fp8e4m3")


# --- struct layout: must match the C compiler's natural alignment for args_t --------------------
def test_build_args_struct_layout_and_offsets():
    # *fp32 (ptr 8B) , i32 (4B) , i64 (8B, 8-aligned) , fp32 (4B) -> classic padding case.
    s = build_args_struct(
        _schema(
            [
                {"name": "p", "type": "*fp32", "index": 0},
                {"name": "a", "type": "i32", "index": 1},
                {"name": "b", "type": "i64", "index": 2},
                {"name": "c", "type": "fp32", "index": 3},
            ]
        )
    )
    assert s.p.offset == 0 and s.p.size == 8
    assert s.a.offset == 8
    assert s.b.offset == 16  # i64 aligned to 8 -> 4 bytes pad after `a`
    assert s.c.offset == 24
    assert ctypes.sizeof(s) == 32  # tail-padded to the 8-byte max alignment


def test_build_args_struct_zero_args_gets_unused_char():
    s = build_args_struct(_schema([]))
    assert [f[0] for f in s._fields_] == ["_unused"]
    assert ctypes.sizeof(s) == 1


def test_build_args_struct_names_from_index_when_missing():
    s = build_args_struct(_schema([{"type": "i32", "index": 0}, {"type": "i32", "index": 1}]))
    assert [f[0] for f in s._fields_] == ["arg0", "arg1"]


def test_build_args_struct_class_name_from_entry():
    s = build_args_struct(_schema([{"name": "p", "type": "*fp32"}], entry_name="my.kernel"))
    assert s.__name__ == "my_kernel_args_t"  # dots -> underscores, matching make_launcher_src


# --- value packing ------------------------------------------------------------------------------
def test_pack_args_round_trips_pointer_and_scalars():
    schema = _schema(
        [
            {"name": "p", "type": "*fp32", "index": 0},
            {"name": "a", "type": "i32", "index": 1},
            {"name": "b", "type": "i64", "index": 2},
            {"name": "c", "type": "fp32", "index": 3},
        ]
    )
    s = build_args_struct(schema)
    obj = pack_args(s, schema, [0xDEADBEEF, 7, 1 << 40, 1.5])
    assert obj.p == 0xDEADBEEF
    assert obj.a == 7
    assert obj.b == 1 << 40
    assert obj.c == pytest.approx(1.5)


def test_pack_args_wrong_count_raises():
    schema = _schema([{"name": "p", "type": "*fp32"}, {"name": "a", "type": "i32"}])
    with pytest.raises(ValueError):
        pack_args(build_args_struct(schema), schema, [123])


def test_pack_args_rejects_by_value_tma():
    schema = _schema([{"name": "d", "type": "nvTmaDesc", "index": 0}])
    with pytest.raises(NotImplementedError):
        pack_args(build_args_struct(schema), schema, [b"\x00" * 128])
