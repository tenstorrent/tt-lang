# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Data type conversion utilities between PyTorch and runtime types."""

import torch

try:
    from _ttmlir_runtime import runtime
except ModuleNotFoundError:
    runtime = None

from ttmlir import ir
from ttmlir.dialects import ttcore


def to_data_type(dtype):
    """
    Convert PyTorch dtype to runtime DataType.

    Args:
        dtype: PyTorch dtype (torch.float32, torch.int32, etc.)

    Returns:
        runtime.DataType enum value

    Raises:
        ValueError: If dtype has no runtime equivalent
    """
    if dtype == torch.float32:
        return runtime.DataType.Float32
    if dtype == torch.float16:
        return runtime.DataType.Float16
    if dtype == torch.bfloat16:
        return runtime.DataType.BFloat16
    if dtype == torch.uint32:
        return runtime.DataType.UInt32
    if dtype == torch.uint16:
        return runtime.DataType.UInt16
    if dtype == torch.uint8:
        return runtime.DataType.UInt8
    if dtype == torch.int32:
        return runtime.DataType.Int32
    if dtype == torch.float64:
        return runtime.DataType.Float64
    if dtype == torch.int64:
        return runtime.DataType.Int64
    if dtype == torch.uint64:
        return runtime.DataType.UInt64
    if dtype == torch.int16:
        return runtime.DataType.Int16
    if dtype == torch.int8:
        return runtime.DataType.Int8
    if dtype == torch.bool:
        return runtime.DataType.Bool
    raise ValueError(f"Torch dtype: {dtype} has no runtime DataType equivalent")


def from_data_type(dtype):
    """
    Convert runtime DataType string to PyTorch dtype.

    Args:
        dtype: String representation of runtime DataType ("Float32", "Int32", etc.)

    Returns:
        PyTorch dtype

    Raises:
        ValueError: If dtype string is not supported
    """
    if dtype == "Float32":
        return torch.float32
    if dtype == "Float16":
        return torch.float16
    if dtype == "BFloat16":
        return torch.bfloat16
    if dtype == "UInt32":
        return torch.uint32
    if dtype == "UInt16":
        return torch.uint16
    if dtype == "UInt8":
        return torch.uint8
    if dtype == "Int32":
        return torch.int32
    if dtype == "Float64":
        return torch.float64
    if dtype == "Int64":
        return torch.int64
    if dtype == "UInt64":
        return torch.uint64
    if dtype == "Int16":
        return torch.int16
    if dtype == "Int8":
        return torch.int8
    if dtype == "Bool":
        return torch.bool
    raise ValueError(f"Unsupported dtype: {dtype}")


def torch_dtype_to_mlir_type(torch_dtype, ctx):
    """
    Convert PyTorch dtype to MLIR type.

    Args:
        torch_dtype: PyTorch dtype (torch.float32, torch.int32, etc.)
        ctx: MLIR context

    Returns:
        MLIR Type object

    Raises:
        ValueError: If dtype is not supported
    """
    if torch_dtype == torch.float32:
        return ir.F32Type.get(ctx)
    if torch_dtype == torch.float16:
        return ir.F16Type.get(ctx)
    if torch_dtype == torch.bfloat16:
        return ir.BF16Type.get(ctx)
    if torch_dtype == torch.float64:
        return ir.F64Type.get(ctx)
    if torch_dtype == torch.int32:
        return ir.IntegerType.get_signless(32, ctx)
    if torch_dtype == torch.int16:
        return ir.IntegerType.get_signless(16, ctx)
    if torch_dtype == torch.int8:
        return ir.IntegerType.get_signless(8, ctx)
    if torch_dtype == torch.int64:
        return ir.IntegerType.get_signless(64, ctx)
    if torch_dtype == torch.uint32:
        return ir.IntegerType.get_unsigned(32, ctx)
    if torch_dtype == torch.uint16:
        return ir.IntegerType.get_unsigned(16, ctx)
    if torch_dtype == torch.uint8:
        return ir.IntegerType.get_unsigned(8, ctx)
    if torch_dtype == torch.uint64:
        return ir.IntegerType.get_unsigned(64, ctx)
    if torch_dtype == torch.bool:
        return ir.IntegerType.get_signless(1, ctx)

    raise ValueError(f"Unsupported torch dtype: {torch_dtype}")


def torch_dtype_to_ttcore_datatype(torch_dtype):
    """
    Convert PyTorch dtype to ttcore.DataType enum.

    Args:
        torch_dtype: PyTorch dtype (torch.float32, torch.int32, etc.)

    Returns:
        ttcore.DataType enum value

    Raises:
        ValueError: If dtype is not supported
    """
    if torch_dtype == torch.float32:
        return ttcore.DataType.Float32
    if torch_dtype == torch.float16:
        return ttcore.DataType.Float16
    if torch_dtype == torch.bfloat16:
        return ttcore.DataType.BFloat16
    if torch_dtype == torch.int32:
        return ttcore.DataType.Int32
    if torch_dtype == torch.uint32:
        return ttcore.DataType.UInt32
    if torch_dtype == torch.uint16:
        return ttcore.DataType.UInt16
    if torch_dtype == torch.uint8:
        return ttcore.DataType.UInt8
    if torch_dtype == torch.bool:
        return ttcore.DataType.Bool

    raise ValueError(f"Unsupported torch dtype for ttcore.DataType: {torch_dtype}")
