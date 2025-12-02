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


# Torch dtype to runtime DataType integer mapping (for C API compatibility)
# TODO: Replace with runtime.DataType enum once C API is updated to accept enum values
# These integer values correspond to:
#   0 = Float32, 1 = Float16, 2 = BFloat16
TORCH_TO_RUNTIME_DTYPE_INT = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}


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
    match dtype:
        case torch.float32:
            return runtime.DataType.Float32
        case torch.float16:
            return runtime.DataType.Float16
        case torch.bfloat16:
            return runtime.DataType.BFloat16
        case torch.uint32:
            return runtime.DataType.UInt32
        case torch.uint16:
            return runtime.DataType.UInt16
        case torch.uint8:
            return runtime.DataType.UInt8
        case torch.int32:
            return runtime.DataType.Int32
        case torch.float64:
            return runtime.DataType.Float64
        case torch.int64:
            return runtime.DataType.Int64
        case torch.uint64:
            return runtime.DataType.UInt64
        case torch.int16:
            return runtime.DataType.Int16
        case torch.int8:
            return runtime.DataType.Int8
        case torch.bool:
            return runtime.DataType.Bool
        case _:
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
    match dtype:
        case "Float32":
            return torch.float32
        case "Float16":
            return torch.float16
        case "BFloat16":
            return torch.bfloat16
        case "UInt32":
            return torch.uint32
        case "UInt16":
            return torch.uint16
        case "UInt8":
            return torch.uint8
        case "Int32":
            return torch.int32
        case "Float64":
            return torch.float64
        case "Int64":
            return torch.int64
        case "UInt64":
            return torch.uint64
        case "Int16":
            return torch.int16
        case "Int8":
            return torch.int8
        case "Bool":
            return torch.bool
        case _:
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


def torch_dtype_to_ttnn_datatype(torch_dtype):
    """
    Convert PyTorch dtype to ttnn.DataType enum.

    Args:
        torch_dtype: PyTorch dtype (torch.float32, torch.bfloat16, etc.)

    Returns:
        ttnn.DataType enum value

    Raises:
        ImportError: If ttnn is not available
        ValueError: If dtype is not supported
    """
    try:
        import ttnn
    except ModuleNotFoundError:
        raise ImportError("ttnn module not available")

    match torch_dtype:
        case torch.float32:
            return ttnn.DataType.FLOAT32
        case torch.float16:
            return ttnn.DataType.FLOAT16
        case torch.bfloat16:
            return ttnn.DataType.BFLOAT16
        case torch.int32:
            return ttnn.DataType.INT32
        case torch.uint32:
            return ttnn.DataType.UINT32
        case torch.uint16:
            return ttnn.DataType.UINT16
        case _:
            raise ValueError(f"Unsupported torch dtype for ttnn.DataType: {torch_dtype}")


def create_borrowed_tensors(torch_tensors):
    """
    Create runtime borrowed tensors from torch tensors.

    Borrowed tensors share memory with the original torch tensors, enabling zero-copy I/O.

    Args:
        torch_tensors: List of torch.Tensor objects

    Returns:
        List of runtime.Tensor objects sharing memory with inputs

    Raises:
        ImportError: If runtime module is not available
    """
    if runtime is None:
        raise ImportError("Runtime module not available")

    result = []
    for tensor in torch_tensors:
        dtype_value = TORCH_TO_RUNTIME_DTYPE_INT.get(tensor.dtype, 0)
        rt_tensor = runtime.create_borrowed_host_tensor(
            tensor.data_ptr(),
            list(tensor.shape),
            list(tensor.stride()),
            tensor.element_size(),
            dtype_value,
        )
        result.append(rt_tensor)
    return result
