# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Data type conversion utilities between PyTorch, TTNN, and runtime types."""

import torch

try:
    import ttnn
except ModuleNotFoundError:
    ttnn = None

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


def _is_ttnn_tensor(tensor) -> bool:
    """Check if tensor is a ttnn.Tensor or a wrapper around one."""
    if ttnn is None:
        return False
    # Check for direct ttnn.Tensor
    if isinstance(tensor, ttnn.Tensor):
        return True
    # Check for wrapper with underlying ttnn.Tensor
    if hasattr(tensor, "_tensor") and isinstance(tensor._tensor, ttnn.Tensor):
        return True
    return False


def _is_ttnn_dtype(dtype) -> bool:
    """Check if dtype is a ttnn.DataType."""
    if ttnn is None:
        return False
    # Check by string representation since ttnn.DataType comparison can be tricky
    dtype_str = str(type(dtype))
    return "ttnn" in dtype_str and "DataType" in dtype_str


def _ttnn_dtype_to_torch(ttnn_dtype):
    """
    Convert ttnn.DataType to torch dtype.

    Args:
        ttnn_dtype: ttnn.DataType enum value

    Returns:
        Corresponding torch.dtype

    Raises:
        ValueError: If ttnn dtype is not supported
    """
    if ttnn is None:
        raise ImportError("ttnn module not available")

    # ttnn.DataType values - compare by value since enum comparison can be tricky
    dtype_str = str(ttnn_dtype)
    if "BFLOAT16" in dtype_str:
        return torch.bfloat16
    if "FLOAT32" in dtype_str:
        return torch.float32
    if "FLOAT16" in dtype_str:
        return torch.float16
    if "UINT32" in dtype_str:
        return torch.uint32
    if "UINT16" in dtype_str:
        return torch.uint16
    if "UINT8" in dtype_str:
        return torch.uint8
    if "INT32" in dtype_str:
        return torch.int32

    raise ValueError(f"Unsupported ttnn dtype: {ttnn_dtype}")


def get_tensor_shape(tensor):
    """
    Get tensor shape as a list of integers.

    Works with both torch.Tensor and ttnn.Tensor.

    Args:
        tensor: torch.Tensor or ttnn.Tensor

    Returns:
        List of integers representing tensor shape
    """
    if _is_ttnn_tensor(tensor):
        # ttnn.Shape can be converted to list directly or via indexing
        shape = tensor.shape
        # Handle ttnn.Shape - convert to list of ints
        return list(shape)
    else:
        # torch.Tensor
        return list(tensor.shape)


def get_tensor_dtype(tensor):
    """
    Get tensor dtype as torch.dtype.

    Works with both torch.Tensor, ttnn.Tensor, and wrappers.

    Args:
        tensor: torch.Tensor, ttnn.Tensor, or wrapper with .dtype

    Returns:
        torch.dtype for the tensor's element type
    """
    dtype = tensor.dtype
    # Check if it's a ttnn dtype and convert to torch dtype
    if _is_ttnn_dtype(dtype):
        return _ttnn_dtype_to_torch(dtype)
    return dtype


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
