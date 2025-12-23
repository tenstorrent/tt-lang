# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Data type conversion utilities between PyTorch and runtime types."""

import torch

try:
    from _ttmlir_runtime import runtime
except ModuleNotFoundError:
    runtime = None

try:
    import ttnn
except ModuleNotFoundError:
    ttnn = None


def is_ttnn_tensor(tensor) -> bool:
    """Check if tensor is a ttnn.Tensor."""
    if ttnn is None:
        return False
    return isinstance(tensor, ttnn.Tensor)


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


def ttnn_dtype_to_mlir_type(ttnn_dtype, ctx):
    """
    Convert ttnn.DataType to MLIR type.

    Args:
        ttnn_dtype: ttnn.DataType enum value
        ctx: MLIR context

    Returns:
        MLIR Type object

    Raises:
        ValueError: If dtype is not supported
    """
    try:
        import ttnn
    except ModuleNotFoundError:
        raise ImportError("ttnn module not available")

    match ttnn_dtype:
        case ttnn.DataType.FLOAT32:
            return ir.F32Type.get(ctx)
        case ttnn.DataType.BFLOAT16:
            return ir.BF16Type.get(ctx)
        case ttnn.DataType.BFLOAT8_B:
            return ir.BF16Type.get(ctx)  # Approximate as BF16
        case ttnn.DataType.BFLOAT4_B:
            return ir.BF16Type.get(ctx)  # Approximate as BF16
        case ttnn.DataType.INT32:
            return ir.IntegerType.get_signless(32, ctx)
        case ttnn.DataType.UINT32:
            return ir.IntegerType.get_unsigned(32, ctx)
        case ttnn.DataType.UINT16:
            return ir.IntegerType.get_unsigned(16, ctx)
        case ttnn.DataType.UINT8:
            return ir.IntegerType.get_unsigned(8, ctx)
        case _:
            raise ValueError(f"Unsupported ttnn dtype: {ttnn_dtype}")


def tensor_dtype_to_mlir_type(dtype, ctx):
    """
    Convert tensor dtype to MLIR type, supporting both torch and ttnn dtypes.

    Args:
        dtype: Either torch dtype or ttnn.DataType
        ctx: MLIR context

    Returns:
        MLIR Type object
    """
    # Check if it's a ttnn DataType by checking for the enum name pattern
    dtype_str = str(dtype)
    if "DataType." in dtype_str:
        return ttnn_dtype_to_mlir_type(dtype, ctx)
    else:
        return torch_dtype_to_mlir_type(dtype, ctx)


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


def ttnn_dtype_to_ttcore_datatype(ttnn_dtype):
    """
    Convert ttnn.DataType to ttcore.DataType enum.

    Args:
        ttnn_dtype: ttnn.DataType enum value

    Returns:
        ttcore.DataType enum value

    Raises:
        ValueError: If dtype is not supported
    """
    try:
        import ttnn
    except ModuleNotFoundError:
        raise ImportError("ttnn module not available")

    match ttnn_dtype:
        case ttnn.DataType.FLOAT32:
            return ttcore.DataType.Float32
        case ttnn.DataType.BFLOAT16:
            return ttcore.DataType.BFloat16
        case ttnn.DataType.BFLOAT8_B:
            return ttcore.DataType.BFloat16  # Approximate
        case ttnn.DataType.BFLOAT4_B:
            return ttcore.DataType.BFloat16  # Approximate
        case ttnn.DataType.INT32:
            return ttcore.DataType.Int32
        case ttnn.DataType.UINT32:
            return ttcore.DataType.UInt32
        case ttnn.DataType.UINT16:
            return ttcore.DataType.UInt16
        case ttnn.DataType.UINT8:
            return ttcore.DataType.UInt8
        case _:
            raise ValueError(
                f"Unsupported ttnn dtype for ttcore.DataType: {ttnn_dtype}"
            )


def tensor_dtype_to_ttcore_datatype(dtype):
    """
    Convert tensor dtype to ttcore.DataType, supporting both torch and ttnn dtypes.

    Args:
        dtype: Either torch dtype or ttnn.DataType

    Returns:
        ttcore.DataType enum value
    """
    dtype_str = str(dtype)
    if "DataType." in dtype_str:
        return ttnn_dtype_to_ttcore_datatype(dtype)
    else:
        return torch_dtype_to_ttcore_datatype(dtype)


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
        case torch.bfloat16:
            return ttnn.DataType.BFLOAT16
        case torch.int32:
            return ttnn.DataType.INT32
        case torch.uint32:
            return ttnn.DataType.UINT32
        case torch.uint16:
            return ttnn.DataType.UINT16
        case _:
            raise ValueError(
                f"Unsupported torch dtype for ttnn.DataType: {torch_dtype}"
            )


def tile_bytes_from_dtype(dtype) -> int:
    """
    Calculate tile size in bytes from ttnn dtype.

    For tiled tensors, each tile is 32x32 elements. The byte size depends on
    the data type's element size plus any format-specific overhead.

    Args:
        dtype: ttnn.DataType enum value

    Returns:
        Tile size in bytes

    Raises:
        ValueError: If dtype is not supported
    """
    dtype_int = int(dtype)
    # Map ttnn DataType enum values to tile sizes
    # Reference: tt-metal/tt_metal/common/constants.hpp
    if dtype_int in (0, 6):  # BFloat16, UInt16
        return 32 * 32 * 2  # 2048
    elif dtype_int in (1, 2, 7):  # Float32, Int32, UInt32
        return 32 * 32 * 4  # 4096
    elif dtype_int == 3:  # BFP8
        return 32 * 32 + 64  # 1088
    elif dtype_int == 5:  # UInt8/Int8
        return 32 * 32  # 1024
    elif dtype_int == 4:  # BFP4
        return 512 + 64  # 576
    else:
        raise ValueError(f"Unsupported dtype for tile size calculation: {dtype}")
