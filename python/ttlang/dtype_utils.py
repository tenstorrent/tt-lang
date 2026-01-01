# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Data type conversion utilities between PyTorch, TTNN, and MLIR types."""

import torch

try:
    import ttnn
except ModuleNotFoundError:
    ttnn = None

from ttmlir.dialects import ttcore


def is_ttnn_tensor(tensor) -> bool:
    """Check if tensor is a ttnn.Tensor."""
    if ttnn is None:
        return False
    return isinstance(tensor, ttnn.Tensor)


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
    # ttnn.DataType is not guaranteed to be directly `int()`-convertible in the
    # Python bindings. Prefer `.value` (enum int payload) when available.
    if isinstance(dtype, int):
        dtype_int = dtype
    elif hasattr(dtype, "value") and isinstance(getattr(dtype, "value"), int):
        dtype_int = int(getattr(dtype, "value"))
    else:
        raise TypeError(
            f"Expected dtype to be an int or an enum-like value with an integer "
            f"`.value` (e.g., ttnn.DataType). Got {type(dtype).__name__}: {dtype!r}."
        )
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
