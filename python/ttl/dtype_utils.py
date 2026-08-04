# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Data type conversion utilities between PyTorch, TTNN, and MLIR types."""

import torch

ttnn = None  # Lazy-loaded via _ensure_ttnn()


def _ensure_ttnn():
    """Lazy import of ttnn."""
    global ttnn
    if ttnn is not None:
        return ttnn
    try:
        import ttnn as _ttnn

        ttnn = _ttnn
    except (ModuleNotFoundError, ImportError):
        pass
    return ttnn


from ttl.dialects import ttcore


def is_ttnn_tensor(tensor) -> bool:
    """Check if tensor is a ttnn.Tensor."""
    _ensure_ttnn()
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
    except (ModuleNotFoundError, ImportError):
        raise ImportError("ttnn module not available")

    match ttnn_dtype:
        case ttnn.DataType.FLOAT32:
            return ttcore.DataType.Float32
        case ttnn.DataType.BFLOAT16:
            return ttcore.DataType.BFloat16
        case ttnn.DataType.BFLOAT8_B:
            return ttcore.DataType.BFP_BFloat8
        case ttnn.DataType.BFLOAT4_B:
            return ttcore.DataType.BFP_BFloat4
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
    except (ModuleNotFoundError, ImportError):
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


def format_name_to_ttnn_dtype(name: str):
    """Convert a data format name string to a ttnn.DataType enum value.

    Accepts names produced by the compiler's DFB metadata, e.g.,
    "bfloat16", "float32".

    Raises:
        ValueError: If the name is not recognized.
    """
    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    match name:
        case "bfloat16" | "bf16":
            return ttnn.DataType.BFLOAT16
        case "bfloat4_b" | "bfp_bf4":
            return ttnn.DataType.BFLOAT4_B
        case "bfloat8_b" | "bfp_bf8":
            return ttnn.DataType.BFLOAT8_B
        case "float16" | "f16":
            return ttnn.DataType.BFLOAT16  # hardware implements f16 as bf16
        case "float32" | "f32":
            return ttnn.DataType.FLOAT32
        case "int32" | "i32" | "si32":
            return ttnn.DataType.INT32
        case "uint32" | "u32" | "ui32":
            return ttnn.DataType.UINT32
        case "uint16" | "u16" | "ui16":
            return ttnn.DataType.UINT16
        case "uint8" | "u8" | "ui8":
            return ttnn.DataType.UINT8
        case _:
            raise ValueError(
                f"Unrecognized data format name '{name}' for ttnn.DataType"
            )
