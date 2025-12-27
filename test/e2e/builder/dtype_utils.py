# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Dtype conversion utilities for E2E tests.

Shared utilities for converting between torch dtypes and MLIR type strings.
"""

import torch
from ttmlir.dialects import ttcore


def torch_dtype_to_mlir_str(dtype: torch.dtype) -> str:
    """
    Convert torch dtype to MLIR type string.

    Args:
        dtype: PyTorch dtype.

    Returns:
        MLIR type string (e.g., "bf16", "f32").
    """
    if dtype == torch.float32:
        return "f32"
    elif dtype == torch.bfloat16:
        return "bf16"
    elif dtype == torch.float16:
        return "f16"
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def torch_dtype_to_ttcore_datatype(dtype: torch.dtype) -> int:
    """
    Convert torch dtype to ttcore DataType integer value.

    Args:
        dtype: PyTorch dtype.

    Returns:
        Integer value for ttcore.DataType enum.
    """
    if dtype == torch.float32:
        return int(ttcore.DataType.Float32)
    elif dtype == torch.bfloat16:
        return int(ttcore.DataType.BFloat16)
    elif dtype == torch.float16:
        return int(ttcore.DataType.Float16)
    else:
        raise ValueError(f"Unsupported dtype for tile: {dtype}")



