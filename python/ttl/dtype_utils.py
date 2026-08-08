# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Data type conversion utilities between PyTorch, TTNN, and MLIR types."""

import operator

import torch

from .constants import DEFAULT_TILE_SIZE

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


def normalize_tile_dimensions(tile) -> tuple[int, int]:
    """Return validated TT-Metal physical tile dimensions."""
    try:
        tile_height, tile_width = tile
        normalized_tile = (
            operator.index(tile_height),
            operator.index(tile_width),
        )
    except (TypeError, ValueError):
        raise ValueError(
            f"Tile must contain exactly two integer dimensions, got {tile!r}"
        ) from None

    if normalized_tile[0] <= 0 or normalized_tile[1] <= 0:
        raise ValueError(f"Tile dimensions must be positive, got {normalized_tile}")
    try:
        is_supported_tile = ttcore.ir.TileType.is_tt_metal_tile_shape(*normalized_tile)
    except (OverflowError, TypeError):
        is_supported_tile = False
    if not is_supported_tile:
        raise ValueError(
            "Tile dimensions are not constructible by tt-metal: "
            f"{normalized_tile[0]}x{normalized_tile[1]}"
        )
    return normalized_tile


def tile_bytes_from_dtype(dtype, tile=(DEFAULT_TILE_SIZE, DEFAULT_TILE_SIZE)) -> int:
    """
    Calculate tile size in bytes from ttnn dtype.

    The byte size matches ttcore::TileType::getSizeBytes(). Dense and BFP
    formats scale with the physical tile dimensions. Every valid ttnn.DataType
    with a corresponding ttcore::DataType is supported; FP8_E4M3 has no
    ttcore representation. Compute eligibility is validated separately by the
    compiler.

    Args:
        dtype: ttnn.DataType enum value
        tile: Physical tile dimensions as (height, width)

    Returns:
        Tile size in bytes

    Raises:
        ValueError: If dtype or its tile dimensions are not supported
    """
    tile_height, tile_width = normalize_tile_dimensions(tile)

    _ensure_ttnn()
    if ttnn is None:
        raise RuntimeError("ttnn is not available")

    tile_elements = tile_height * tile_width
    # Local sizing keeps metadata generation independent of MetalContext, which
    # tt-metal's Tile::get_tile_size uses to query L1 alignment.
    # Keep this mapping synchronized with ttcore::TileType::getSizeBytes().
    if dtype in (ttnn.DataType.BFLOAT16, ttnn.DataType.UINT16):
        return tile_elements * 2
    if dtype in (
        ttnn.DataType.FLOAT32,
        ttnn.DataType.INT32,
        ttnn.DataType.UINT32,
    ):
        return tile_elements * 4
    if dtype == ttnn.DataType.UINT8:
        return tile_elements
    bfp_dtypes = (
        ttnn.DataType.BFLOAT8_B,
        ttnn.DataType.BFLOAT4_B,
    )
    if dtype not in bfp_dtypes:
        raise ValueError(f"Unsupported dtype for tile size calculation: {dtype}")
    # tt-metal Tile::get_tile_size stores one exponent byte per 16-element face
    # row and aligns the complete exponent section to L1.
    # TODO(#511): Source L1 alignment from shared target metadata.
    elements_per_exponent = 16
    l1_alignment_bytes = 16
    if tile_elements % elements_per_exponent != 0:
        raise ValueError(
            "BFP tile element count must be divisible by "
            f"{elements_per_exponent}, got {tile_elements}"
        )
    exponent_count = tile_elements // elements_per_exponent
    exponent_bytes = (
        (exponent_count + l1_alignment_bytes - 1) // l1_alignment_bytes
    ) * l1_alignment_bytes
    if dtype == ttnn.DataType.BFLOAT8_B:
        return tile_elements + exponent_bytes
    return tile_elements // 2 + exponent_bytes
