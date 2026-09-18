# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Load TT-Metal's AGMM configuration resolver from the measured revision."""

import importlib
import os
import subprocess
import sys
from pathlib import Path


def load_ttmetal_symbol(
    module_name: str,
    symbol_name: str,
    source_root: Path | None = None,
    expected_revision: str | None = None,
):
    if source_root is None:
        runtime_root = os.environ.get("TT_METAL_RUNTIME_ROOT")
        if runtime_root is None:
            raise RuntimeError(
                "TT_METAL_RUNTIME_ROOT or --ttmetal-source-root is required "
                "to load TT-Metal model utilities"
            )
        source_root = Path(runtime_root)
    source_root = Path(source_root).resolve()
    if expected_revision is not None:
        actual_revision = subprocess.check_output(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual_revision != expected_revision:
            raise RuntimeError(
                f"TT-Metal source revision is {actual_revision}, "
                f"expected {expected_revision}"
            )
    sys.path.insert(0, str(source_root))
    try:
        module = importlib.import_module(module_name)
    finally:
        sys.path.pop(0)

    module_file = Path(module.__file__).resolve()
    if not module_file.is_relative_to(source_root):
        raise RuntimeError(
            f"loaded AGMM heuristic from {module_file}, expected {source_root}"
        )
    return getattr(module, symbol_name)


def resolve_agmm_config(
    *,
    ttnn_module,
    m_elements: int,
    full_k_elements: int,
    n_elements_per_device: int,
    full_grid,
    device_count: int,
    num_links: int,
    compute_grid: tuple[int, int] | None,
    source_root: Path | None,
    expected_revision: str,
    fuse_swiglu: bool,
    use_addcmul: bool,
) -> tuple[tuple[int, int], int, int, int, tuple[int, int], int]:
    """Return the model's grid, blocking, subblock, and worker count."""

    get_agmm_config = load_ttmetal_symbol(
        "models.tt_dit.utils.matmul",
        "get_agmm_config",
        source_root,
        expected_revision,
    )
    agmm_block_size = load_ttmetal_symbol(
        "models.tt_dit.models.transformers.minimax_h3.agmm_config",
        "agmm_block_size",
        source_root,
        expected_revision,
    )
    default_block_size = agmm_block_size(full_k_elements, n_elements_per_device)
    use_heuristic = default_block_size is None
    resolved_grid, config, workers_per_link = get_agmm_config(
        m_elements,
        full_k_elements,
        n_elements_per_device,
        full_grid=full_grid,
        cluster_size=device_count,
        num_links=num_links,
        core_grid=(
            ttnn_module.CoreCoord(*compute_grid) if compute_grid is not None else None
        ),
        default_block_size=default_block_size,
        use_heuristic=use_heuristic,
        fuse_swiglu=fuse_swiglu,
        use_addcmul=use_addcmul,
        force_transpose=True,
    )
    k_tiles_per_device = full_k_elements // (32 * device_count)
    if k_tiles_per_device % config.K_block_size:
        raise ValueError(
            f"K block {config.K_block_size} does not divide "
            f"{k_tiles_per_device} K tiles per device"
        )
    if config.N_block_size % config.subblock_w:
        raise ValueError(
            f"N block {config.N_block_size} is not divisible by "
            f"subblock width {config.subblock_w}"
        )
    if fuse_swiglu and config.N_block_size % 2:
        raise ValueError(
            f"fused SwiGLU requires an even N block, got {config.N_block_size}"
        )
    return (
        (resolved_grid.x, resolved_grid.y),
        config.M_block_size,
        config.K_block_size,
        config.N_block_size,
        (config.subblock_h, config.subblock_w),
        workers_per_link,
    )
