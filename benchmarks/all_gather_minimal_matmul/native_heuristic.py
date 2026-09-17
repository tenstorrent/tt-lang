# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Load TT-Metal's AGMM configuration resolver from the measured revision."""

import importlib
import os
import subprocess
import sys
from pathlib import Path


def _load_get_agmm_config(
    source_root: Path | None = None, expected_revision: str | None = None
):
    if source_root is None:
        runtime_root = os.environ.get("TT_METAL_RUNTIME_ROOT")
        if runtime_root is None:
            raise RuntimeError(
                "TT_METAL_RUNTIME_ROOT or --ttmetal-source-root is required "
                "for --native-heuristic"
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
        module = importlib.import_module("models.tt_dit.utils.matmul")
    finally:
        sys.path.pop(0)

    module_file = Path(module.__file__).resolve()
    if not module_file.is_relative_to(source_root):
        raise RuntimeError(
            f"loaded AGMM heuristic from {module_file}, expected {source_root}"
        )
    return module.get_agmm_config


def resolve_agmm_config(
    *,
    ttnn_module,
    m_elements: int,
    full_k_elements: int,
    n_elements_per_device: int,
    full_grid,
    device_count: int,
    num_links: int,
    compute_grid: tuple[int, int],
    source_root: Path | None,
    expected_revision: str,
    fuse_swiglu: bool,
    use_addcmul: bool,
) -> tuple[tuple[int, int], int, int, int, tuple[int, int], int]:
    """Return the grid, blocking, subblock, and worker count selected upstream."""

    get_agmm_config = _load_get_agmm_config(source_root, expected_revision)
    resolved_grid, config, workers_per_link = get_agmm_config(
        m_elements,
        full_k_elements,
        n_elements_per_device,
        full_grid=full_grid,
        cluster_size=device_count,
        num_links=num_links,
        core_grid=ttnn_module.CoreCoord(*compute_grid),
        use_heuristic=True,
        fuse_swiglu=fuse_swiglu,
        use_addcmul=use_addcmul,
        force_transpose=True,
    )
    return (
        (resolved_grid.x, resolved_grid.y),
        config.M_block_size,
        config.K_block_size,
        config.N_block_size,
        (config.subblock_h, config.subblock_w),
        workers_per_link,
    )
