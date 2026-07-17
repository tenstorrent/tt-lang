# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared test utilities for tt-lang test suite.

Provides unified feature detection (ttnn availability, hardware detection),
tensor creation helpers, and comparison utilities. Used across pytest conftest
files, lit configuration, and test scripts.

Device availability is checked without importing ttnn.
"""

import glob
import importlib.util
import os
import sys
from contextlib import contextmanager
from typing import Any, Sequence

# =============================================================================
# Feature detection
# =============================================================================

# Prefer runtime state over the wheel's build-time device flag.
_hardware_available = False


def _has_tenstorrent_device_node() -> bool:
    return bool(glob.glob("/dev/tenstorrent/*") or glob.glob("/dev/tenstorrent[0-9]*"))


if os.environ.get("TT_METAL_SIMULATOR"):
    _hardware_available = True
elif os.environ.get("TTLANG_HAS_DEVICE") == "1":
    _hardware_available = True
elif _has_tenstorrent_device_node():
    _hardware_available = True
else:
    try:
        from ttl.config import HAS_TT_DEVICE

        _hardware_available = HAS_TT_DEVICE
    except ImportError:
        _hardware_available = False

# Set compile-only mode if no hardware.
if not _hardware_available:
    os.environ["TTLANG_COMPILE_ONLY"] = "1"

# Check if TTNN is available (lightweight check without importing).
_ttnn_available = False
try:
    _ttnn_available = importlib.util.find_spec("ttnn") is not None
except Exception:
    pass

ttnn = None  # Lazy import - loaded when first needed


def _get_ttnn():
    """Lazy import of ttnn module."""
    global ttnn, _ttnn_available
    if ttnn is None and _ttnn_available:
        try:
            import ttnn as _ttnn

            ttnn = _ttnn
        except (ImportError, ModuleNotFoundError):
            _ttnn_available = False
            ttnn = None
    return ttnn


def is_ttnn_available() -> bool:
    """
    Check if ttnn module is available without importing it.

    Uses importlib.util.find_spec for lightweight detection that avoids
    the overhead of actually importing ttnn (which can be slow).

    Returns:
        True if ttnn can be imported, False otherwise.
    """
    return _ttnn_available


def is_hardware_available() -> bool:
    """
    Check if Tenstorrent hardware is available.

    Checks in order:
    1. TT_METAL_SIMULATOR environment variable (simulation mode)
    2. TTLANG_HAS_DEVICE environment variable (set by CMake)
    3. Runtime device nodes (/dev/tenstorrent/* or /dev/tenstorrent[0-9]*)
    4. ttl.config.HAS_TT_DEVICE, the wheel's build-time value (fallback)

    Step 3 precedes step 4 so an installed light wheel, built with no device
    and therefore HAS_TT_DEVICE=False, still runs on a host that has a chip.

    Returns:
        True if hardware or simulator is available, False otherwise.
    """
    return _hardware_available


def pin_xdist_worker_to_device() -> None:
    """Restrict a pytest-xdist worker to one chip and cache directory."""
    if os.environ.get("TTLANG_PIN_XDIST_WORKERS_TO_DEVICES") != "1":
        return
    worker_name = os.environ.get("PYTEST_XDIST_WORKER")
    if not worker_name:
        return
    worker_index = "".join(
        character for character in worker_name if character.isdigit()
    )
    if not worker_index:
        return
    if "TT_VISIBLE_DEVICES" not in os.environ:
        os.environ["TT_VISIBLE_DEVICES"] = worker_index
    cache_root = os.environ.get("TTLANG_XDIST_TT_METAL_CACHE_ROOT")
    if cache_root:
        cache_root = os.path.abspath(cache_root)
        cache_dir = os.path.join(cache_root, f"worker-{worker_index}")
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["TT_METAL_CACHE"] = cache_dir


def require_ttnn():
    """Exit test if TTNN is not available."""
    if not _ttnn_available:
        print("TTNN not available - exiting")
        sys.exit(0)


def require_hardware(message: str = "Skipping test - no hardware available"):
    """Exit early if no hardware available.

    Use this at the start of `if __name__ == "__main__":` blocks in tests
    that need access to Tenstorrent hardware (even just for compilation).

    Note: This does NOT check the TTLANG_COMPILE_ONLY env var - tests can still compile
    kernels in compile-only mode, they just won't execute on device.
    """
    if not _hardware_available:
        print(message)
        sys.exit(0)


# =============================================================================
# Mesh fabric utilities
# =============================================================================


class FabricMeshUnavailable(RuntimeError):
    pass


@contextmanager
def open_fabric_mesh(requested_mesh_shape: tuple[int, int] | None = None):
    """Open a 1D fabric mesh spanning every visible device by default."""
    ttnn_module = _get_ttnn()
    if ttnn_module is None:
        raise FabricMeshUnavailable("TTNN not available")

    if requested_mesh_shape is None:
        # FABRIC_1D requires a 1D topology even when physical discovery is 2-D.
        requested_mesh_shape = (1, ttnn_module.get_num_devices())
    else:
        requested_mesh_shape = tuple(requested_mesh_shape)
    if (
        len(requested_mesh_shape) != 2
        or requested_mesh_shape[0] != 1
        or requested_mesh_shape[1] < 1
    ):
        raise ValueError(
            "FABRIC_1D requires a logical mesh shape of (1, num_devices) with "
            "num_devices greater than zero"
        )

    mesh_device = None
    try:
        ttnn_module.set_fabric_config(ttnn_module.FabricConfig.FABRIC_1D)
        mesh_device = ttnn_module.open_mesh_device(
            ttnn_module.MeshShape(requested_mesh_shape)
        )
        yield mesh_device
    finally:
        if mesh_device is not None:
            ttnn_module.close_mesh_device(mesh_device)
        ttnn_module.set_fabric_config(ttnn_module.FabricConfig.DISABLED)


# =============================================================================
# Tensor creation utilities
# =============================================================================


def torch_dtype_from_name(name: str):
    """Parse common test dtype names into PyTorch dtypes."""
    import torch

    normalized = name.lower()
    if normalized in ("bf16", "bfloat16"):
        return torch.bfloat16
    if normalized in ("fp32", "f32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype name {name!r}")


def torch_dtype_from_env(var_name: str, default: str = "bf16"):
    """Read a PyTorch dtype name from an environment variable."""
    return torch_dtype_from_name(os.environ.get(var_name, default))


def to_dram(torch_tensor, device):
    """Create a TTNN tensor in DRAM from a torch tensor.

    Args:
        torch_tensor: Source torch tensor
        device: TTNN device handle

    Returns:
        TTNN tensor in DRAM with TILE_LAYOUT
    """
    from ttl.dtype_utils import torch_dtype_to_ttnn_datatype

    ttnn = _get_ttnn()
    if ttnn is None:
        raise RuntimeError("TTNN not available")
    return ttnn.from_torch(
        torch_tensor,
        dtype=torch_dtype_to_ttnn_datatype(torch_tensor.dtype),
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def to_l1(torch_tensor, device):
    """Create a TTNN tensor in L1 from a torch tensor.

    Creates in DRAM first then moves to L1 (required by TTNN).

    Args:
        torch_tensor: Source torch tensor
        device: TTNN device handle

    Returns:
        TTNN tensor in L1 with TILE_LAYOUT
    """
    ttnn = _get_ttnn()
    if ttnn is None:
        raise RuntimeError("TTNN not available")
    dram_tensor = to_dram(torch_tensor, device)
    return ttnn.to_memory_config(dram_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)


def make_compare_inputs(shape: Sequence[int], dtype: Any) -> tuple[Any, Any]:
    """Create compare inputs with equal, less-than, and greater-than lanes.

    Random f32 operands can be near-equal enough for hardware compare precision
    to disagree with torch. This deterministic pattern keeps values separated
    by a wide margin while still covering equality and both inequalities.
    """
    import torch

    numel = 1
    for dim in shape:
        numel *= dim
    idx = torch.arange(numel, dtype=torch.float32).reshape(tuple(shape))
    lhs = ((idx % 17) / 64.0).to(dtype)
    rhs = lhs.clone()
    selector = (idx.to(torch.int64) % 3).to(torch.int64)
    rhs = torch.where(selector == 1, (lhs.float() - 0.5).to(dtype), rhs)
    rhs = torch.where(selector == 2, (lhs.float() + 0.5).to(dtype), rhs)
    return lhs, rhs


def assert_compare_result(result: Any, expected: Any) -> None:
    """Assert exact numeric 0/1 mask equality, matching the result dtype."""
    import torch

    expected = expected.to(result.dtype)
    assert torch.equal(result, expected), (
        f"Compare mask mismatch: {(result != expected).sum().item()} "
        f"/ {result.numel()}"
    )


# =============================================================================
# Tensor comparison utilities
# =============================================================================


def assert_pcc(golden, actual, threshold=0.9999):
    """Assert Pearson correlation coefficient between tensors exceeds threshold.

    Args:
        golden: Expected tensor values
        actual: Actual tensor values from computation
        threshold: Minimum PCC required (default 0.9999, consistent with tt-metal)

    Raises:
        AssertionError: If PCC is below threshold
    """
    import torch

    golden_flat = golden.flatten().float()
    actual_flat = actual.flatten().float()

    # Handle constant tensors (no variance).
    if golden_flat.std() == 0 and actual_flat.std() == 0:
        # Both constant - check if same constant.
        if torch.allclose(golden_flat, actual_flat):
            return 1.0
        else:
            raise AssertionError(
                f"Both tensors are constant but differ: "
                f"golden={golden_flat[0].item()}, actual={actual_flat[0].item()}"
            )

    if golden_flat.std() == 0 or actual_flat.std() == 0:
        raise AssertionError(
            f"Cannot compute PCC: one tensor is constant "
            f"(golden std={golden_flat.std()}, actual std={actual_flat.std()})"
        )

    # Compute Pearson correlation.
    golden_centered = golden_flat - golden_flat.mean()
    actual_centered = actual_flat - actual_flat.mean()

    numerator = (golden_centered * actual_centered).sum()
    denominator = torch.sqrt((golden_centered**2).sum() * (actual_centered**2).sum())

    pcc = numerator / denominator

    if pcc < threshold:
        raise AssertionError(
            f"PCC {pcc:.6f} is below threshold {threshold}. "
            f"Golden: mean={golden_flat.mean():.4f}, std={golden_flat.std():.4f}. "
            f"Actual: mean={actual_flat.mean():.4f}, std={actual_flat.std():.4f}."
        )

    return pcc.item()


def assert_allclose(actual, expected, rtol=1e-5, atol=1e-8, verbose=True):
    """Assert tensors are element-wise close within tolerance.

    Args:
        actual: Actual tensor from computation
        expected: Expected tensor values
        rtol: Relative tolerance
        atol: Absolute tolerance
        verbose: If True, print diff stats on failure

    Raises:
        AssertionError: If tensors differ beyond tolerance
    """
    import torch

    if not torch.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = (actual - expected).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()

        msg = (
            f"Tensors not close: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, "
            f"rtol={rtol}, atol={atol}"
        )

        if verbose:
            # Find location of max difference.
            max_idx = diff.argmax().item()
            actual_val = actual.flatten()[max_idx].item()
            expected_val = expected.flatten()[max_idx].item()
            msg += (
                f"\nMax diff at flat index {max_idx}: "
                f"actual={actual_val:.6e}, expected={expected_val:.6e}"
            )

        raise AssertionError(msg)


def to_l1_sharded(torch_tensor, device, layout="height"):
    """Create a sharded TTNN tensor in L1 from a torch tensor.

    Shards the tensor across a single core with the full tensor as one shard.
    This exercises the sharded TensorAccessor path while keeping the test simple.

    Args:
        torch_tensor: Source torch tensor (dimensions must be multiples of 32)
        device: TTNN device handle
        layout: Shard layout -- "height", "width", or "block"

    Returns:
        Sharded TTNN tensor in L1 with TILE_LAYOUT
    """
    ttnn = _get_ttnn()
    if ttnn is None:
        raise RuntimeError("TTNN not available")
    layout_map = {
        "height": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        "width": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        "block": ttnn.TensorMemoryLayout.BLOCK_SHARDED,
    }
    if layout not in layout_map:
        raise ValueError(
            f"Unknown shard layout {layout!r}, expected one of {list(layout_map)}"
        )
    dram_tensor = to_dram(torch_tensor, device)
    rows, cols = torch_tensor.shape[-2], torch_tensor.shape[-1]
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}),
        (rows, cols),
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    sharded_mem_config = ttnn.MemoryConfig(
        layout_map[layout],
        ttnn.BufferType.L1,
        shard_spec,
    )
    return ttnn.to_memory_config(dram_tensor, memory_config=sharded_mem_config)


__all__ = [
    "is_ttnn_available",
    "is_hardware_available",
    "require_ttnn",
    "require_hardware",
    "torch_dtype_from_name",
    "torch_dtype_from_env",
    "to_dram",
    "to_l1",
    "to_l1_sharded",
    "assert_pcc",
    "assert_allclose",
]
