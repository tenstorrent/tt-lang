# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Pytest fixtures for middle-end tests.

Provides device management, skip conditions, platform-aware markers,
and test metadata extraction.
"""

import os
from typing import Optional, Tuple, Set

import pytest

from .test_utils import (
    TTLCompileException,
    TTLRuntimeException,
    TTLGoldenException,
    torch_dtype_to_abbrev,
    ALL_BACKENDS,
    ALL_SYSTEMS,
)

# Check for ttnn availability.
try:
    import ttnn

    TTNN_AVAILABLE = True
except ImportError:
    TTNN_AVAILABLE = False
    ttnn = None

# Device caching for efficiency.
_current_device = None
_current_system_type: Optional[str] = None


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_device: mark test as requiring Tenstorrent hardware"
    )
    config.addinivalue_line(
        "markers",
        "skip_config(*configs): skip test for specific platform/backend combos",
    )
    config.addinivalue_line(
        "markers",
        "only_config(*configs): run test only on specific platform/backend combos",
    )


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--sys-desc",
        action="store",
        default=None,
        help="Path to system descriptor file",
    )
    parser.addoption(
        "--dump-mlir",
        action="store_true",
        default=False,
        help="Save generated MLIR to build directory",
    )
    parser.addoption(
        "--check-pcc",
        action="store_true",
        default=False,
        help="Use PCC-based comparison instead of allclose",
    )


def _get_system_type() -> Optional[str]:
    """
    Detect the system type from the current device.

    Returns:
        System type string ('n150', 'n300', 'p150', 'p300', 'llmbox', 'tg') or None.
    """
    global _current_system_type

    if _current_system_type is not None:
        return _current_system_type

    if not TTNN_AVAILABLE:
        return None

    try:
        # Try to get system info from device.
        # This is a simplified detection - real detection would use system_desc.
        device = ttnn.open_device(device_id=0)
        # For now, default to n150 for single device.
        # Real detection would inspect chip type and count.
        _current_system_type = "n150"
        ttnn.close_device(device)
        return _current_system_type
    except Exception:
        return None


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection based on platform and extract metadata.

    Handles:
    - Skipping tests if ttnn unavailable
    - skip_config marker processing
    - only_config marker processing
    - Test metadata extraction for XML reporting
    """
    if not TTNN_AVAILABLE:
        skip_ttnn = pytest.mark.skip(reason="ttnn not available")
        for item in items:
            item.add_marker(skip_ttnn)
        return

    system_type = _get_system_type()

    for item in items:
        # Process skip_config markers.
        for marker in item.iter_markers(name="skip_config"):
            for platform_config in marker.args:
                platform_set = (
                    set(platform_config)
                    if isinstance(platform_config, (list, tuple))
                    else {platform_config}
                )

                # Validate config.
                invalid = platform_set - ALL_BACKENDS - ALL_SYSTEMS
                if invalid:
                    raise ValueError(
                        f"Invalid skip_config: {invalid}. "
                        f"Valid values: {ALL_BACKENDS | ALL_SYSTEMS}"
                    )

                # Skip if current system matches.
                if system_type and system_type in platform_set:
                    reason = marker.kwargs.get("reason", "")
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"Skipped for {platform_config}. {reason}"
                        )
                    )

        # Process only_config markers.
        for marker in item.iter_markers(name="only_config"):
            for platform_config in marker.args:
                platform_set = (
                    set(platform_config)
                    if isinstance(platform_config, (list, tuple))
                    else {platform_config}
                )

                # Validate config.
                invalid = platform_set - ALL_BACKENDS - ALL_SYSTEMS
                if invalid:
                    raise ValueError(
                        f"Invalid only_config: {invalid}. "
                        f"Valid values: {ALL_BACKENDS | ALL_SYSTEMS}"
                    )

                # Skip if current system does NOT match.
                if system_type and system_type not in platform_set:
                    reason = marker.kwargs.get("reason", "")
                    item.add_marker(
                        pytest.mark.skip(
                            reason=f"Only runs on {platform_config}. {reason}"
                        )
                    )

        # Extract test metadata for XML reporting.
        _extract_test_metadata(item)


def _extract_test_metadata(item: pytest.Item) -> None:
    """
    Extract test metadata for XML reporting.

    Adds properties like op_name, input_dtype, config to test item.
    """
    if not hasattr(item, "callspec"):
        return

    params = item.callspec.params

    # Extract operation name.
    if "op" in params:
        op = params["op"]
        if hasattr(op, "name"):
            _safe_add_property(item, "op_name", op.name)
            _safe_add_property(item, "op_arity", str(getattr(op, "arity", "?")))

    # Extract config details.
    if "config" in params:
        config = params["config"]
        if hasattr(config, "dtype"):
            import torch

            _safe_add_property(item, "dtype", torch_dtype_to_abbrev(config.dtype))
        if hasattr(config, "tile_h") and hasattr(config, "tile_w"):
            _safe_add_property(item, "tile_size", f"{config.tile_h}x{config.tile_w}")
        if hasattr(config, "num_tiles"):
            _safe_add_property(item, "num_tiles", str(config.num_tiles))


def _safe_add_property(item: pytest.Item, key: str, value: str) -> None:
    """Safely add a property to test item."""
    try:
        if hasattr(item, "user_properties"):
            item.user_properties.append((key, value))
    except Exception:
        pass


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item: pytest.Item):
    """
    Classify failure stage during test execution.

    Distinguishes between compile, runtime, and golden failures.
    """
    failure_stage = "success"

    outcome = yield
    try:
        outcome.get_result()
    except TTLCompileException:
        failure_stage = "compile"
        raise
    except TTLRuntimeException:
        failure_stage = "runtime"
        raise
    except TTLGoldenException:
        failure_stage = "golden"
        raise
    except Exception:
        failure_stage = "unknown"
        raise
    finally:
        _safe_add_property(item, "failure_stage", failure_stage)


@pytest.fixture(scope="session")
def device():
    """
    Session-scoped fixture for Tenstorrent device.

    Opens the device once for all tests and closes it when done.
    Device is cached for efficiency across tests.
    """
    global _current_device

    if not TTNN_AVAILABLE:
        pytest.skip("ttnn not available")

    if _current_device is not None:
        yield _current_device
        return

    dev = ttnn.open_device(device_id=0)
    _current_device = dev
    yield dev
    ttnn.close_device(dev)
    _current_device = None


@pytest.fixture(scope="session")
def system_desc_path(request):
    """
    Path to the system descriptor file.

    Uses --sys-desc option if provided, SYSTEM_DESC_PATH env var,
    or generates one from the current device.
    """
    # Check command line option first.
    cli_path = request.config.getoption("--sys-desc")
    if cli_path and os.path.exists(cli_path):
        return cli_path

    # Check environment variable.
    env_path = os.environ.get("SYSTEM_DESC_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # Generate system descriptor if not provided.
    try:
        from _ttmlir_runtime import runtime

        system_desc = runtime.get_current_system_desc()
        generated_path = "/tmp/ttlang_test_system.ttsys"
        system_desc.store(generated_path)
        return generated_path
    except (ImportError, Exception) as e:
        pytest.skip(f"Cannot get system descriptor: {e}")


@pytest.fixture
def dump_mlir(request):
    """Fixture to get --dump-mlir option value."""
    return request.config.getoption("--dump-mlir")


@pytest.fixture
def check_pcc(request):
    """Fixture to get --check-pcc option value."""
    return request.config.getoption("--check-pcc")


@pytest.fixture
def assert_close():
    """
    Fixture providing tensor comparison with appropriate tolerances.

    Returns a function that compares tensors with bfloat16-appropriate tolerances.
    """
    import torch

    def _assert_close(
        actual: torch.Tensor,
        expected: torch.Tensor,
        rtol: float = 5e-2,
        atol: float = 1e-1,
    ):
        """Compare tensors with tolerance appropriate for bfloat16."""
        # Convert to float32 for comparison to avoid bfloat16 precision issues.
        actual_f32 = actual.float()
        expected_f32 = expected.float()

        if not torch.allclose(actual_f32, expected_f32, rtol=rtol, atol=atol):
            max_abs_diff = (actual_f32 - expected_f32).abs().max().item()
            max_rel_diff = (
                ((actual_f32 - expected_f32).abs() / (expected_f32.abs() + 1e-8))
                .max()
                .item()
            )
            pytest.fail(
                f"Tensor mismatch:\n"
                f"  Max absolute diff: {max_abs_diff:.6f} (tol: {atol})\n"
                f"  Max relative diff: {max_rel_diff:.6f} (tol: {rtol})\n"
                f"  Shape: {actual.shape}"
            )

    return _assert_close
