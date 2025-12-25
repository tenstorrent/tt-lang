# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Base classes for E2E tests.

Provides E2ETestBase with ordered pipeline stages and cache management.
"""

from pathlib import Path
from typing import Any, List, TypedDict

import pytest
from ttmlir.ir import Module
from ttmlir.passmanager import PassManager


class E2ETestCache(TypedDict, total=False):
    """Cache for passing data between ordered test stages."""

    module: Module  # TTL MLIR module
    compiled_module: Module  # After pass pipeline
    noc_kernels: List[Any]  # Data movement kernels (KernelSpec)
    compute_kernel: Any  # Compute kernel (KernelSpec)
    torch_inputs: List[Any]  # Input tensors
    device_inputs: List[Any]  # On-device tensors
    output_tensor: Any  # Device output
    golden: Any  # Expected result
    result: Any  # Actual result


class E2ETestBase:
    """
    Base class for all E2E tests.

    Defines ordered pipeline stages with dependency checking and caching.
    Subclasses override test_build_module() and optionally test_validate_golden().

    Use @pytest.mark.xfail decorators on individual test methods for expected failures.
    """

    CACHE: E2ETestCache
    OUTPUT_DIR: Path

    @pytest.fixture(scope="class", autouse=True)
    def setup(self, request, device, system_desc_path):
        """Initialize test class with cache and output directory."""
        request.cls.CACHE = {}
        request.cls.OUTPUT_DIR = Path(f"build/test/e2e/{request.cls.__name__}")
        request.cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _check_cache_dependencies(self, keys: List[str]) -> None:
        """
        Skip test if any dependency is missing from cache.

        This allows downstream tests to skip gracefully when an upstream
        test fails, rather than failing with confusing errors.

        Args:
            keys: List of cache keys that must be present.
        """
        for key in keys:
            if key not in self.CACHE:
                pytest.skip(f"Missing dependency: {key}")

    def _test_pipeline_step(
        self, input_key: str, output_key: str, passes: List[str]
    ) -> None:
        """
        Run MLIR passes and cache output.

        Helper for common pattern of running a pass pipeline on a cached module.

        Args:
            input_key: Cache key for input module.
            output_key: Cache key to store output module.
            passes: List of pass names to run.
        """
        self._check_cache_dependencies([input_key])
        module = self.CACHE[input_key]
        pipeline = f"builtin.module({','.join(passes)})"
        pm = PassManager.parse(pipeline)
        pm.run(module.operation)
        self.CACHE[output_key] = module

    # Ordered test stages - to be implemented/overridden by subclasses
    def test_build_module(self):
        """Build or load the TTL MLIR module. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement test_build_module()")

    @pytest.mark.order(after="test_build_module")
    def test_compile_to_ttkernel(self):
        """Run TTL-to-TTKernel pass pipeline."""
        from .pipeline import compile_ttl_to_ttkernel

        self._check_cache_dependencies(["module"])
        self.CACHE["compiled_module"] = compile_ttl_to_ttkernel(
            self.CACHE["module"], self.CACHE.get("system_desc_path")
        )

    @pytest.mark.order(after="test_compile_to_ttkernel")
    def test_translate_to_cpp(self):
        """Translate TTKernel ops to C++ kernel sources."""
        from .kernels import translate_module_to_kernels

        self._check_cache_dependencies(["compiled_module"])
        noc_kernels, compute_kernel = translate_module_to_kernels(
            self.CACHE["compiled_module"]
        )
        self.CACHE["noc_kernels"] = noc_kernels
        self.CACHE["compute_kernel"] = compute_kernel

    @pytest.mark.order(after="test_translate_to_cpp")
    def test_execute(self):
        """Execute kernels on device."""
        from .kernels import execute_kernels

        self._check_cache_dependencies(
            ["noc_kernels", "compute_kernel", "torch_inputs", "device_inputs"]
        )
        result = execute_kernels(
            self.CACHE["noc_kernels"],
            self.CACHE["compute_kernel"],
            self.CACHE["device_inputs"],
            self.CACHE["output_tensor"],
            self.CACHE.get("device"),
        )
        self.CACHE["result"] = result

    @pytest.mark.order(after=["test_execute", "test_build_module"])
    def test_validate_golden(self):
        """
        Validate result against golden.

        Default implementation compares cached result with cached golden.
        Subclasses can override for custom validation logic.
        """
        from .utils import compare_tensors

        self._check_cache_dependencies(["golden", "result"])
        comparison = compare_tensors(
            self.CACHE["golden"],
            self.CACHE["result"],
            error_tol=getattr(self, "ERROR_TOL", 1e-2),
        )
        assert comparison.passed, comparison.message
