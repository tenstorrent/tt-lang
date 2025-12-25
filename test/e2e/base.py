# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Base classes for E2E tests.

Provides E2ETestBase with ordered pipeline stages.
"""

from pathlib import Path

import pytest


class E2ETestBase:
    """
    Base class for all E2E tests.

    Defines ordered pipeline stages. Subclasses override test_build_module()
    and optionally other test methods.

    Use @pytest.mark.xfail decorators on individual test methods for expected failures.
    """

    OUTPUT_DIR: Path

    @pytest.fixture(scope="class", autouse=True)
    def setup(self, request, device, system_desc_path):
        """Initialize test class with output directory."""
        request.cls.OUTPUT_DIR = Path(f"build/test/e2e/{request.cls.__name__}")
        request.cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def output_file(self, name: str) -> Path:
        """Get path for intermediate file."""
        return self.OUTPUT_DIR / name

    # Ordered test stages - to be implemented/overridden by subclasses
    # Note: Subclasses may add fixture parameters to test_build_module signature.
    @pytest.mark.order(1)
    def test_build_module(self, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """Build or load the TTL MLIR module. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement test_build_module()")

    @pytest.mark.order(2)
    def test_compile_to_ttkernel(self):
        """Run TTL-to-TTKernel pass pipeline on the generated module."""
        from .builder.pipeline import compile_ttl_to_ttkernel
        from ttmlir.ir import Context, Module

        # Load module from file saved by test_build_module
        module_file = self.output_file("module.mlir")
        if not module_file.exists():
            pytest.skip(
                f"Module file not found: {module_file}. Run test_build_module first."
            )

        with Context() as ctx:
            with open(module_file) as f:
                module = Module.parse(f.read(), ctx)

            # Compile through pass pipeline
            compiled = compile_ttl_to_ttkernel(module, None)  # Auto-detect system desc

            # Save compiled module
            compiled_file = self.output_file("compiled_module.mlir")
            with open(compiled_file, "w") as f:
                f.write(str(compiled))

    @pytest.mark.order(3)
    @pytest.mark.skip(
        reason="Generated MLIR needs proper TTKernel structure (kernel functions with thread types)"
    )
    def test_translate_to_cpp(self):
        """Translate TTKernel ops to C++ kernel sources."""
        from .builder.kernels import translate_module_to_kernels, write_kernels
        from ttmlir.ir import Context, Module

        compiled_file = self.output_file("compiled_module.mlir")
        if not compiled_file.exists():
            pytest.skip(
                f"Compiled module not found: {compiled_file}. Run test_compile_to_ttkernel first."
            )

        with Context() as ctx:
            with open(compiled_file) as f:
                module = Module.parse(f.read(), ctx)

            # Translate to C++ kernels
            noc_kernels, compute_kernel = translate_module_to_kernels(module)

            # Write kernels to files
            kernel_dir = self.OUTPUT_DIR / "kernels"
            write_kernels(noc_kernels, compute_kernel, kernel_dir)

    @pytest.mark.order(4)
    def test_execute(self):
        """Execute kernels on device."""
        pytest.skip("Kernel execution requires ttnn.generic_op integration")

    @pytest.mark.order(5)
    def test_validate_golden(self):
        """
        Validate result against golden.

        Default implementation compares result with golden from files.
        Subclasses can override for custom validation logic.
        """
        pytest.skip("Depends on test_execute - requires execution infrastructure")
