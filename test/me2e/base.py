# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Base classes for E2E tests.

Provides E2ETestBase with ordered pipeline stages:
1. Build MLIR module (with reader, compute, writer threads)
2. Compile TTL → TTKernel
3. Translate TTKernel → C++
4. Execute on device
5. Validate against golden
"""

from pathlib import Path

import pytest
import torch


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
        request.cls.OUTPUT_DIR = Path(f"build/test/me2e/{request.cls.__name__}")
        request.cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def output_file(self, name: str) -> Path:
        """Get path for intermediate file."""
        return self.OUTPUT_DIR / name

    # Ordered test stages - to be implemented/overridden by subclasses.

    @pytest.mark.order(1)
    def test_build_module(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        """
        Build or load the TTL MLIR module.

        Subclasses override this with their own fixture parameters.
        Must save module.mlir, inputs.pt, and golden.pt to self.OUTPUT_DIR.
        """
        raise NotImplementedError("Subclasses must implement test_build_module()")

    @pytest.mark.order(2)
    def test_compile_to_ttkernel(self):
        """Run TTL-to-TTKernel pass pipeline on the generated module."""
        from .builder.pipeline import compile_ttl_to_ttkernel

        import ttl.dialects.ttl as ttl
        from ttmlir.ir import Context, Module

        # Load module from file saved by test_build_module.
        module_file = self.output_file("module.mlir")
        if not module_file.exists():
            pytest.skip(
                f"Module file not found: {module_file}. Run test_build_module first."
            )

        ctx = Context()
        ttl.ensure_dialects_registered(ctx)

        with ctx:
            with open(module_file) as f:
                module = Module.parse(f.read(), ctx)

            # Compile through pass pipeline.
            compiled = compile_ttl_to_ttkernel(module, None)  # Auto-detect system desc.

            # Save compiled module.
            compiled_file = self.output_file("compiled_module.mlir")
            with open(compiled_file, "w") as f:
                f.write(str(compiled))

    @pytest.mark.order(3)
    def test_translate_to_cpp(self):
        """Translate TTKernel ops to C++ kernel sources."""
        from .builder.kernels import translate_module_to_kernels, write_kernels

        import ttl.dialects.ttl as ttl
        from ttmlir.ir import Context, Module

        compiled_file = self.output_file("compiled_module.mlir")
        if not compiled_file.exists():
            pytest.skip(
                f"Compiled module not found: {compiled_file}. Run test_compile_to_ttkernel first."
            )

        ctx = Context()
        ttl.ensure_dialects_registered(ctx)

        with ctx:
            with open(compiled_file) as f:
                module = Module.parse(f.read(), ctx)

            # Translate to C++ kernels.
            noc_kernels, compute_kernel = translate_module_to_kernels(module)

            # Write kernels to files.
            kernel_dir = self.OUTPUT_DIR / "kernels"
            write_kernels(noc_kernels, compute_kernel, kernel_dir)

    @pytest.mark.order(4)
    def test_execute(self, device):
        """Execute kernels on device."""
        from .builder.kernels import KernelSpec, ThreadType
        from .builder.ttnn_runner import run_binary_op, run_unary_op

        # Check for kernel files.
        kernel_dir = self.OUTPUT_DIR / "kernels"
        if not kernel_dir.exists():
            pytest.skip(
                f"Kernel directory not found: {kernel_dir}. Run test_translate_to_cpp first."
            )

        # Load kernel specs from files.
        cpp_files = list(kernel_dir.glob("*.cpp"))
        if not cpp_files:
            pytest.skip("No kernel C++ files found.")

        # Load inputs saved by test_build_module.
        inputs_file = self.output_file("inputs.pt")
        if not inputs_file.exists():
            pytest.skip(f"Inputs file not found: {inputs_file}.")

        inputs = torch.load(inputs_file)

        # Build kernel specs from C++ files.
        noc_kernels = []
        compute_kernel = None

        for cpp_file in cpp_files:
            name = cpp_file.stem
            with open(cpp_file) as f:
                source = f.read()

            # Determine thread type from name.
            if "compute" in name.lower():
                compute_kernel = KernelSpec(
                    name=name, thread_type=ThreadType.COMPUTE, source=source
                )
            else:
                noc_kernels.append(
                    KernelSpec(name=name, thread_type=ThreadType.NOC, source=source)
                )

        if compute_kernel is None:
            pytest.skip("No compute kernel found in kernel files.")

        # Run based on arity.
        if len(inputs) == 2:
            result = run_binary_op(
                device=device,
                noc_kernels=noc_kernels,
                compute_kernel=compute_kernel,
                input_a=inputs[0],
                input_b=inputs[1],
                kernel_dir=kernel_dir,
            )
        else:
            result = run_unary_op(
                device=device,
                noc_kernels=noc_kernels,
                compute_kernel=compute_kernel,
                input_a=inputs[0],
                kernel_dir=kernel_dir,
            )

        # Save result for validation.
        torch.save(result, self.output_file("result.pt"))

    @pytest.mark.order(5)
    def test_validate_golden(self):
        """
        Validate result against golden.

        Default implementation compares result with golden from files.
        Subclasses can override for custom validation logic.
        """
        from .utils import compare_tensors_ulp

        result_file = self.output_file("result.pt")
        golden_file = self.output_file("golden.pt")

        if not result_file.exists():
            pytest.skip(
                f"Result file not found: {result_file}. Run test_execute first."
            )

        if not golden_file.exists():
            pytest.skip(f"Golden file not found: {golden_file}.")

        result = torch.load(result_file)
        golden = torch.load(golden_file)

        # Compare using ULP.
        max_ulp, _ = compare_tensors_ulp(result, golden)

        # Default threshold - subclasses can override.
        ulp_threshold = getattr(self, "ULP_THRESHOLD", 10.0)
        assert (
            max_ulp <= ulp_threshold
        ), f"Max ULP {max_ulp} exceeds threshold {ulp_threshold}"
