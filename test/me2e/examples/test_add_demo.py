# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Demonstration test for the add operation.

Shows the full workflow: MLIR generation → compiler pass → verification.
No hardware execution required.
"""

import subprocess
import tempfile
import os

import torch

from ..config import E2EConfig
from ..builder.ttl_builder import build_ttl_module
from ..ops import OP_TORCH_MAP


class TestAddOperation:
    """Complete demonstration of add operation testing."""

    def test_add_mlir_generation(self):
        """Verify add operation generates correct MLIR."""
        config = E2EConfig(grid_shape=(2, 2), dtype=torch.bfloat16)

        # Create input tensors.
        inputs = [torch.rand(config.tensor_shape, dtype=config.dtype) for _ in range(2)]

        # Build TTL module.
        module = build_ttl_module("add", 2, config, inputs)
        mlir_str = str(module)

        # Verify structure.
        assert "func.func @compute_add" in mlir_str
        assert "ttl.add" in mlir_str
        assert "ttl.bind_cb" in mlir_str
        assert "ttl.attach_cb" in mlir_str

        print(f"\n✅ Generated MLIR for add operation ({len(mlir_str)} chars)")

    def test_add_lowering_to_compute(self):
        """Verify add lowers to ttl.compute with ttl.tile_add."""
        config = E2EConfig(grid_shape=(2, 2), dtype=torch.bfloat16)
        inputs = [torch.rand(config.tensor_shape, dtype=config.dtype) for _ in range(2)]

        module = build_ttl_module("add", 2, config, inputs)

        # Run through convert-ttl-to-compute pass.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mlir", delete=False) as f:
            f.write(str(module))
            input_path = f.name

        try:
            result = subprocess.run(
                [
                    "ttlang-opt",
                    "--pass-pipeline=builtin.module(func.func(convert-ttl-to-compute))",
                    input_path,
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            output = result.stdout

            # Verify lowering.
            assert "ttl.compute" in output, "Expected ttl.compute region"
            assert "ttl.tile_add" in output, "Expected ttl.tile_add in compute body"
            assert "#map = affine_map" in output, "Expected affine maps"

            print(f"\n✅ Successfully lowered add to ttl.compute with ttl.tile_add")
            print(f"   Output size: {len(output)} chars")

        finally:
            os.unlink(input_path)

    def test_add_torch_reference_exists(self):
        """Verify torch reference function is available."""
        assert "add" in OP_TORCH_MAP, "add should be in OP_TORCH_MAP"

        torch_add = OP_TORCH_MAP["add"]

        # Test it works.
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([4.0, 5.0, 6.0])
        result = torch_add(a, b)

        expected = torch.tensor([5.0, 7.0, 9.0])
        assert torch.allclose(result, expected)

        print(f"\n✅ Torch reference for add works correctly")

    def test_add_different_shapes(self):
        """Verify add works with different grid shapes."""
        for grid_shape in [(1, 1), (2, 2), (4, 4)]:
            config = E2EConfig(grid_shape=grid_shape, dtype=torch.bfloat16)
            inputs = [
                torch.rand(config.tensor_shape, dtype=config.dtype) for _ in range(2)
            ]

            module = build_ttl_module("add", 2, config, inputs)
            mlir_str = str(module)

            # Verify grid shape appears in CB types.
            assert f"[{grid_shape[0]}, {grid_shape[1]}]" in mlir_str

        print(f"\n✅ Add operation works with shapes: 1x1, 2x2, 4x4")

    def test_add_different_dtypes(self):
        """Verify add works with different data types."""
        dtype_map = {
            torch.bfloat16: "bf16",
            torch.float32: "f32",
        }

        for dtype, mlir_dtype in dtype_map.items():
            config = E2EConfig(grid_shape=(2, 2), dtype=dtype)
            inputs = [torch.rand(config.tensor_shape, dtype=dtype) for _ in range(2)]

            module = build_ttl_module("add", 2, config, inputs)
            mlir_str = str(module)

            # Verify dtype appears in tile types.
            assert f"tile<32x32, {mlir_dtype}>" in mlir_str

        print(f"\n✅ Add operation works with dtypes: bf16, f32")


if __name__ == "__main__":
    # Can run directly for quick testing.
    import sys

    test = TestAddOperation()

    try:
        test.test_add_mlir_generation()
        test.test_add_lowering_to_compute()
        test.test_add_torch_reference_exists()
        test.test_add_different_shapes()
        test.test_add_different_dtypes()
        print("\n🎉 All add operation tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error: {e}")
        sys.exit(1)
