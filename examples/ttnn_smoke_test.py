# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Smoke test - absolute minimum TTNN interop kernel.

Just takes an input tensor and does nothing - tests that the runtime
can dispatch a kernel with TTNN tensor inputs without hanging.
"""

from ttlang.d2m_api import *
import torch

try:
    import ttnn
except ImportError:
    print("TTNN not available - this example requires ttnn to be installed")
    exit(0)


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1)])
def noop_kernel(inp, out):
    """Kernel that does absolutely nothing - just tests dispatch."""

    @compute()
    def noop_compute(inp_cb: CircularBuffer, out_cb: CircularBuffer):
        # Do nothing at all
        pass

    @datamovement()
    def noop_dm(inp_cb: CircularBuffer, out_cb: CircularBuffer):
        # Do nothing at all
        pass

    return Program(noop_compute, noop_dm)(inp, out)


if __name__ == "__main__":
    print("=" * 60)
    print("TTNN Smoke Test - Noop Kernel")
    print("=" * 60)

    # Open device
    device = ttnn.open_device(device_id=0)

    try:
        # Create torch tensors
        inp_torch = torch.full((32, 32), 1.0, dtype=torch.bfloat16)
        out_torch = torch.full((32, 32), 0.0, dtype=torch.bfloat16)

        # Convert to TTNN tensors on device
        inp = ttnn.from_torch(
            inp_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        print(f"\ninp shape: {inp.shape}, dtype: {inp.dtype}")
        print(f"out shape: {out.shape}, dtype: {out.dtype}")

        # Run the noop kernel
        print("\n=== RUNNING NOOP KERNEL ===")
        noop_kernel(inp, out)
        print("=== KERNEL COMPLETED ===")

        print("\nSmoke test PASSED - kernel dispatched and returned successfully")

    finally:
        ttnn.close_device(device)
