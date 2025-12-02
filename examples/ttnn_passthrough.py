# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Passthrough test - reads data from input and writes to output.

Slightly more complex than smoke test - actually moves data through
the kernel pipeline (DMA read -> CB -> compute copy -> CB -> DMA write).
"""

from ttlang.d2m_api import *
import torch

try:
    import ttnn
except ImportError:
    print("TTNN not available - this example requires ttnn to be installed")
    exit(0)


@pykernel_gen(grid=(1, 1), block_factors=[(1, 1), (1, 1)])
def passthrough_kernel(inp, out):
    """Kernel that copies input to output - tests data movement.

    Full flow for TTNN interop:
        DRAM(inp) --[reader DMA]--> inp_cb --[compute copy]--> out_cb --[writer DMA]--> DRAM(out)
    """
    inp_accessor = TensorAccessor(inp)
    out_accessor = TensorAccessor(out)

    @compute()
    def copy_compute(inp_cb: CircularBuffer, out_cb: CircularBuffer):
        # Wait for input data from reader DMA
        tile = inp_cb.wait()
        # Reserve output buffer
        o = out_cb.reserve()
        # Copy input to output (identity operation through DST register)
        o.store(tile)
        # Signal done - release input, push output for writer DMA
        inp_cb.pop()
        out_cb.push()

    @datamovement()
    def reader_dm(inp_cb: CircularBuffer, out_cb: CircularBuffer):
        # DMA read: DRAM(inp) -> inp_cb (L1)
        shard = inp_cb.reserve()
        tx = dma(inp_accessor[0, 0], shard)
        tx.wait()
        inp_cb.push()  # Signal to compute that data is ready

    @datamovement()
    def writer_dm(inp_cb: CircularBuffer, out_cb: CircularBuffer):
        # DMA write: out_cb (L1) -> DRAM(out)
        shard = out_cb.wait()  # Wait for compute to produce output
        tx = dma(shard, out_accessor[0, 0])
        tx.wait()
        out_cb.pop()  # Release the CB slot

    return Program(copy_compute, reader_dm, writer_dm)(inp, out)


if __name__ == "__main__":
    print("=" * 60)
    print("TTNN Passthrough Test - Copy Input to Output")
    print("=" * 60)

    # Open device
    device = ttnn.open_device(device_id=0)

    try:
        # Create torch tensors with recognizable values
        inp_torch = torch.full((32, 32), 42.0, dtype=torch.bfloat16)
        out_torch = torch.full((32, 32), -999.0, dtype=torch.bfloat16)

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
        print(f"inp value: {inp_torch[0, 0].item()}")
        print(f"out initial: {out_torch[0, 0].item()}")

        # Run the passthrough kernel
        print("\n=== RUNNING PASSTHROUGH KERNEL ===")
        passthrough_kernel(inp, out)
        print("=== KERNEL COMPLETED ===")

        # Copy result back to host for verification
        out_result = ttnn.to_torch(out)

        print(f"\nout after kernel: {out_result[0, 0].item()}")

        # Verify results
        if torch.allclose(out_result.float(), inp_torch.float(), rtol=1e-2, atol=1e-2):
            print("\nPassthrough test PASSED - output matches input")
        else:
            print(f"\nPassthrough test FAILED - output does not match input")
            print(f"  inp[0,0] = {inp_torch[0, 0].item()}")
            print(f"  out[0,0] = {out_result[0, 0].item()}")

    finally:
        ttnn.close_device(device)
