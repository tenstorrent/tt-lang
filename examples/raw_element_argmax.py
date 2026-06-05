# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Row-scan argmax using raw_element_read/write.

Demonstrates per-element L1 access in a data movement thread to find the
maximum value in a tile row. The pattern mirrors the element-scanning
approach used in tt-metal argmax kernels:

  1. Read initial element as the running maximum.
  2. Scan remaining elements with raw_element_read.
  3. Compare each element against the running max via ``>``.
  4. Update the running max when a larger value is found.
  5. Write the final maximum to the output tile via raw_element_write.

The comparison lowers through the scalar cmpf pipeline:
  Python ``>`` -> arith.cmpf ogt -> ttkernel.float32_greater
"""

import torch
import ttl
import ttnn


@ttl.operation(grid=(1, 1))
def row_argmax(inp: ttnn.Tensor, out: ttnn.Tensor) -> None:
    """Find the maximum value in row 0 of the input tile.

    Writes the maximum to output[0, 0].
    """
    inp_dfb = ttl.make_dataflow_buffer_like(inp, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        pass

    @ttl.datamovement()
    def dm_read():
        with inp_dfb.reserve() as blk:
            tx = ttl.copy(inp[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with inp_dfb.wait() as rblk:
            with out_dfb.reserve() as wblk:
                max_val = ttl.raw_element_read(rblk, 0, 0)
                for c in range(1, 32):
                    val = ttl.raw_element_read(rblk, 0, c)
                    if val > max_val:
                        max_val = val
                ttl.raw_element_write(wblk, 0, 0, max_val)
                tx = ttl.copy(wblk, out[0, 0])
                tx.wait()


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        inp_torch = torch.randn(32, 32, dtype=torch.float32)
        expected_max = inp_torch[0, :].max().item()

        inp = ttnn.from_torch(
            inp_torch,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        out = ttnn.from_torch(
            torch.zeros(32, 32, dtype=torch.float32),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        row_argmax(inp, out)

        result = ttnn.to_torch(out).float()
        actual_max = result[0, 0].item()

        assert (
            abs(actual_max - expected_max) < 1e-5
        ), f"Mismatch: got {actual_max}, expected {expected_max}"
        print(f"PASSED! Row max = {actual_max:.6f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
