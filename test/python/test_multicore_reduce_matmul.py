# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multicore gather tests - 2, 3, and 4 core versions.

Tests the pattern where workers read their tiles, send to Core 0, which sums them.
Uses CBs: inp_cb for tensor read, gather_cb for pipe ops, out_cb for tensor write,
and acc_cb for intermediate accumulation when summing multiple tiles.
"""

# REQUIRES: ttnn
# UNSUPPORTED: system-darwin
# RUN: %python -m pytest %s -v

import pytest
import torch

ttnn = pytest.importorskip("ttnn", exc_type=ImportError)

from ttlang_test_utils import to_l1

import ttl


# =============================================================================
# 2-core gather (working baseline)
# =============================================================================

@ttl.kernel(grid=(2, 1))
def gather_2core(inp, out):
    """Core 1 sends its tile to Core 0."""
    pipe = ttl.Pipe(src=(1, 0), dst=(0, 0))

    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)
    gather_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        x, y = ttl.core(dims=2)
        if x == 0:
            # Core 0: gather_cb (received) -> out_cb
            inp_tile = gather_cb.wait()
            out_tile = out_cb.reserve()
            out_tile.store(ttl.math.abs(inp_tile))
            gather_cb.pop()
            out_cb.push()
        else:
            # Core 1: inp_cb -> gather_cb (for sending)
            inp_tile = inp_cb.wait()
            out_tile = gather_cb.reserve()
            out_tile.store(ttl.math.abs(inp_tile))
            inp_cb.pop()
            gather_cb.push()

    @ttl.datamovement()
    def dm_read():
        # Core 1: read tile into inp_cb
        with pipe.if_src():
            inp_blk = inp_cb.reserve()
            tx_read = ttl.copy(inp[0, 1], inp_blk)
            tx_read.wait()
            inp_cb.push()

    @ttl.datamovement()
    def dm_write():
        # Core 1 (src): send gather_cb via pipe
        with pipe.if_src():
            gather_blk = gather_cb.wait()
            tx_send = ttl.copy(gather_blk, pipe)
            tx_send.wait()
            gather_cb.pop()

        # Core 0 (dst): receive into gather_cb
        with pipe.if_dst():
            gather_blk = gather_cb.reserve()
            tx_recv = ttl.copy(pipe, gather_blk)
            tx_recv.wait()
            gather_cb.push()

        # Core 0: write out_cb to output
        x, y = ttl.core(dims=2)
        if x == 0:
            out_blk = out_cb.wait()
            tx_write = ttl.copy(out_blk, out[0, 0])
            tx_write.wait()
            out_cb.pop()


# =============================================================================
# 3-core gather (Core 0 receives from Core 1 and Core 2, sums them)
# =============================================================================


@ttl.kernel(grid=(3, 1))
def gather_3core(inp, out):
    """Cores 1,2 send tiles to Core 0 which sums them."""
    pipe1 = ttl.Pipe(src=(1, 0), dst=(0, 0))
    pipe2 = ttl.Pipe(src=(2, 0), dst=(0, 0))

    inp_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(1, 1), buffer_factor=2)
    gather_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=4)
    # Intermediate CB for partial accumulation (compute-only, not used by dm)
    acc_cb = ttl.make_circular_buffer_like(inp, shape=(1, 1), buffer_factor=2)

    @ttl.compute()
    def compute():
        x, y = ttl.core(dims=2)
        if x == 0:
            # Core 0: accumulate tiles one at a time using intermediate acc_cb
            # Pattern: wait-pop-wait-pop with acc_cb holding partial result

            # First tile -> store to acc_cb
            t1 = gather_cb.wait()
            acc_tile = acc_cb.reserve()
            acc_tile.store(ttl.math.abs(t1))
            gather_cb.pop()
            acc_cb.push()

            # Second tile -> add to acc_cb, store to out_cb
            t2 = gather_cb.wait()
            acc = acc_cb.wait()
            out_tile = out_cb.reserve()
            out_tile.store(ttl.math.abs(acc) + ttl.math.abs(t2))
            gather_cb.pop()
            acc_cb.pop()
            out_cb.push()
        else:
            # Workers: inp_cb -> gather_cb (for sending)
            inp_tile = inp_cb.wait()
            out_tile = gather_cb.reserve()
            out_tile.store(ttl.math.abs(inp_tile))
            inp_cb.pop()
            gather_cb.push()

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.core(dims=2)
        # Workers: read their tile into inp_cb
        if x == 1:
            inp_blk = inp_cb.reserve()
            tx_read = ttl.copy(inp[0, 1], inp_blk)
            tx_read.wait()
            inp_cb.push()
        elif x == 2:
            inp_blk = inp_cb.reserve()
            tx_read = ttl.copy(inp[0, 2], inp_blk)
            tx_read.wait()
            inp_cb.push()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)

        # Core 1: send via pipe1
        if x == 1:
            gather_blk = gather_cb.wait()
            tx_send = ttl.copy(gather_blk, pipe1)
            tx_send.wait()
            gather_cb.pop()

        # Core 2: send via pipe2
        elif x == 2:
            gather_blk = gather_cb.wait()
            tx_send = ttl.copy(gather_blk, pipe2)
            tx_send.wait()
            gather_cb.pop()

        # Core 0: receive from both pipes (must complete each before starting next)
        elif x == 0:
            with gather_cb.reserve() as blk1:
                tx_recv1 = ttl.copy(pipe1, blk1)
                tx_recv1.wait()
            with gather_cb.reserve() as blk2:
                tx_recv2 = ttl.copy(pipe2, blk2)
                tx_recv2.wait()

            # Write output
            out_blk = out_cb.wait()
            tx_write = ttl.copy(out_blk, out[0, 0])
            tx_write.wait()
            out_cb.pop()


# =============================================================================
# 4-core gather (Core 0 receives from Cores 1, 2, 3 and sums them)
# =============================================================================


@ttl.kernel(grid=(4, 1))
def gather_4core(inp, out):
    """All 4 cores read 4x4 tile blocks, cores 1-3 send to Core 0 which sums all 4."""
    pipe1 = ttl.Pipe(src=(1, 0), dst=(0, 0))
    pipe2 = ttl.Pipe(src=(2, 0), dst=(0, 0))
    pipe3 = ttl.Pipe(src=(3, 0), dst=(0, 0))

    inp_cb = ttl.make_circular_buffer_like(inp, shape=(4, 4), buffer_factor=2)
    out_cb = ttl.make_circular_buffer_like(out, shape=(4, 4), buffer_factor=2)
    gather_cb = ttl.make_circular_buffer_like(inp, shape=(4, 4), buffer_factor=6)
    acc_cb = ttl.make_circular_buffer_like(inp, shape=(4, 4), buffer_factor=2)

    @ttl.compute()
    def compute():
        x, y = ttl.core(dims=2)
        if x == 0:
            # Core 0: start with own block, then accumulate 3 gathered blocks

            # Own block -> store to acc_cb
            t0 = inp_cb.wait()
            acc_tile = acc_cb.reserve()
            acc_tile.store(ttl.math.abs(t0))
            inp_cb.pop()
            acc_cb.push()

            # First gathered block -> add to acc_cb
            t1 = gather_cb.wait()
            acc1 = acc_cb.wait()
            acc_tile2 = acc_cb.reserve()
            acc_tile2.store(ttl.math.abs(acc1) + ttl.math.abs(t1))
            gather_cb.pop()
            acc_cb.pop()
            acc_cb.push()

            # Second gathered block -> add to acc_cb
            t2 = gather_cb.wait()
            acc2 = acc_cb.wait()
            acc_tile3 = acc_cb.reserve()
            acc_tile3.store(ttl.math.abs(acc2) + ttl.math.abs(t2))
            gather_cb.pop()
            acc_cb.pop()
            acc_cb.push()

            # Third gathered block -> add to acc_cb, store to out_cb
            t3 = gather_cb.wait()
            acc3 = acc_cb.wait()
            out_tile = out_cb.reserve()
            out_tile.store(ttl.math.abs(acc3) + ttl.math.abs(t3))
            gather_cb.pop()
            acc_cb.pop()
            out_cb.push()
        else:
            # Workers: inp_cb -> gather_cb (for sending)
            inp_tile = inp_cb.wait()
            out_tile = gather_cb.reserve()
            out_tile.store(ttl.math.abs(inp_tile))
            inp_cb.pop()
            gather_cb.push()

    @ttl.datamovement()
    def dm_read():
        x, y = ttl.core(dims=2)
        # All cores read their 4x4 tile block into inp_cb
        if x == 0:
            inp_blk = inp_cb.reserve()
            tx_read = ttl.copy(inp[0:4, 0:4], inp_blk)
            tx_read.wait()
            inp_cb.push()
        elif x == 1:
            inp_blk = inp_cb.reserve()
            tx_read = ttl.copy(inp[0:4, 4:8], inp_blk)
            tx_read.wait()
            inp_cb.push()
        elif x == 2:
            inp_blk = inp_cb.reserve()
            tx_read = ttl.copy(inp[0:4, 8:12], inp_blk)
            tx_read.wait()
            inp_cb.push()
        elif x == 3:
            inp_blk = inp_cb.reserve()
            tx_read = ttl.copy(inp[0:4, 12:16], inp_blk)
            tx_read.wait()
            inp_cb.push()

    @ttl.datamovement()
    def dm_write():
        x, y = ttl.core(dims=2)

        # Core 1: send via pipe1
        if x == 1:
            gather_blk = gather_cb.wait()
            tx_send = ttl.copy(gather_blk, pipe1)
            tx_send.wait()
            gather_cb.pop()

        # Core 2: send via pipe2
        elif x == 2:
            gather_blk = gather_cb.wait()
            tx_send = ttl.copy(gather_blk, pipe2)
            tx_send.wait()
            gather_cb.pop()

        # Core 3: send via pipe3
        elif x == 3:
            gather_blk = gather_cb.wait()
            tx_send = ttl.copy(gather_blk, pipe3)
            tx_send.wait()
            gather_cb.pop()

        # Core 0: receive from all 3 pipes (must complete each before starting next)
        elif x == 0:
            with gather_cb.reserve() as blk1:
                tx_recv1 = ttl.copy(pipe1, blk1)
                tx_recv1.wait()
            with gather_cb.reserve() as blk2:
                tx_recv2 = ttl.copy(pipe2, blk2)
                tx_recv2.wait()
            with gather_cb.reserve() as blk3:
                tx_recv3 = ttl.copy(pipe3, blk3)
                tx_recv3.wait()

            # Write output
            out_blk = out_cb.wait()
            tx_write = ttl.copy(out_blk, out[0:4, 0:4])
            tx_write.wait()
            out_cb.pop()


class TestGather:
    """Test gather patterns."""

    def test_gather_2core(self, device):
        """Core 1 sends tile (value 2.0) to Core 0."""
        inp_torch = torch.zeros((32, 64), dtype=torch.bfloat16)
        inp_torch[:, 0:32] = 1.0   # Core 0's tile (not used)
        inp_torch[:, 32:64] = 2.0  # Core 1's tile

        out_torch = torch.zeros((32, 64), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        gather_2core(inp, out)
        result = ttnn.to_torch(out)

        print(f"\n=== 2-Core Gather Test ===")
        print(f"Expected: 2.0")
        print(f"Got: {result[0, 0].item()}")

        assert result[0, 0].item() == pytest.approx(2.0, rel=0.01)

    def test_gather_3core(self, device):
        """Cores 1,2 send tiles to Core 0: sum = 10 + 100 = 110."""
        inp_torch = torch.zeros((32, 96), dtype=torch.bfloat16)
        inp_torch[:, 0:32] = 1.0      # Core 0's tile (not used)
        inp_torch[:, 32:64] = 10.0    # Core 1's tile
        inp_torch[:, 64:96] = 100.0   # Core 2's tile

        out_torch = torch.zeros((32, 96), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        gather_3core(inp, out)
        result = ttnn.to_torch(out)

        print(f"\n=== 3-Core Gather Test ===")
        print(f"Expected: 110.0 (10 + 100)")
        print(f"Got: {result[0, 0].item()}")

        assert result[0, 0].item() == pytest.approx(110.0, rel=0.01)

    def test_gather_4core(self, device):
        """All 4 cores contribute 4x4 tile blocks: sum = 1 + 10 + 100 + 1000 = 1111."""
        # Each core has 4x4 tiles = 128x128 elements
        # Total input: 128 rows x 512 cols (4 cores x 128 cols each)
        # Use distinct values to identify which cores contribute to each tile
        inp_torch = torch.zeros((128, 512), dtype=torch.bfloat16)
        inp_torch[:, 0:128] = 1.0       # Core 0's 4x4 block
        inp_torch[:, 128:256] = 10.0    # Core 1's 4x4 block
        inp_torch[:, 256:384] = 100.0   # Core 2's 4x4 block
        inp_torch[:, 384:512] = 1000.0  # Core 3's 4x4 block

        out_torch = torch.zeros((128, 512), dtype=torch.bfloat16)

        inp = to_l1(inp_torch, device)
        out = to_l1(out_torch, device)

        gather_4core(inp, out)
        result = ttnn.to_torch(out)

        print(f"\n=== 4-Core Gather Test (4x4 blocks) ===")
        print(f"Expected: 1111.0 (1 + 10 + 100 + 1000)")
        print(f"Core 0 = 1, Core 1 = 10, Core 2 = 100, Core 3 = 1000")

        # Print corner of each tile in the 4x4 output
        print("Output block (corner of each tile):")
        for r in range(4):
            row = [f"{result[r*32, c*32].item():.0f}" for c in range(4)]
            print(f"  Tile row {r}: {row}")

        # Check multiple positions in the output block
        assert result[0, 0].item() == pytest.approx(1111.0, rel=0.01)
        assert result[64, 64].item() == pytest.approx(1111.0, rel=0.01)
        assert result[127, 127].item() == pytest.approx(1111.0, rel=0.01)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
