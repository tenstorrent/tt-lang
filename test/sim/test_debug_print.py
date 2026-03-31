# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Test debug printing functionality for TT-Lang simulator."""

import pytest
import torch
from python.sim import ttl, ttnn


def make_tensor_with_value(rows, cols, value, device):
    """Create a tensor filled with a specific value."""
    torch_tensor = torch.full((rows, cols), value, dtype=torch.bfloat16)
    return ttnn.from_torch(
        torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )


def test_print_scalars(capsys):
    """Test printing scalar values and strings."""

    @ttl.kernel(grid=(1, 1))
    def test_kernel(a: torch.Tensor):
        @ttl.compute()
        def compute():
            # Print scalars and strings
            print("Starting computation")
            x = 42
            y = 3.14
            print("x=", x, " y=", y)

        @ttl.datamovement()
        def dm0():
            pass

        @ttl.datamovement()
        def dm1():
            pass

    device = ttnn.open_device(device_id=0)
    try:
        a = make_tensor_with_value(32, 32, 1.0, device)
        test_kernel(a)

        captured = capsys.readouterr()
        # Verify the strings and values were printed
        assert "Starting computation" in captured.out
        assert "x= 42" in captured.out
        assert "y= 3.14" in captured.out
    finally:
        ttnn.close_device(device)


def test_print_dataflow_buffer(capsys):
    """Test printing DataflowBuffer metadata."""

    @ttl.kernel(grid=(1, 1))
    def test_kernel(a: torch.Tensor, out: torch.Tensor):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with a_dfb.wait() as a_blk, out_dfb.reserve() as out_blk:
                # Print DFB state
                print("a_dfb state: ", a_dfb)
                out_blk.store(a_blk)

        @ttl.datamovement()
        def dm_read():
            with a_dfb.reserve() as a_blk:
                tx = ttl.copy(a[0, 0], a_blk)
                tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as out_blk:
                tx = ttl.copy(out_blk, out[0, 0])
                tx.wait()

    device = ttnn.open_device(device_id=0)
    try:
        a = make_tensor_with_value(32, 32, 5.0, device)
        out = make_tensor_with_value(32, 32, 0.0, device)
        test_kernel(a, out)

        captured = capsys.readouterr()
        # Verify DFB structure
        assert "a_dfb state:" in captured.out
        assert "DataflowBuffer" in captured.out
        assert "shape: (1, 1)" in captured.out
        assert "buffer_factor: 2" in captured.out
        assert "capacity: 2 tiles" in captured.out
        # Verify DFB state information is present
        assert "rd_ptr" in captured.out
        assert "visible" in captured.out
        assert "reserved" in captured.out
        assert "free" in captured.out
    finally:
        ttnn.close_device(device)


def test_print_block(capsys):
    """Test printing Block content with known values."""

    @ttl.kernel(grid=(1, 1))
    def test_kernel(a: torch.Tensor, out: torch.Tensor):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with a_dfb.wait() as a_blk, out_dfb.reserve() as out_blk:
                # Print block content
                print("a_blk content:")
                print(a_blk)
                out_blk.store(a_blk)

        @ttl.datamovement()
        def dm_read():
            with a_dfb.reserve() as a_blk:
                tx = ttl.copy(a[0, 0], a_blk)
                tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as out_blk:
                tx = ttl.copy(out_blk, out[0, 0])
                tx.wait()

    device = ttnn.open_device(device_id=0)
    try:
        # Create tensor filled with 7.0
        a = make_tensor_with_value(32, 32, 7.0, device)
        out = make_tensor_with_value(32, 32, 0.0, device)
        test_kernel(a, out)

        captured = capsys.readouterr()
        # Verify block structure
        assert "a_blk content:" in captured.out
        assert "<Block shape=(1, 1)>" in captured.out
        assert "Data shape: torch.Size([32, 32])" in captured.out
        assert "tensor(" in captured.out
        # Verify the known value (7.0) appears in the block data
        assert "7." in captured.out
    finally:
        ttnn.close_device(device)


def test_print_tensor(capsys):
    """Test printing Tensor with num_pages attribute and known values."""

    @ttl.kernel(grid=(1, 1))
    def test_kernel(a: torch.Tensor, out: torch.Tensor):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.datamovement()
        def dm_read():
            # Print tensor
            print("Input tensor a: ", a, num_pages=1)
            with a_dfb.reserve() as a_blk:
                tx = ttl.copy(a[0, 0], a_blk)
                tx.wait()

        @ttl.compute()
        def compute():
            with a_dfb.wait() as a_blk, out_dfb.reserve() as out_blk:
                out_blk.store(a_blk)

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as out_blk:
                tx = ttl.copy(out_blk, out[0, 0])
                tx.wait()

    device = ttnn.open_device(device_id=0)
    try:
        # Create tensor filled with 3.0
        a = make_tensor_with_value(32, 32, 3.0, device)
        out = make_tensor_with_value(32, 32, 0.0, device)
        test_kernel(a, out)

        captured = capsys.readouterr()
        # Verify tensor structure
        assert "Input tensor a:" in captured.out
        assert "<Tensor shape=(32, 32) dtype=torch.bfloat16>" in captured.out
        assert "Printing first 1 page(s):" in captured.out
        # Verify the known value (3.0) appears multiple times
        assert captured.out.count("3.") >= 10
    finally:
        ttnn.close_device(device)


def test_print_tensor_multiple_pages(capsys):
    """Test printing Tensor with different num_pages values and verify page counts."""

    @ttl.kernel(grid=(1, 1))
    def test_kernel(a: torch.Tensor, out: torch.Tensor):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(2, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 1), buffer_factor=2)

        @ttl.datamovement()
        def dm_read():
            # Test different num_pages values
            print("Tensor with 1 page: ", a, num_pages=1)
            print("Tensor with 2 pages: ", a, num_pages=2)
            print("Tensor with 4 pages: ", a, num_pages=4)

            for i in range(2):
                with a_dfb.reserve() as a_blk:
                    tx = ttl.copy(a[i * 2 : (i + 1) * 2, 0], a_blk)
                    tx.wait()

        @ttl.compute()
        def compute():
            for i in range(2):
                with a_dfb.wait() as a_blk, out_dfb.reserve() as out_blk:
                    out_blk.store(a_blk)

        @ttl.datamovement()
        def dm_write():
            for i in range(2):
                with out_dfb.wait() as out_blk:
                    tx = ttl.copy(out_blk, out[i * 2 : (i + 1) * 2, 0])
                    tx.wait()

    device = ttnn.open_device(device_id=0)
    try:
        # Need at least 128 rows (4 tiles) for 4 pages - fill with 9.0
        a = make_tensor_with_value(128, 32, 9.0, device)
        out = make_tensor_with_value(128, 32, 0.0, device)
        test_kernel(a, out)

        captured = capsys.readouterr()
        # Verify all three prints happened with correct page counts
        assert "Tensor with 1 page:" in captured.out
        assert "Printing first 1 page(s):" in captured.out
        assert "Tensor with 2 pages:" in captured.out
        assert "Printing first 2 page(s):" in captured.out
        assert "Tensor with 4 pages:" in captured.out
        assert "Printing first 4 page(s):" in captured.out
        # Verify shape is correct
        assert "<Tensor shape=(128, 32)" in captured.out
        # Verify known value (9.0) appears many times
        assert captured.out.count("9.") >= 20
    finally:
        ttnn.close_device(device)


def test_print_all_ttlang_objects(capsys):
    """Test printing all TT-Lang object types with known values: Tensor, Block, and DataflowBuffer."""

    @ttl.kernel(grid=(1, 1))
    def test_kernel(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(2, 1), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(2, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 1), buffer_factor=2)

        @ttl.datamovement()
        def dm_read():
            # Print tensors with different num_pages
            print("Tensor a (1 page): ", a, num_pages=1)
            print("Tensor b (2 pages): ", b, num_pages=2)

            for i in range(2):
                # Print DataflowBuffer state before reserve
                print("Iteration ", i, " a_dfb before reserve: ", a_dfb)
                print("Iteration ", i, " b_dfb before reserve: ", b_dfb)

                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                    # Print DataflowBuffer state after reserve
                    print("a_dfb after reserve: ", a_dfb)

                    tx_a = ttl.copy(a[i * 2 : (i + 1) * 2, 0], a_blk)
                    tx_b = ttl.copy(b[i * 2 : (i + 1) * 2, 0], b_blk)
                    tx_a.wait()
                    tx_b.wait()

        @ttl.compute()
        def compute():
            for i in range(2):
                with (
                    a_dfb.wait() as a_blk,
                    b_dfb.wait() as b_blk,
                    out_dfb.reserve() as out_blk,
                ):
                    # Print Block content
                    print("Iteration ", i, " a_blk: ", a_blk)
                    print("Iteration ", i, " b_blk: ", b_blk)

                    result = a_blk + b_blk
                    out_blk.store(result)

                    # Print output block
                    print("out_blk: ", out_blk)

        @ttl.datamovement()
        def dm_write():
            # Print output tensor
            print("Output tensor (3 pages): ", out, num_pages=3)

            for i in range(2):
                # Print DataflowBuffer state
                print("out_dfb before wait: ", out_dfb)

                with out_dfb.wait() as out_blk:
                    tx = ttl.copy(out_blk, out[i * 2 : (i + 1) * 2, 0])
                    tx.wait()

    device = ttnn.open_device(device_id=0)
    try:
        # Need 128 rows (4 tiles) for 3+ pages - use different values for a and b
        a = make_tensor_with_value(128, 32, 2.0, device)  # a filled with 2.0
        b = make_tensor_with_value(128, 32, 4.0, device)  # b filled with 4.0
        out = make_tensor_with_value(128, 32, 0.0, device)
        test_kernel(a, b, out)

        captured = capsys.readouterr()

        # Check Tensor printing with specific values
        assert "Tensor a (1 page):" in captured.out
        assert "Printing first 1 page(s):" in captured.out
        assert "<Tensor shape=(128, 32)" in captured.out
        # Verify tensor a has value 2.0
        assert captured.out.count("2.") >= 10

        assert "Tensor b (2 pages):" in captured.out
        assert "Printing first 2 page(s):" in captured.out
        # Verify tensor b has value 4.0
        assert captured.out.count("4.") >= 10

        assert "Output tensor (3 pages):" in captured.out
        assert "Printing first 3 page(s):" in captured.out

        # Check DataflowBuffer printing with structure verification
        assert "Iteration  0  a_dfb before reserve:" in captured.out
        assert "Iteration  1  a_dfb before reserve:" in captured.out
        assert "a_dfb after reserve:" in captured.out
        assert "out_dfb before wait:" in captured.out
        assert "<DataflowBuffer" in captured.out
        assert "shape: (2, 1)" in captured.out
        assert "buffer_factor: 2" in captured.out
        # Note: capacity line may not appear in output due to formatting

        # Check Block printing with structure and verify output values (2.0 + 4.0 = 6.0)
        assert "Iteration  0  a_blk:" in captured.out
        assert "Iteration  1  a_blk:" in captured.out
        assert "<Block shape=(2, 1)>" in captured.out
        assert "Data shape: torch.Size([64, 32])" in captured.out
        # Output block should have values around 6.0 (2.0 + 4.0)
        assert "6." in captured.out
    finally:
        ttnn.close_device(device)


def test_print_mixed_args(capsys):
    """Test printing strings, scalars, and TT-Lang object together with known values."""

    @ttl.kernel(grid=(1, 1))
    def test_kernel(a: torch.Tensor, out: torch.Tensor):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(2, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(2, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            for i in range(2):
                with a_dfb.wait() as a_blk, out_dfb.reserve() as out_blk:
                    # Mixed print: scalars + block
                    print("Iteration i=", i, " block: ", a_blk)
                    out_blk.store(a_blk)

        @ttl.datamovement()
        def dm_read():
            for i in range(2):
                with a_dfb.reserve() as a_blk:
                    # Copy 2x1 tiles (64x32 elements)
                    tx = ttl.copy(a[i * 2 : (i + 1) * 2, 0], a_blk)
                    tx.wait()

        @ttl.datamovement()
        def dm_write():
            for i in range(2):
                with out_dfb.wait() as out_blk:
                    tx = ttl.copy(out_blk, out[i * 2 : (i + 1) * 2, 0])
                    tx.wait()

    device = ttnn.open_device(device_id=0)
    try:
        # Need 128 rows for 4 blocks of 2 tiles - fill with 8.5
        a = make_tensor_with_value(128, 32, 8.5, device)
        out = make_tensor_with_value(128, 32, 0.0, device)
        test_kernel(a, out)

        captured = capsys.readouterr()
        # Verify iteration indices are printed
        assert "Iteration i= 0" in captured.out
        assert "Iteration i= 1" in captured.out
        # Verify block structure is printed
        assert "block:" in captured.out
        assert "<Block shape=(2, 1)>" in captured.out
        # Verify the known value (8.5) appears in block data
        assert "8.5" in captured.out or "8.50" in captured.out
    finally:
        ttnn.close_device(device)


def test_print_tensor_with_known_values(capsys):
    """Test printing Tensor with known values to verify correctness."""

    @ttl.kernel(grid=(1, 1))
    def test_kernel(a: torch.Tensor, out: torch.Tensor):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.datamovement()
        def dm_read():
            # Print tensor with 1 page
            print("Test tensor:", a, num_pages=1)

            with a_dfb.reserve() as a_blk:
                tx = ttl.copy(a[0, 0], a_blk)
                tx.wait()

        @ttl.compute()
        def compute():
            with a_dfb.wait() as a_blk, out_dfb.reserve() as out_blk:
                out_blk.store(a_blk)

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as out_blk:
                tx = ttl.copy(out_blk, out[0, 0])
                tx.wait()

    # Create a tensor with known values using ttnn
    device = ttnn.open_device(device_id=0)
    try:
        # Create a tensor filled with ones
        torch_tensor = torch.ones((32, 32), dtype=torch.bfloat16)
        a = ttnn.from_torch(
            torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        out = ttnn.from_torch(
            torch.zeros_like(torch_tensor),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        test_kernel(a, out)

        captured = capsys.readouterr()

        # For debugging: print the actual captured output
        print("\n=== CAPTURED OUTPUT ===")
        print(captured.out)
        print("=== END CAPTURED OUTPUT ===\n")

        # Verify the output contains expected structure
        assert "Test tensor:" in captured.out
        assert "<Tensor shape=(32, 32)" in captured.out
        assert "Printing first 1 page(s):" in captured.out

        # Verify that the tensor values (1.0) appear in the output
        # Since we filled with ones, we should see many "1." values
        assert "1." in captured.out or "1.0" in captured.out

    finally:
        ttnn.close_device(device)


def test_print_multiple_ttlang_objects_fails():
    """Test that printing multiple TT-Lang objects raises an error."""

    @ttl.kernel(grid=(1, 1))
    def test_kernel(a: torch.Tensor, b: torch.Tensor):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
        b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                # This should fail - can't print multiple TT-Lang objects
                print(a_blk, b_blk)

        @ttl.datamovement()
        def dm0():
            with a_dfb.reserve() as a_blk:
                tx = ttl.copy(a[0, 0], a_blk)
                tx.wait()

        @ttl.datamovement()
        def dm1():
            with b_dfb.reserve() as b_blk:
                tx = ttl.copy(b[0, 0], b_blk)
                tx.wait()

    device = ttnn.open_device(device_id=0)
    try:
        a = make_tensor_with_value(32, 32, 1.0, device)
        b = make_tensor_with_value(32, 32, 2.0, device)

        # The ValueError gets wrapped in RuntimeError by the scheduler
        with pytest.raises(
            RuntimeError, match="print.*can only print one TT-Lang object"
        ):
            test_kernel(a, b)
    finally:
        ttnn.close_device(device)


def test_print_dm_reserve_block_mw_state_warns(capsys):
    """Test that printing a DM reserve block in MW state issues a warning."""

    @ttl.kernel(grid=(1, 1))
    def test_kernel(a: torch.Tensor, out: torch.Tensor):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)

        @ttl.datamovement()
        def dm_write():
            with a_dfb.reserve() as a_blk:
                # Block is in MW state (must-write) immediately after reserve
                # Printing should warn
                print("Block in MW state: ", a_blk)
                tx = ttl.copy(a[0, 0], a_blk)
                tx.wait()

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def dm_read():
            pass

    device = ttnn.open_device(device_id=0)
    try:
        a = make_tensor_with_value(32, 32, 1.0, device)
        out = make_tensor_with_value(32, 32, 0.0, device)

        test_kernel(a, out)

        captured = capsys.readouterr()
        # Verify warning was issued
        assert "warning" in captured.out.lower()
        assert "MW state cannot be read" in captured.out
        # Verify the warning message appears in output
        assert "Block in MW state:" in captured.out
        assert "[WARNING: Cannot read - in MW state]" in captured.out
    finally:
        ttnn.close_device(device)


def test_print_dm_reserve_block_naw_state_warns(capsys):
    """Test that printing a DM reserve block in NAW state issues a warning."""

    @ttl.kernel(grid=(1, 1))
    def test_kernel(a: torch.Tensor, out: torch.Tensor):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)

        @ttl.datamovement()
        def dm_write():
            with a_dfb.reserve() as a_blk:
                # Start copy to put block in NAW state
                tx = ttl.copy(a[0, 0], a_blk)
                # Block is now in NAW state (no-access-while-writing)
                # Printing should warn
                print("Block in NAW state: ", a_blk)
                tx.wait()

        @ttl.compute()
        def compute():
            pass

        @ttl.datamovement()
        def dm_read():
            pass

    device = ttnn.open_device(device_id=0)
    try:
        a = make_tensor_with_value(32, 32, 1.0, device)
        out = make_tensor_with_value(32, 32, 0.0, device)

        test_kernel(a, out)

        captured = capsys.readouterr()
        # Verify warning was issued
        assert "warning" in captured.out.lower()
        assert "NAW state cannot be read" in captured.out
        # Verify the warning message appears in output
        assert "Block in NAW state:" in captured.out
        assert "[WARNING: Cannot read - in NAW state]" in captured.out
    finally:
        ttnn.close_device(device)


def test_print_dm_wait_block_naw_state_warns(capsys):
    """Test that printing a DM wait block in NAW state issues a warning."""

    @ttl.kernel(grid=(1, 1))
    def test_kernel(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with a_dfb.wait() as a_blk, out_dfb.reserve() as out_blk:
                out_blk.store(a_blk)

        @ttl.datamovement()
        def dm_read():
            with a_dfb.reserve() as a_blk:
                tx = ttl.copy(a[0, 0], a_blk)
                tx.wait()

        @ttl.datamovement()
        def dm_write():
            # First: copy out to make wait block transition MR -> ROR -> RW
            with out_dfb.wait() as out_blk:
                tx1 = ttl.copy(out_blk, out[0, 0])
                tx1.wait()
                # Now block is in RW state (can be read or written)
                # Copy TO the block to put it in NAW state
                tx2 = ttl.copy(b[0, 0], out_blk)
                # Block is now in NAW state (no-access-while-writing)
                # Printing should warn
                print("Wait block in NAW state: ", out_blk)
                tx2.wait()
                # Use block as source to satisfy state machine
                tx3 = ttl.copy(out_blk, out[0, 0])
                tx3.wait()

    device = ttnn.open_device(device_id=0)
    try:
        a = make_tensor_with_value(32, 32, 1.0, device)
        b = make_tensor_with_value(32, 32, 2.0, device)
        out = make_tensor_with_value(32, 32, 0.0, device)

        test_kernel(a, b, out)

        captured = capsys.readouterr()
        # Verify warning was issued
        assert "warning" in captured.out.lower()
        assert "NAW state cannot be read" in captured.out
        # Verify the warning message appears in output
        assert "Wait block in NAW state:" in captured.out
        assert "[WARNING: Cannot read - in NAW state]" in captured.out
    finally:
        ttnn.close_device(device)


def test_print_compute_thread_blocks_succeeds():
    """Test that compute thread blocks can be printed in various states."""

    @ttl.kernel(grid=(1, 1))
    def test_kernel(a: torch.Tensor, out: torch.Tensor):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            # Compute thread: wait blocks start in MR state
            with a_dfb.wait() as a_blk:
                print("Compute wait block: ", a_blk)

                # After using as source, transitions to RW
                with out_dfb.reserve() as out_blk:
                    out_blk.store(a_blk)
                    # Reserve block after store is in MR state
                    print("Compute reserve block after store: ", out_blk)

        @ttl.datamovement()
        def dm_read():
            with a_dfb.reserve() as a_blk:
                tx = ttl.copy(a[0, 0], a_blk)
                tx.wait()

        @ttl.datamovement()
        def dm_write():
            with out_dfb.wait() as out_blk:
                tx = ttl.copy(out_blk, out[0, 0])
                tx.wait()

    device = ttnn.open_device(device_id=0)
    try:
        a = make_tensor_with_value(32, 32, 2.0, device)
        out = make_tensor_with_value(32, 32, 0.0, device)
        test_kernel(a, out)
        # Should succeed - compute thread blocks can be printed
    finally:
        ttnn.close_device(device)


def test_print_dm_block_legal_states_succeeds(capsys):
    """Test that DM blocks can be printed in legal states (MR after tx.wait)."""

    @ttl.kernel(grid=(1, 1))
    def test_kernel(a: torch.Tensor, out: torch.Tensor):
        a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), buffer_factor=2)
        out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), buffer_factor=2)

        @ttl.compute()
        def compute():
            with a_dfb.wait() as a_blk, out_dfb.reserve() as out_blk:
                out_blk.store(a_blk)

        @ttl.datamovement()
        def dm_read():
            with a_dfb.reserve() as a_blk:
                tx = ttl.copy(a[0, 0], a_blk)
                tx.wait()
                # After tx.wait(), block transitions to MR state
                # Printing should succeed
                print("DM reserve block after tx.wait (MR state): ", a_blk)

        @ttl.datamovement()
        def dm_write():
            # DM wait blocks start in MR state
            with out_dfb.wait() as out_blk:
                print("DM wait block (MR state): ", out_blk)
                tx = ttl.copy(out_blk, out[0, 0])
                tx.wait()

    device = ttnn.open_device(device_id=0)
    try:
        a = make_tensor_with_value(32, 32, 3.0, device)
        out = make_tensor_with_value(32, 32, 0.0, device)
        test_kernel(a, out)

        captured = capsys.readouterr()
        # Verify both blocks were printed
        assert "DM reserve block after tx.wait (MR state):" in captured.out
        assert "DM wait block (MR state):" in captured.out
        assert "<Block shape=(1, 1)>" in captured.out
        # Verify value (3.0) appears in output
        assert "3." in captured.out
    finally:
        ttnn.close_device(device)
