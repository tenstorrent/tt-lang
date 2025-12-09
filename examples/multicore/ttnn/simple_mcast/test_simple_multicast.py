#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple multicast test to verify semaphore signaling works correctly.
Core 0 multicasts start signal to workers, workers increment done semaphore.
"""

from pathlib import Path
import ttnn

# Get kernel path
KERNEL_PATH = str(Path(__file__).parent / "simple_multicast.cpp")


def test_simple_multicast():
    """Test basic multicast semaphore signaling with 4 cores."""
    device = ttnn.open_device(device_id=0)

    try:
        # Configure 4-core grid: (0,0), (1,0), (2,0), (3,0)
        max_core = ttnn.CoreCoord(3, 0)
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), max_core)])
        num_cores = 4

        print(f"Testing multicast with {num_cores} cores")

        # IMPORTANT: Get physical coordinates for multicast
        print(f"Device type: {type(device)}")
        print(f"Device attributes: {[a for a in dir(device) if not a.startswith('_')][:10]}")

        # Try different methods to access coordinate translation
        coord_method_found = False

        # Method 1: Direct access
        if hasattr(device, 'worker_core_from_logical_core'):
            reduce_core_phys = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
            start_core_phys = device.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
            end_core_phys = device.worker_core_from_logical_core(ttnn.CoreCoord(3, 0))
            print(f"Method 1: Using device.worker_core_from_logical_core")
            coord_method_found = True

        # Method 2: Check for devices attribute (MeshDevice might have this)
        elif hasattr(device, 'devices'):
            print(f"Method 2: Trying device.devices")
            devices_list = device.devices
            if devices_list and hasattr(devices_list[0], 'worker_core_from_logical_core'):
                underlying_dev = devices_list[0]
                reduce_core_phys = underlying_dev.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
                start_core_phys = underlying_dev.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
                end_core_phys = underlying_dev.worker_core_from_logical_core(ttnn.CoreCoord(3, 0))
                coord_method_found = True

        # Method 3: Try getting device from mesh
        elif hasattr(device, 'get_device'):
            print(f"Method 3: Trying device.get_device()")
            underlying_dev = device.get_device(0)
            if hasattr(underlying_dev, 'worker_core_from_logical_core'):
                reduce_core_phys = underlying_dev.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
                start_core_phys = underlying_dev.worker_core_from_logical_core(ttnn.CoreCoord(0, 0))
                end_core_phys = underlying_dev.worker_core_from_logical_core(ttnn.CoreCoord(3, 0))
                coord_method_found = True

        if coord_method_found:
            print(f"Using virtual coordinates from device API:")
            print(f"  Reduce core: ({reduce_core_phys.x}, {reduce_core_phys.y})")
            print(f"  Multicast range: ({start_core_phys.x}, {start_core_phys.y}) to ({end_core_phys.x}, {end_core_phys.y})")
        else:
            # Hardcode virtual coordinates for Blackhole
            # Based on DPRINT output, physical (1,2)-(4,2)
            # For Blackhole, virtual == physical for worker cores
            print("WARNING: Could not find worker_core_from_logical_core method")
            print("Using hardcoded virtual coordinates for Blackhole device")
            reduce_core_phys = ttnn.CoreCoord(1, 2)  # Virtual coords for logical (0,0)
            start_core_phys = ttnn.CoreCoord(1, 2)   # Virtual coords for logical (0,0)
            end_core_phys = ttnn.CoreCoord(4, 2)     # Virtual coords for logical (3,0)
            print(f"  Reduce core (virtual): ({reduce_core_phys.x}, {reduce_core_phys.y})")
            print(f"  Multicast range (virtual): ({start_core_phys.x}, {start_core_phys.y}) to ({end_core_phys.x}, {end_core_phys.y})")

        # Create semaphores
        start_sem_descriptor = ttnn.SemaphoreDescriptor(
            core_type=ttnn.CoreType.WORKER,
            core_ranges=core_grid,
            initial_value=0,
        )
        done_sem_descriptor = ttnn.SemaphoreDescriptor(
            core_type=ttnn.CoreType.WORKER,
            core_ranges=core_grid,
            initial_value=0,
        )

        # Compile-time args (using logical coordinates - works on this device)
        compile_time_args = [
            0,  # start_sem_idx
            1,  # done_sem_idx
            reduce_core_phys.x,  # reduce_core_x
            reduce_core_phys.y,  # reduce_core_y
            start_core_phys.x,   # mcast_start_x
            start_core_phys.y,   # mcast_start_y
            end_core_phys.x,     # mcast_end_x
            end_core_phys.y,     # mcast_end_y
            num_cores - 1,       # num_dests (number of OTHER cores for loopback multicast)
        ]

        # Runtime args per core
        num_x_cores = max_core.x + 1
        num_y_cores = max_core.y + 1
        runtime_args = [[[] for _ in range(num_y_cores)] for _ in range(num_x_cores)]

        core_id = 0
        for x in range(num_x_cores):
            for y in range(num_y_cores):
                runtime_args[x][y] = [core_id]
                core_id += 1

        # Kernel descriptor
        kernel_descriptor = ttnn.KernelDescriptor(
            kernel_source=KERNEL_PATH,
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid,
            compile_time_args=compile_time_args,
            runtime_args=runtime_args,
            defines=[],
            config=ttnn.ReaderConfigDescriptor(),
        )

        # Program descriptor
        program_descriptor = ttnn.ProgramDescriptor(
            kernels=[kernel_descriptor],
            semaphores=[start_sem_descriptor, done_sem_descriptor],
            cbs=[],
        )

        # Create dummy input/output tensors (generic_op requires at least one of each)
        import torch
        dummy_data = torch.zeros([1, 1], dtype=torch.bfloat16)
        input_tensor = ttnn.from_torch(
            dummy_data,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        output_tensor = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1]),
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            device,
            ttnn.DRAM_MEMORY_CONFIG,
        )

        print("Executing multicast test...")
        result = ttnn.generic_op([input_tensor, output_tensor], program_descriptor)
        print("Kernel launched, waiting for completion...")

        # Force synchronization to ensure kernel actually completes
        ttnn.synchronize_device(device)
        print("SUCCESS: Multicast test completed without hanging!")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    test_simple_multicast()
