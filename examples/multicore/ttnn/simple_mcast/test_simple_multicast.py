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
        print("NOTE: Kernel computes multicast coordinates dynamically using my_x[0]/my_y[0]")

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

        # Compile-time args (kernel computes coordinates dynamically)
        compile_time_args = [
            0,  # start_sem_idx
            1,  # done_sem_idx
            num_cores,  # num_cores
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
