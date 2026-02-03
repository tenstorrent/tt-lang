---
description: Export TT-Lang kernel to TT-Metal C++ kernel code
argument-hint: <kernel-file>
---

## Tools Available

```bash
run-test.sh /path/to/kernel.py    # Run kernel on VM simulator (ONLY way to test)
copy-file.sh /path/to/file.py     # Copy a file to the VM
```

**Reading VM logs (output is saved, not streamed):**
```bash
limactl shell ttsim -- cat /tmp/ttlang_test_output.log        # Full log
limactl shell ttsim -- tail -100 /tmp/ttlang_test_output.log  # Last 100 lines
limactl shell ttsim -- grep -i "error" /tmp/ttlang_test_output.log
```

## Task

Export a TT-Lang kernel to standalone TT-Metal C++ code with a Python entry point using `ttnn.generic_op`. The primary goal is a working, correct kernel that can run independently of ttlang.

## Input

$ARGUMENTS

## Priorities

**P1 - Must Work**: Generate correct C++ kernels and a working Python runner
**P2 - Readability**: Rename generated variables (v1 -> input_lhs), collapse redundant casts
**P3 - Low Priority**: Use clearer APIs only if huge clarity win with zero risk

**Non-goals**: Performance optimizations, architectural changes, risky transformations

## Process

### Step 1: Compile the TT-Lang Kernel

Run the kernel with environment variables to capture C++ output:

```bash
cd /path/to/tt-lang
source build/env/activate

# Run the kernel - C++ is printed to stdout and written to /tmp
python $ARGUMENTS 2>&1 | tee /tmp/kernel_output.log
```

The compiler automatically:
- Prints each kernel's C++ to stdout with headers like `=== kernel_name kernel written to /tmp/... ===`
- Writes files to `/tmp/$USER/ttlang_kernel_{name}_{hash}.cpp`

### Step 2: Extract the Three Kernel Files

TTLang generates exactly 3 kernels for `ttnn.generic_op`:
1. **Compute kernel** - runs on Tensix FPU, performs math operations
2. **Reader kernel** - data movement, reads from DRAM/L1 to circular buffers
3. **Writer kernel** - data movement, writes from circular buffers to DRAM/L1

Extract each kernel from the output. They will look like:

**Compute kernel structure:**
```cpp
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_binary.h"
// ... other compute includes

namespace NAMESPACE {
void kernel_main() {
    // CB indices from compile-time args
    // cb_wait_front, cb_reserve_back for synchronization
    // tile_regs_acquire/commit/wait/release for DST register lifecycle
    // Compute operations (add_binary_tile, matmul_tiles, etc.)
    // pack_tile to write results
    // cb_pop_front, cb_push_back to signal completion
}
}
void MAIN { kernel_main(); }
```

**Reader kernel structure:**
```cpp
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // TensorAccessorArgs and TensorAccessor for DRAM addressing
    // cb_reserve_back to get write space
    // noc_async_read_tile to read from DRAM
    // noc_async_read_barrier to wait for completion
    // cb_push_back to signal data ready
}
```

**Writer kernel structure:**
```cpp
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // TensorAccessorArgs and TensorAccessor for DRAM addressing
    // cb_wait_front to wait for compute output
    // noc_async_write_tile to write to DRAM
    // noc_async_write_barrier to wait for completion
    // cb_pop_front to signal buffer consumed
}
```

### Step 3: Create the Output Directory Structure

```
my_kernel/
├── kernels/
│   ├── compute.cpp      # Compute kernel
│   ├── reader.cpp       # Data movement reader
│   └── writer.cpp       # Data movement writer
└── run_kernel.py        # Python entry point
```

### Step 4: Write the Python Entry Point

Create `run_kernel.py` using `ttnn.generic_op`. This is the template - fill in values from the original TTLang kernel:

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

def run_kernel():
    device = ttnn.open_device(device_id=0)

    try:
        # =================================================================
        # 1. CREATE TENSORS - match shapes/dtypes from original kernel
        # =================================================================
        # Example for element-wise add of 32x32 tiles:
        lhs_torch = torch.randn((32, 32), dtype=torch.bfloat16)
        rhs_torch = torch.randn((32, 32), dtype=torch.bfloat16)
        out_torch = torch.zeros((32, 32), dtype=torch.bfloat16)

        # Convert to TTNN tensors - MUST be tilized and interleaved
        memory_config = ttnn.L1_MEMORY_CONFIG  # or ttnn.DRAM_MEMORY_CONFIG

        lhs = ttnn.from_torch(
            lhs_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )
        rhs = ttnn.from_torch(
            rhs_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )
        out = ttnn.from_torch(
            out_torch,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

        # =================================================================
        # 2. CONFIGURE CIRCULAR BUFFERS
        # =================================================================
        # CB indices must match get_compile_time_arg_val(N) in kernels
        # Standard indices: 0, 1 for inputs, 16 for output (or 2 for simple cases)

        TILE_SIZE = 32
        dtype_size = 2  # bfloat16 = 2 bytes
        cb_page_size = dtype_size * TILE_SIZE * TILE_SIZE  # 2048 bytes per tile
        buffer_factor = 2  # Double buffering

        # Core grid - must match original kernel's grid=(rows, cols)
        grid_rows, grid_cols = 1, 1  # Single core example
        core_start = ttnn.CoreCoord(0, 0)
        core_end = ttnn.CoreCoord(grid_cols - 1, grid_rows - 1)
        core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core_start, core_end)])

        # CB format descriptors - one per circular buffer
        lhs_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=0,
            data_format=ttnn.bfloat16,
            page_size=cb_page_size,
        )
        rhs_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=1,
            data_format=ttnn.bfloat16,
            page_size=cb_page_size,
        )
        out_cb_format = ttnn.CBFormatDescriptor(
            buffer_index=2,  # Or 16 for output CB
            data_format=ttnn.bfloat16,
            page_size=cb_page_size,
        )

        # CB descriptors with total size
        cb_total_size = buffer_factor * cb_page_size

        lhs_cb = ttnn.CBDescriptor(
            total_size=cb_total_size,
            core_ranges=core_grid,
            format_descriptors=[lhs_cb_format],
        )
        rhs_cb = ttnn.CBDescriptor(
            total_size=cb_total_size,
            core_ranges=core_grid,
            format_descriptors=[rhs_cb_format],
        )
        out_cb = ttnn.CBDescriptor(
            total_size=cb_total_size,
            core_ranges=core_grid,
            format_descriptors=[out_cb_format],
        )

        # =================================================================
        # 3. CONFIGURE KERNELS
        # =================================================================

        # Compile-time args for reader: TensorAccessorArgs for each input tensor
        reader_ct_args = ttnn.TensorAccessorArgs(lhs).get_compile_time_args()
        reader_ct_args.extend(ttnn.TensorAccessorArgs(rhs).get_compile_time_args())

        # Compile-time args for writer: TensorAccessorArgs for output tensor
        writer_ct_args = ttnn.TensorAccessorArgs(out).get_compile_time_args()

        # Compute compile-time args: CB indices (from original kernel)
        compute_ct_args = [0, 1, 2]  # CB indices used in compute kernel

        # Runtime args structure: [x][y][args] where x=cols, y=rows
        # For single core: [[args_for_core_0_0]]
        reader_rt_args = [[[
            lhs.buffer_address(),
            rhs.buffer_address(),
        ]]]

        writer_rt_args = [[[
            out.buffer_address(),
        ]]]

        compute_rt_args = [[[]]]  # Often empty for compute kernels

        # Kernel descriptors
        reader_kernel = ttnn.KernelDescriptor(
            kernel_source="kernels/reader.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid,
            compile_time_args=reader_ct_args,
            runtime_args=reader_rt_args,
            config=ttnn.ReaderConfigDescriptor(),
        )

        writer_kernel = ttnn.KernelDescriptor(
            kernel_source="kernels/writer.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid,
            compile_time_args=writer_ct_args,
            runtime_args=writer_rt_args,
            config=ttnn.WriterConfigDescriptor(),
        )

        compute_config = ttnn.ComputeConfigDescriptor()
        # Optional: configure math fidelity
        # compute_config.math_fidelity = ttnn.MathFidelity.HiFi4
        # compute_config.fp32_dest_acc_en = True

        compute_kernel = ttnn.KernelDescriptor(
            kernel_source="kernels/compute.cpp",
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=core_grid,
            compile_time_args=compute_ct_args,
            runtime_args=compute_rt_args,
            config=compute_config,
        )

        # =================================================================
        # 4. CREATE PROGRAM AND EXECUTE
        # =================================================================
        program = ttnn.ProgramDescriptor(
            kernels=[reader_kernel, writer_kernel, compute_kernel],
            cbs=[lhs_cb, rhs_cb, out_cb],
            semaphores=[],
        )

        # Execute - tensor list order must match kernel expectations
        output = ttnn.generic_op([lhs, rhs, out], program)

        # =================================================================
        # 5. VERIFY RESULTS
        # =================================================================
        result = ttnn.to_torch(output).to(torch.bfloat16)
        expected = lhs_torch + rhs_torch  # Reference computation

        if torch.allclose(result, expected, rtol=0.01, atol=0.01):
            print("SUCCESS: Output matches expected result")
        else:
            print("MISMATCH: Output differs from expected")
            print(f"Max diff: {(result - expected).abs().max()}")

        return result

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    run_kernel()
```

### Step 5: P2 Beautification (Safe Transformations Only)

Apply ONLY these safe, mechanical transformations:

**1. Rename generated variables:**
```cpp
// Before (generated)
int32_t v1 = ...;
int32_t v2 = ...;

// After (readable)
int32_t input_lhs = ...;
int32_t input_rhs = ...;
```

Note: make sure you read the input tt-lang python kernel (the one that you ran through the compiler). USE THIS TO guide variable names, comments, etc.

**2. Collapse redundant casts:**
```cpp
// Before (generated)
int32_t v1 = (int32_t)(int32_t)get_compile_time_arg_val(0);

// After
int32_t cb_lhs = get_compile_time_arg_val(0);
```

**3. Name CB indices meaningfully:**
```cpp
// Before
cb_wait_front(get_compile_time_arg_val(0), 1);
cb_wait_front(get_compile_time_arg_val(1), 1);

// After (add constexpr at top, keep the call identical)
constexpr uint32_t cb_lhs = 0;
constexpr uint32_t cb_rhs = 1;
constexpr uint32_t cb_out = 2;

cb_wait_front(cb_lhs, 1);
cb_wait_front(cb_rhs, 1);
```

**4. Add comments:**
* Add comments from the input tt-lang program
* Add other comments that describe how the kernel works
* Comments are zero risk and help clarify what's happening. Make your comments succinct. Only comment what's not immediately obvious from the code.

Bad comment (no extra context provided): 
```
// read from CB 0
cb_wait_front(0, 1);
```

Good comment (succinct, provides context not obvious):
```
cb_wait_front(/*lhs*/0, 1);
```

**DO NOT change:**
- Loop structures or bounds
- API calls or their arguments (except variable names)
- Order of operations
- Synchronization patterns (barriers, waits, etc.)
- Any logic or control flow

### Step 6: Copy Files to VM and Test

**The VM is the ONLY place to test.** The exported kernel must be copied to the VM and run there.

#### File Structure on VM

When you copy files to the VM, they go to the user's home directory by default.
The kernel paths in `run_kernel.py` must match where files are on the VM.

```bash
# Copy all files to VM
copy-file.sh my_kernel/run_kernel.py
copy-file.sh my_kernel/kernels/compute.cpp kernels/
copy-file.sh my_kernel/kernels/reader.cpp kernels/
copy-file.sh my_kernel/kernels/writer.cpp kernels/
```

This creates on the VM:
```
/home/<user>/
├── run_kernel.py
└── kernels/
    ├── compute.cpp
    ├── reader.cpp
    └── writer.cpp
```

#### Update Kernel Paths for VM

The `kernel_source` paths in `run_kernel.py` must be **relative to where run_kernel.py is located on the VM**, or use absolute paths:

```python
# Option 1: Relative paths (if run_kernel.py is in same dir as kernels/)
reader_kernel = ttnn.KernelDescriptor(
    kernel_source="kernels/reader.cpp",  # Relative to run_kernel.py location
    ...
)

# Option 2: Absolute paths on VM
reader_kernel = ttnn.KernelDescriptor(
    kernel_source="/home/user/kernels/reader.cpp",
    ...
)
```

#### Run the Test

```bash
# Run the exported kernel on VM
run-test.sh /path/to/my_kernel/run_kernel.py

# Check results
limactl shell ttsim -- tail -50 /tmp/ttlang_test_output.log
```

#### Debugging Failures

If it fails, check:
1. CB indices in compute kernel match the CBDescriptor buffer_index values
2. Tensor order in `ttnn.generic_op([...])` matches kernel expectations
3. Grid dimensions match between Python and original TTLang kernel
4. Compile-time args order matches `get_compile_time_arg_val(N)` usage
5. **Kernel file paths are correct for the VM filesystem**

## Key Data Structures Reference

### ttnn.KernelDescriptor
```python
ttnn.KernelDescriptor(
    kernel_source: str,           # File path or inline source
    source_type: SourceType,      # FILE_PATH or SOURCE_CODE
    core_ranges: CoreRangeSet,    # Which cores run this kernel
    compile_time_args: list,      # Static args (CB indices, tensor metadata)
    runtime_args: list,           # Per-core dynamic args [x][y][args]
    config: ConfigDescriptor,     # Reader/Writer/ComputeConfigDescriptor
)
```

### ttnn.CBDescriptor
```python
ttnn.CBDescriptor(
    total_size: int,              # buffer_factor * page_size
    core_ranges: CoreRangeSet,    # Where this CB exists
    format_descriptors: list,     # [CBFormatDescriptor]
)
```

### ttnn.CBFormatDescriptor
```python
ttnn.CBFormatDescriptor(
    buffer_index: int,            # CB index (0, 1, 2, ... or 16, 24 for outputs)
    data_format: DataType,        # ttnn.bfloat16, ttnn.float32, etc.
    page_size: int,               # Bytes per tile (dtype_size * 32 * 32)
)
```

### Runtime Args Structure
```python
# Structure: runtime_args[x][y] = [args for core at (x, y)]
# x = column (0 to grid_cols-1), y = row (0 to grid_rows-1)

# Single core (1x1 grid):
runtime_args = [[[arg0, arg1, arg2]]]

# 2x2 grid:
runtime_args = [
    [[args_0_0], [args_0_1]],  # column 0: rows 0, 1
    [[args_1_0], [args_1_1]],  # column 1: rows 0, 1
]
```

## Workflow

### Step A: Verify the TT-Lang Kernel Works First

Before exporting, confirm the original TT-Lang kernel runs correctly:
```bash
run-test.sh /path/to/original_kernel.py
limactl shell ttsim -- tail -50 /tmp/ttlang_test_output.log
```

**If the TT-Lang kernel doesn't work: STOP.** Ask the user to fix the original kernel first. Do not proceed with export until the source kernel is working.

### Step B: Extract and Verify Generated C++

The TT-Lang compiler outputs working C++ to `/tmp/`. Extract the three kernel files and verify they work with `ttnn.generic_op` before any beautification.

### Step C: Beautify (Working > Beautiful)

Apply P2 beautification (rename variables, collapse casts, add comments). After each change:
1. Copy updated files to VM
2. Run test
3. If it breaks, **revert the change** - keep what works

**A working kernel with ugly variable names is better than a broken kernel with nice names.**

### Step D: Verify File Locations on VM

Use `limactl` to check file structure on VM:
```bash
limactl shell ttsim -- ls -la ~/kernels/          # Check kernel files exist
limactl shell ttsim -- pwd                         # Confirm working directory
limactl shell ttsim -- cat ~/kernels/compute.cpp  # Verify file contents
```

This helps you set correct paths in the KernelDescriptor.

## Output

Write to the output directory:
1. `kernels/compute.cpp` - Beautified compute kernel
2. `kernels/reader.cpp` - Beautified reader kernel
3. `kernels/writer.cpp` - Beautified writer kernel
4. `run_kernel.py` - Working Python entry point with correct VM paths

**You MUST:**
1. Verify the original TT-Lang kernel works before starting
2. Copy all files to VM using `copy-file.sh`
3. Run `run-test.sh` on the exported kernel
4. Read the log and confirm it runs correctly
5. Only mark complete after successful VM execution
