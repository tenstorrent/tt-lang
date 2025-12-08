# Fused Elementwise Operation Example

This example demonstrates how to create a fused element-wise operation that performs three consecutive operations without storing intermediate results:

```
Output = exp(A + B) + C
```

## Features

- Block-based processing: Processes tiles in 8×8 blocks (64 tiles per block).
- Maximum DST utilization: Processes 4 tiles per DST acquire/release cycle (uses all 8 DST registers, which is the max
available for f16 with `dst_full_sync_en=false`; this is the default, not modified in this example).
- No intermediate CB storage: All operations happen in DST registers.
- Reduced memory bandwidth: Only read inputs and write final output (eliminates intermediate writes/reads).

## File Structure

```
ttnn_fused_example/
├── kernels/
│   ├── compute/
│   │   └── fused_elementwise.cpp    # Compute kernel (3 fused ops, block processing)
│   └── dataflow/
│       ├── reader_ternary.cpp        # Single-buffered reader
│       ├── reader_ternary_db.cpp     # Double-buffered reader
│       └── writer_unary.cpp          # Writes output tensor in blocks
├── test_fused_kernel.py         # Python test using ttnn.generic_op
├── utils.py                           # Tensor comparison utilities
└── README.md                          # This file
```

## Prerequisites

- TT-Metal (https://github.com/tenstorrent/tt-metal) built with Python bindings;
set the `TT_METAL_ROOT` environment variable to the installed location.
By default, that's the top-level `tt-metal/` directory.

## How It Works

### JIT Compilation

When `ttnn.generic_op()` is called, TTNN compiles the kernel source files on-the-fly. The `KernelDescriptor` specifies:
- `kernel_source`: Path to the C++ source file.
- `compile_time_args`: Values included in the kernel at compile time (e.g., tensor accessor metadata),
e.g., `TensorAccessorArgs` (metadata about how to access tensors in memory -- strides, shapes, memory layout info).
- `defines`: Preprocessor macros (e.g., `BLOCK_H=8`) passed as `-D` flags to the compiler and available in the C++ sources.
Macros are optional; kernels could use literal constants instead.
- `runtime_args`: Values passed to the kernel at execution time via `get_arg_val<T>()`,
e.g., `start_id` (starting tile index for this core, which is different for each core in multi-core runs).

The compiled kernels are cached, so subsequent runs with the same configuration skip recompilation.

### Block Configuration

Block dimensions are configured via defines passed from Python:
- `BLOCK_H = 8` (tiles per block height)
- `BLOCK_W = 8` (tiles per block width)
- Block size = 64 tiles

Each kernel has `#ifndef` guards providing defaults; the Python test overrides these via `KernelDescriptor.defines`.

### DST Register Usage

With 8 DST registers available and 2 needed per tile (for the chain of binary and unary ops), we process 4 tiles per DST acquire/release cycle:
- Registers 0-3 hold the primary operand (A tiles, then intermediate results)
- Registers 4-7 hold the secondary operand for binary ops (B tiles, then C tiles)

### Compute Kernel

The compute kernel processes tiles in blocks, with inner loops over DST-sized chunks:

```
for each block (64 tiles):
    wait for block from reader
    for each DST cycle (4 tiles):
        tile_regs_acquire()

        # Copy A tiles to DST registers 0-3
        copy_tile_init(cb_in0)
        for i in 0..3: copy_tile(cb_in0, base+i, i)

        # Copy B tiles to DST registers 4-7
        copy_tile_init(cb_in1)
        for i in 0..3: copy_tile(cb_in1, base+i, i+4)

        # A + B → registers 0-3
        add_binary_tile_init()
        for i in 0..3: add_binary_tile(i, i+4, i)

        # exp(A+B) in-place on registers 0-3
        exp_tile_init()
        for i in 0..3: exp_tile(i)

        # Copy C to DST registers 4-7
        copy_tile_init(cb_in2)
        for i in 0..3: copy_tile(cb_in2, base+i, i+4)

        # exp(A+B) + C → registers 0-3
        add_binary_tile_init()
        for i in 0..3: add_binary_tile(i, i+4, i)

        tile_regs_commit()
        tile_regs_wait()
        pack 4 tiles from registers 0-3
        tile_regs_release()

    pop block from input CBs
```

### Circular Buffer Sizing

CBs must hold an entire block of tiles:
- Page size: 2KB per tile (32×32 × 2 bytes for bfloat16)
- Total CB size: 64 tiles × 2KB = 128KB per CB

## Running the Test

```bash
# From ttnn_fused_example directory
source $TT_METAL_ROOT/python_env/bin/activate && pytest -svx test_fused_kernel.py
```

Or with specific tile counts:
```bash
source $TT_METAL_ROOT/python_env/bin/activate && pytest -svx test_fused_kernel.py -k "num_tiles-64"
```

Note: No pre-compilation needed. The kernels are compiled JIT (just-in-time) by `ttnn.generic_op` when the test runs.

Test parameters: 64, 128, 256 tiles × single/double buffering modes.

## Example Output
```plaintext
================================================================= test session starts =================================================================
platform linux -- Python 3.10.12, pytest-8.4.2, pluggy-1.6.0 -- /home/bnorris/tt/tt-metal/python_env/bin/python3
cachedir: .pytest_cache
benchmark: 5.1.0 (defaults: timer=time.perf_counter disable_gc=False min_rounds=5 min_time=0.000005 max_time=1.0 calibration_precision=10 warmup=False warmup_iterations=100000)
rootdir: /home/bnorris/tt/playground/ttnn_fused_example
plugins: xdist-3.8.0, dash-2.15.0, anyio-4.12.0, github-actions-annotate-failures-0.3.0, timeout-2.4.0, split-0.10.0, benchmark-5.1.0
collecting ... 2025-12-07 21:18:51.440 | DEBUG    | ttnn:<module>:77 - Initial ttnn.CONFIG:
Config{cache_path=/home/bnorris/.cache/ttnn,model_cache_path=/home/bnorris/.cache/ttnn/models,tmp_dir=/tmp/ttnn,enable_model_cache=false,enable_fast_runtime_mode=true,throw_exception_on_fallback=false,enable_logging=false,enable_graph_report=false,enable_detailed_buffer_report=false,enable_detailed_tensor_report=false,enable_comparison_mode=false,comparison_mode_should_raise_exception=false,comparison_mode_pcc=0.9999,root_report_path=generated/ttnn/reports,report_name=std::nullopt,std::nullopt}
collected 3 items

test_fused_kernel.py::test_fused_kernel[64] 2025-12-07 21:18:51.528 | info     |             UMD | Starting topology discovery. (topology_discovery.cpp:69)
2025-12-07 21:18:52.044 | info     |             UMD | Established firmware bundle version: 18.12.0 (topology_discovery.cpp:369)
2025-12-07 21:18:52.044 | info     |             UMD | Established ETH FW version: 1.7.0 (topology_discovery_blackhole.cpp:305)
2025-12-07 21:18:52.044 | info     |             UMD | Completed topology discovery. (topology_discovery.cpp:73)
2025-12-07 21:18:52.276 | info     |          Device | Opening user mode device driver (tt_cluster.cpp:211)
2025-12-07 21:18:52.276 | info     |             UMD | Starting topology discovery. (topology_discovery.cpp:69)
2025-12-07 21:18:52.791 | info     |             UMD | Established firmware bundle version: 18.12.0 (topology_discovery.cpp:369)
2025-12-07 21:18:52.791 | info     |             UMD | Established ETH FW version: 1.7.0 (topology_discovery_blackhole.cpp:305)
2025-12-07 21:18:52.791 | info     |             UMD | Completed topology discovery. (topology_discovery.cpp:73)
2025-12-07 21:18:53.023 | info     |             UMD | Starting topology discovery. (topology_discovery.cpp:69)
2025-12-07 21:18:53.549 | info     |             UMD | Established firmware bundle version: 18.12.0 (topology_discovery.cpp:369)
2025-12-07 21:18:53.549 | info     |             UMD | Established ETH FW version: 1.7.0 (topology_discovery_blackhole.cpp:305)
2025-12-07 21:18:53.549 | info     |             UMD | Completed topology discovery. (topology_discovery.cpp:73)
2025-12-07 21:18:53.830 | info     |             UMD | Harvesting masks for chip 0 tensix: 0x0 dram: 0x0 eth: 0x120 pcie: 0x2 l2cpu: 0x0 (cluster.cpp:358)
2025-12-07 21:18:53.982 | info     |             UMD | Initializing iommu for sysmem (size: 0x40000000). (sysmem_manager.cpp:307)
2025-12-07 21:18:53.982 | info     |             UMD | Allocating sysmem without hugepages (size: 0x40000000). (sysmem_manager.cpp:313)
2025-12-07 21:18:54.390 | info     |             UMD | Opening local chip ids/PCIe ids: {0}/[0] and remote chip ids {} (cluster.cpp:202)
2025-12-07 21:18:54.390 | info     |             UMD | IOMMU: enabled (cluster.cpp:178)
2025-12-07 21:18:54.390 | info     |             UMD | KMD version: 2.4.1 (cluster.cpp:181)
2025-12-07 21:18:54.570 | info     |             UMD | Mapped sysmem without hugepages to IOVA 0xfffffffec0000000; NOC address 0x1000000000000000 (sysmem_manager.cpp:361)
Authorization required, but no authorization protocol specified
Authorization required, but no authorization protocol specified
Authorization required, but no authorization protocol specified
2025-12-07 21:18:56.679 | info     |          Fabric | TopologyMapper mapping start (mesh=0): n_log=1, n_phys=1, log_deg_hist={0:1}, phys_deg_hist={0:1} (topology_mapper_utils.cpp:167)
2025-12-07 21:18:56.871 | INFO     | utils:check_tensors_match:27 - PASS | dtype eps=7.81e-03 | max abs diff=9.38e-02 (tol=1.00e-01) | max rel diff=5.00e-01 (tol=5.00e-02)
PASSED
test_fused_kernel.py::test_fused_kernel[128] 2025-12-07 21:18:56.879 | INFO     | utils:check_tensors_match:27 - PASS | dtype eps=7.81e-03 | max abs diff=9.38e-02 (tol=1.00e-01) | max rel diff=5.00e-01 (tol=5.00e-02)
PASSED
test_fused_kernel.py::test_fused_kernel[256] 2025-12-07 21:18:56.889 | INFO     | utils:check_tensors_match:27 - PASS | dtype eps=7.81e-03 | max abs diff=9.38e-02 (tol=1.00e-01) | max rel diff=5.00e-01 (tol=5.00e-02)
PASSED

================================================================== 3 passed in 7.88s ==================================================================
2025-12-07 21:18:58.123 | info     |          Device | Closing user mode device drivers (tt_cluster.cpp:442)
```

## Extending the Example

To create your own fused operation:

1. Modify the compute kernel:
   - Add more operations in the DST register space
   - Chain binary and unary ops as needed
   - Remember: binary ops write to a different DST register
   - SFPU ops can operate in-place

2. Update reader/writer if needed:
   - Add more input CBs for additional operands
   - Adjust TensorAccessorArgs offsets

3. Update the test:
   - Add more input tensors
   - Compute the golden reference in PyTorch
   - Adjust CB descriptors and runtime args

## Notes

- Binary operations: Cannot write to the same register as inputs (use different DST index)
- SFPU operations: Can operate in-place on DST registers
- Register management: Always acquire before use, commit after compute, release after pack
- CB synchronization: Use `cb_wait_front()`, `cb_reserve_back()`, `cb_push_back()`, `cb_pop_front()`
- Barriers: Single `noc_async_read_barrier()` after multiple reads is more efficient
- `copy_tile_init()` must be called before `copy_tile()` for each buffer; cannot interleave inits
- Block processing: Reader/writer operate on full blocks, compute processes DST-sized chunks within
- Performance: Compared to separate operations, 3x fewer DRAM accesses (only 3 reads + 1 write vs 3 reads + 2 intermediate writes + 1 read + 1 write)
