# Elementwise Broadcast Add Example

This example demonstrates a custom `ttnn.generic_op` that performs elementwise addition with broadcast on the second input. The broadcast dimension is selectable at runtime and encoded in the compute kernel via preprocessor defines.
## Features

- Block-based processing: Processes tiles in 8×8 blocks (64 tiles per block).
- Broadcast semantics for column, row, and scalar tiles.
- Single-buffered CB sizing for clarity.

## File Structure

```
ttnn_bcast_example/
├── kernels/
│   ├── compute/
│   │   └── bcast_add.cpp            # Compute kernel (A + B with column broadcast)
│   └── dataflow/
│       ├── reader_binary_bcast_cols.cpp # Reader for A with a broadcast tile B
│       └── writer_unary.cpp         # Writes output tensor in blocks
├── test_bcast.py                    # Python test using ttnn.generic_op
├── utils.py                         # Tensor comparison utilities
└── README.md                        # This file
```

## Prerequisites

- TT-Metal (https://github.com/tenstorrent/tt-metal) built with Python bindings;
set the `TT_METAL_ROOT` environment variable to the installed location.
By default, that's the top-level `tt-metal/` directory.

-

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

### Compute Kernel

The compute kernel processes tiles in blocks, applying broadcast in DST-sized chunks:

```
for each block:
    cb_wait_front(cb_in0, block_size)
    cb_wait_front(cb_in1, 1)  # single broadcast tile

    for each DST cycle:
        tile_regs_acquire()
        for i in 0..7:
            add_tiles_bcast_cols(cb_in0, cb_in1, base+i, 0, i)
        tile_regs_commit()
        tile_regs_wait()
        pack tiles to cb_out
        tile_regs_release()
```

## Broadcast Implementation Details

This example implements broadcast by changing both the reader behavior and the compute kernel operations.

Reader changes:
- Input A is read as normal tiles in block order.
- Input B is a single tile (shape [1, 1, 32, 32]) with values placed in the first column, first row, or [0,0].
- The reader loads this single B tile once per core into `cb_in1` and keeps it there while streaming A tiles into `cb_in0`.
- `cb_in1` is sized for one tile, so the reader does not push per-tile B data.

Compute changes:
- The kernel calls `init_bcast<EltwiseBinaryType::ELWADD, BroadcastType::*>` to configure broadcast math and unpack.
- Each A tile is combined with the single B tile using one of:
  - `add_tiles_bcast_cols` for column broadcast.
  - `add_tiles_bcast_rows` for row broadcast.
  - `add_tiles_bcast_scalar` for scalar broadcast.
- These ops interpret B as a broadcast source based on the selected mode, while A remains a normal tile stream.

### Circular Buffer Sizing

CBs must hold an entire block of tiles:
CBs must hold an entire block of tiles:
- Page size: 2KB per tile (32×32 × 2 bytes for bfloat16).
- Single-buffered: 64 tiles × 2KB = 128KB per CB.

The broadcast test constructs `B` as a single tile and compares against:

```
output = A + B[..., :, :1]     # column
output = A + B[..., :1, :]     # row
output = A + B[..., :1, :1]    # scalar
```

## Running the Test

```bash
# From ttnn_bcast_example directory
source $TT_METAL_ROOT/python_env/bin/activate && pytest -svx test_bcast.py
```

Broadcast example:
```bash
source $TT_METAL_ROOT/python_env/bin/activate && pytest -svx test_bcast.py -k "test_bcast_add"
```

Note: No pre-compilation needed. The kernels are compiled JIT (just-in-time) by `ttnn.generic_op` when the test runs.

Test parameters:
- **Tile counts**: 64, 128, 256, 512, 1024
- **Broadcast modes**: col, row, scalar
- **Core modes**: single-core, multi-core

## Notes

- Binary operations: Cannot write to the same register as inputs (use different DST index)
- SFPU operations: Can operate in-place on DST registers
- Register management: Always acquire before use, commit after compute, release after pack
- CB synchronization: Use `cb_wait_front()`, `cb_reserve_back()`, `cb_push_back()`, `cb_pop_front()`
- Barriers: Single `noc_async_read_barrier()` after multiple reads is more efficient
- `copy_tile_init()` must be called before `copy_tile()` for each buffer; cannot interleave inits
- Block processing: Reader/writer operate on full blocks, compute processes DST-sized chunks within
