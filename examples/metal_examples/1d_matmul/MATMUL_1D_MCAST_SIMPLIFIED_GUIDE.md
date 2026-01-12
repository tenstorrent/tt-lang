# Simplified 1D Mcast Matmul - 3 Kernel Configuration

This guide documents the simplest 1D multicast matmul configuration that uses exactly 3 kernels (1 reader, 1 writer, 1 compute) for creating a programming example.

## Configuration Requirements

To achieve the 3-kernel configuration, you need:

1. **`in0_is_sharded = true`** - Input 0 must be block-sharded in L1
2. **`mcast_in0 = true`** - Multicast input 0 (activations) across cores
3. **All cores have work** - No idle cores in the grid
4. **`in0_mcast_receiver_num_dests == num_cores_with_work`** - Receiver grid exactly matches working cores

## Three Kernels Used

### 1. Reader Kernel (RISCV_1)
**File:** `reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp`
- **Processor:** DataMovementProcessor::RISCV_1
- **Function:** Reads from block-sharded L1 buffer, multicasts to all cores (including self)
- **Applied to:** `all_cores_with_work` (which equals `in0_mcast_cores_with_work_and_in_receiver_grid`)

**Key Compile-Time Args:**
```cpp
{
    1,  // core_has_output_block_work
    1,  // core_in_in0_receiver_mcast_grid
    in0_block_num_tiles,
    in0_block_size_bytes,
    in0_last_ktile_w,
    num_blocks,
    num_blocks_w_dim,
    num_blocks_h_dim,
    in0_mcast_sender_semaphore_id,
    in0_mcast_receiver_semaphore_id,
    in0_mcast_num_dests,
    in0_mcast_num_cores,
    num_x,  // grid x size
    num_y,  // grid y size
    transpose_mcast,
    shard_width_in_tiles,
    shard_height_in_tiles,
    in0_block_w,
    in0_block_h,
    batch,
    fuse_op
}
```

**Key Runtime Args:**
```cpp
{
    sender_id,
    in0_mcast_dest_noc_start_x,
    in0_mcast_dest_noc_start_y,
    in0_mcast_dest_noc_end_x,
    in0_mcast_dest_noc_end_y,
    // followed by arrays:
    in0_mcast_noc_x[num_x],
    in0_mcast_noc_y[num_y]
}
```

### 2. Writer Kernel (RISCV_0)
**File:** `reader_bmm_tile_layout_in1_sender_writer_padding.cpp`
- **Processor:** DataMovementProcessor::RISCV_0
- **Function:** Reads in1 (weights) from DRAM/L1, writes output
- **Applied to:** `all_cores_with_work`

**Key Compile-Time Args:**
```cpp
{
    in1_tensor_stride_w,
    in1_tensor_stride_h,
    in1_tensor_next_block_stride,
    in1_tensor_next_w_dim_block_stride,
    in1_block_w,
    in1_block_h,
    in1_block_num_tiles,
    num_blocks,
    num_blocks_w_dim,
    num_blocks_h_dim,
    in1_mcast_sender_semaphore_id,  // Not used when SKIP_MCAST defined
    in1_mcast_receiver_semaphore_id,
    in1_mcast_num_dests,  // Set to 0
    in1_mcast_num_cores,
    KtNt,
    batch,
    bcast_B,
    batchB,  // sparsity args
    sparsity_pagesize,
    out_tensor_stride_w,
    out_tensor_stride_h,
    out_tensor_next_subblock_stride_w,
    out_tensor_next_subblock_stride_h,
    out_tensor_next_w_dim_block_stride,
    out_tensor_next_h_dim_block_stride,
    out_subblock_w,
    out_subblock_h,
    out_subblock_tile_count,
    MtNt,
    in3_tensor_stride_w,  // bias (if FUSE_BIAS)
    fuse_op_all_gather,
    fuse_op_reduce_scatter,
    // TensorAccessorArgs for in1, sparsity, out, bias
}
```

**Key Runtime Args:**
```cpp
{
    in1_tensor_addr,
    in1_tensor_start_tile_id,
    in1_mcast_dest_noc_start_x,  // Not used with SKIP_MCAST
    in1_mcast_dest_noc_start_y,
    in1_mcast_dest_noc_end_x,
    in1_mcast_dest_noc_end_y,
    sparsity_addr,
    out_tensor_addr,
    out_tensor_start_tile_id,
    last_block_w,  // padding args
    out_num_nonzero_subblocks_h,
    out_last_subblock_h,
    padded_block_tiles_h_skip,
    out_num_nonzero_subblocks_w,
    out_last_num_nonzero_subblocks_w,
    out_last_subblock_w,
    padded_subblock_tiles_addr_skip,
    padded_block_tiles_w_skip,
    // bias args (if FUSE_BIAS)
    // dram sharded args (if IN1_DRAM_SHARDED)
}
```

**Key Defines:**
```cpp
SKIP_MCAST=1  // No multicast needed for in1 in this config
```

### 3. Compute Kernel
**File:** `bmm_large_block_zm_fused_bias_activation.cpp`
- **Function:** Matrix multiplication compute with optional bias and activation
- **Applied to:** `all_cores_with_work`

**Compile-Time Args:**
```cpp
{
    in0_block_w,             // inner block size in tiles
    in0_num_subblocks,       // number of in0 subblocks
    in0_block_num_tiles,     // total tiles in in0 block
    in0_subblock_num_tiles,  // tiles per in0 subblock
    in1_num_subblocks,       // number of in1 subblocks
    in1_block_num_tiles,     // total tiles in in1 block
    in1_per_core_w,          // in1 width per core
    num_blocks,              // K-dimension blocks
    out_num_blocks_x,        // output blocks in x
    out_num_blocks_y,        // output blocks in y
    out_subblock_h,          // output subblock height
    out_subblock_w,          // output subblock width
    out_subblock_num_tiles,  // tiles per output subblock
    batch,                   // batch dimension
    out_block_tiles,         // total output block tiles
    untilize_out,            // whether to untilize output
    false,                   // get_batch_from_reader
    in0_transpose_tile       // transpose in0 tiles
}
```

## Circular Buffer Configuration

### CB0 (in0): Input 0 - Multicast Buffer
- **Size:** `in0_CB_tiles * in0_single_tile_size`
- **Tiles:** `in0_block_tiles * MCAST_INPUT_BUFFERING_DEPTH` (typically 2 or 4)
- **Purpose:** Receives multicast in0 data

### CB2 (in2): Input 0 - Shard Buffer  
- **Size:** `in2_CB_tiles * in0_single_tile_size`
- **Tiles:** `per_core_M * in0_shard_width_in_tiles`
- **Purpose:** Globally allocated, holds the block-sharded input
- **Config:** `.set_globally_allocated_address(*in0_buffer)`

### CB1 (in1): Input 1 - Weights
- **Size:** `in1_CB_tiles * in1_single_tile_size`
- **Tiles:** `in1_block_tiles * MCAST_INPUT_BUFFERING_DEPTH`
- **Purpose:** Weights read from DRAM

### CB16 (out): Output
- **Size:** `out_CB_tiles * output_single_tile_size`
- **Tiles:** `out_block_tiles`
- **Purpose:** Output accumulation

### CB24 (interm0): Intermediate (if packer_l1_acc_en)
- **Size:** `interm0_CB_tiles * interm0_single_tile_size`
- **Tiles:** `out_block_tiles`
- **Purpose:** L1 accumulation buffer

### CB6: L1 Array for Semaphore
- **Purpose:** Holds semaphore valid value for multicast synchronization

## Simplified Example Parameters

For a minimal working example using the full Blackhole compute grid, use these constraints:

### Matrix Dimensions
```cpp
M = 2560   // divisible by 32, fits 10 rows (256 tiles / 10 = 25.6 -> use 2560/32 = 80 tiles)
N = 4160   // divisible by 32, fits 13 cols (4160/32 = 130 tiles)
K = 2048   // divisible by 32 (64 tiles)
B = 1      // batch size
```

### Grid Configuration (Blackhole Full Grid)
```cpp
grid_size = CoreCoord(13, 10)  // Full Blackhole compute grid: 130 cores
num_cores = 130
```

### Block Sizing
```cpp
per_core_M = 8     // Mt / grid_size.y = 80 / 10 = 8 tiles per core in M dimension
per_core_N = 10    // Nt / grid_size.x = 130 / 13 = 10 tiles per core in N dimension
in0_block_w = 4    // K-tiles per block (64 / 16 = 4, for 16 K-blocks)
out_block_h = 8    // same as per_core_M
out_block_w = 10   // same as per_core_N
out_subblock_h = 4 // must divide out_block_h, h*w <= 8
out_subblock_w = 2 // must divide out_block_w (10/2=5 subblocks), h*w = 8
```

### Sharding Configuration
```cpp
shard_width_in_tiles = 64    // K-dimension shard width (full K dimension)
shard_height_in_tiles = 8    // per_core_M (8 tiles = 256 rows)
shard_shape = [256, 2048]    // M x K per core (8*32 x 64*32)
```

## Memory Layout Requirements

### Input 0 (Activations)
- **Layout:** Block-sharded, WIDTH_SHARDED
- **Buffer Type:** L1
- **Shard Grid:** Same as compute grid (13x10 for Blackhole)
- **Shard Shape:** `[per_core_M * TILE_HEIGHT, K]` = `[256, 2048]` per core
- **Data Format:** Float16_b or BFloat16
- **Total Sharded Size:** 2560 x 2048 elements across 130 cores

### Input 1 (Weights)
- **Layout:** Interleaved (DRAM or L1)
- **Buffer Type:** DRAM
- **Data Format:** Float16_b, BFloat16, or BFloat8_b
- **Access Pattern:** Each core reads different N-slice

### Output
- **Layout:** Interleaved or Block-sharded
- **Buffer Type:** DRAM or L1
- **Data Format:** Float16_b, BFloat16, or BFloat8_b
- **Shard Grid:** (if sharded) Same as compute grid (13x10)
- **Shard Shape:** (if sharded) `[256, 320]` per core (8 x 10 tiles)

## Implementation Notes

### NOC Assignment
```cpp
in0_noc = NOC1  // optimized for DRAM write
in1_noc = NOC0  // optimized for DRAM read
```

### Semaphores
- **in0_mcast_sender_semaphore:** Used by cores to signal they're ready to send
- **in0_mcast_receiver_semaphore:** Used by receivers to signal data received

### Key Differences from Simpler Examples

1. **Sharding Required:** Unlike single-core or multicore-reuse, input must be pre-sharded
2. **Multicast Coordination:** Requires semaphore-based synchronization
3. **Complex Addressing:** NOC coordinates arrays needed for multicast destinations
4. **Shard Extraction:** May need to extract sub-blocks from shards if `shard_height > 1`

### Restrictions

- `out_subblock_h * out_subblock_w <= 8` (hardware constraint)
- `per_core_M % out_block_h == 0`
- `per_core_N % out_block_w == 0`
- `K % in0_block_w == 0`
- All dimensions must be tile-aligned (divisible by 32)
- Grid size must accommodate all work: `num_cores >= (Mt/per_core_M) * (Nt/per_core_N)`

## Comparison with Other Examples

| Feature | Single Core | Multicore | Multicore Reuse | 1D Mcast (Sharded) |
|---------|-------------|-----------|-----------------|---------------------|
| Cores Used | 1 | Multiple | Multiple | Multiple |
| Input Layout | Interleaved | Interleaved | Interleaved | Block-Sharded |
| Data Reuse | None | None | K-dimension | K-dimension + Mcast |
| Reader Kernels | 1 | 1 per core | 1 per core | 1 (sender/receiver combined) |
| Writer Kernels | 1 | 1 per core | 1 per core | 1 per core |
| Compute Kernels | 1 | 1 per core | 1 per core | 1 per core |
| Coordination | None | None | DRAM reads | Semaphore-based mcast |
| Performance | Low | Medium | High | Highest |

## Example Test Case

For Blackhole (13x10 grid), a typical configuration:
```python
# Optimized for full Blackhole compute grid
M = 2560   # 80 tiles
K = 2048   # 64 tiles
N = 4160   # 130 tiles
grid_size = (13, 10)  # Full Blackhole grid = 130 cores
in0_block_w = 4       # 64 K-tiles / 16 blocks
out_subblock_h = 4    # 8 / 2 = 2 subblocks vertically
out_subblock_w = 2    # 10 / 2 = 5 subblocks horizontally
per_core_M = 8        # 80 tiles / 10 rows
per_core_N = 10       # 130 tiles / 13 cols

# Sharding
shard_width_in_tiles = 64   # Full K dimension
shard_height_in_tiles = 8   # per_core_M (256 rows)
shard_shape = [256, 2048]   # 8 tiles x 64 tiles per core

# Each core computes an 8x10 tile output block (256x320 elements)
# Total output: 2560 x 4160 (80x130 tiles)
```

**Alternative smaller configuration for testing:**
```python
# Smaller problem using full grid
M = 1280   # 40 tiles (4 per core)
K = 1024   # 32 tiles  
N = 2080   # 65 tiles (5 per core)
grid_size = (13, 10)
in0_block_w = 2
out_subblock_h = 2
out_subblock_w = 1
per_core_M = 4
per_core_N = 5
```

This configuration achieves exactly 3 kernels with optimal multicast performance for 1D width-sharded matmul on Blackhole's full 13x10 compute grid.
