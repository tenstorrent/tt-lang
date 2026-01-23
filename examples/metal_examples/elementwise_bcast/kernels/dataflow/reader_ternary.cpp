// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for 3 input tensors (ternary operation)
// Reads tiles in blocks for better memory access patterns

#include "api/dataflow/dataflow_api.h"

// Block configuration (passed via defines from Python)
#ifndef BLOCK_H
#define BLOCK_H 8
#endif
#ifndef BLOCK_W
#define BLOCK_W 8
#endif

void kernel_main() {
  const uint32_t a_addr = get_arg_val<uint32_t>(0);
  const uint32_t b_addr = get_arg_val<uint32_t>(1);
  const uint32_t c_addr = get_arg_val<uint32_t>(2);
  const uint32_t n_tiles = get_arg_val<uint32_t>(3);
  const uint32_t start_id =
      get_arg_val<uint32_t>(4); // Starting tile ID for this core

  constexpr auto cb_in0 = tt::CBIndex::c_0;
  constexpr auto cb_in1 = tt::CBIndex::c_1;
  constexpr auto cb_in2 = tt::CBIndex::c_2;

  const uint32_t tile_size_bytes = get_tile_size(cb_in0);

  constexpr auto args_a = TensorAccessorArgs<0>();
  constexpr auto args_b =
      TensorAccessorArgs<args_a.next_compile_time_args_offset()>();
  constexpr auto args_c =
      TensorAccessorArgs<args_b.next_compile_time_args_offset()>();

  const auto a = TensorAccessor(args_a, a_addr, tile_size_bytes);
  const auto b = TensorAccessor(args_b, b_addr, tile_size_bytes);
  const auto c = TensorAccessor(args_c, c_addr, tile_size_bytes);

  constexpr uint32_t block_h = BLOCK_H;
  constexpr uint32_t block_w = BLOCK_W;
  constexpr uint32_t block_size = block_h * block_w;

  uint32_t num_blocks = n_tiles / block_size;
  uint32_t remaining_tiles = n_tiles % block_size;

  uint32_t tile_offset = 0;

  // Read full blocks
  for (uint32_t block = 0; block < num_blocks; block++) {
    // Reserve space for entire block in all input buffers
    cb_reserve_back(cb_in0, block_size);
    cb_reserve_back(cb_in1, block_size);
    cb_reserve_back(cb_in2, block_size);

    // Read all tiles in the block
    for (uint32_t h = 0; h < block_h; h++) {
      for (uint32_t w = 0; w < block_w; w++) {
        uint32_t t = h * block_w + w;
        const uint32_t tile_id = start_id + tile_offset + t;

        const uint32_t a_cb_addr = get_write_ptr(cb_in0) + t * tile_size_bytes;
        const uint32_t b_cb_addr = get_write_ptr(cb_in1) + t * tile_size_bytes;
        const uint32_t c_cb_addr = get_write_ptr(cb_in2) + t * tile_size_bytes;

        noc_async_read_tile(tile_id, a, a_cb_addr);
        noc_async_read_tile(tile_id, b, b_cb_addr);
        noc_async_read_tile(tile_id, c, c_cb_addr);
      }
    }

    // Wait for all reads in the block to complete
    noc_async_read_barrier();

    // Push entire block to make it available to compute kernel
    cb_push_back(cb_in0, block_size);
    cb_push_back(cb_in1, block_size);
    cb_push_back(cb_in2, block_size);

    tile_offset += block_size;
  }

  // Read remaining tiles (partial block)
  if (remaining_tiles > 0) {
    cb_reserve_back(cb_in0, remaining_tiles);
    cb_reserve_back(cb_in1, remaining_tiles);
    cb_reserve_back(cb_in2, remaining_tiles);

    for (uint32_t t = 0; t < remaining_tiles; t++) {
      const uint32_t tile_id = start_id + tile_offset + t;

      const uint32_t a_cb_addr = get_write_ptr(cb_in0) + t * tile_size_bytes;
      const uint32_t b_cb_addr = get_write_ptr(cb_in1) + t * tile_size_bytes;
      const uint32_t c_cb_addr = get_write_ptr(cb_in2) + t * tile_size_bytes;

      noc_async_read_tile(tile_id, a, a_cb_addr);
      noc_async_read_tile(tile_id, b, b_cb_addr);
      noc_async_read_tile(tile_id, c, c_cb_addr);
    }

    noc_async_read_barrier();

    cb_push_back(cb_in0, remaining_tiles);
    cb_push_back(cb_in1, remaining_tiles);
    cb_push_back(cb_in2, remaining_tiles);
  }
}
