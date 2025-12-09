// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Writer kernel for output tensor
// Writes tiles in blocks for better memory access patterns

#include <dataflow_api.h>

// Block configuration (passed via defines from Python)
#ifndef BLOCK_H
#define BLOCK_H 8
#endif
#ifndef BLOCK_W
#define BLOCK_W 8
#endif

void kernel_main() {
  const uint32_t dst_addr = get_arg_val<uint32_t>(0);
  const uint32_t n_tiles = get_arg_val<uint32_t>(1);
  const uint32_t start_id =
      get_arg_val<uint32_t>(2); // Starting tile ID for this core

  constexpr auto cb_out = tt::CBIndex::c_16;
  const uint32_t tile_size_bytes = get_tile_size(cb_out);

  constexpr auto args_dst = TensorAccessorArgs<0>();
  const auto dst = TensorAccessor(args_dst, dst_addr, tile_size_bytes);

  constexpr uint32_t block_h = BLOCK_H;
  constexpr uint32_t block_w = BLOCK_W;
  constexpr uint32_t block_size = block_h * block_w;

  uint32_t num_blocks = n_tiles / block_size;
  uint32_t remaining_tiles = n_tiles % block_size;

  uint32_t tile_offset = 0;

  // Write full blocks
  for (uint32_t block = 0; block < num_blocks; block++) {
    // Wait for entire block from compute kernel
    cb_wait_front(cb_out, block_size);

    // Write all tiles in the block
    for (uint32_t h = 0; h < block_h; h++) {
      for (uint32_t w = 0; w < block_w; w++) {
        uint32_t t = h * block_w + w;
        const uint32_t tile_id = start_id + tile_offset + t;

        const uint32_t cb_addr = get_read_ptr(cb_out) + t * tile_size_bytes;
        noc_async_write_tile(tile_id, dst, cb_addr);
      }
    }

    // Wait for all writes in the block to complete
    noc_async_write_barrier();

    // Pop entire block
    cb_pop_front(cb_out, block_size);

    tile_offset += block_size;
  }

  // Write remaining tiles (partial block)
  if (remaining_tiles > 0) {
    cb_wait_front(cb_out, remaining_tiles);

    for (uint32_t t = 0; t < remaining_tiles; t++) {
      const uint32_t tile_id = start_id + tile_offset + t;

      const uint32_t cb_addr = get_read_ptr(cb_out) + t * tile_size_bytes;
      noc_async_write_tile(tile_id, dst, cb_addr);
    }

    noc_async_write_barrier();

    cb_pop_front(cb_out, remaining_tiles);
  }
}
