// SPDX-FileCopyrightText: (C) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader kernel for binary op with column broadcast on the second input.

#include "api/dataflow/dataflow_api.h"

// Block configuration (passed via defines from Python).
#ifndef BLOCK_H
#define BLOCK_H 8
#endif
#ifndef BLOCK_W
#define BLOCK_W 8
#endif

void kernel_main() {
  const uint32_t a_addr = get_arg_val<uint32_t>(0);
  const uint32_t b_addr = get_arg_val<uint32_t>(1);
  const uint32_t n_tiles = get_arg_val<uint32_t>(2);
  const uint32_t start_id =
      get_arg_val<uint32_t>(3); // Starting tile ID for this core.

  constexpr auto cb_in0 = tt::CBIndex::c_0;
  constexpr auto cb_in1 = tt::CBIndex::c_1;

  const uint32_t tile_size_bytes = get_tile_size(cb_in0);

  constexpr auto args_a = TensorAccessorArgs<0>();
  constexpr auto args_b =
      TensorAccessorArgs<args_a.next_compile_time_args_offset()>();

  const auto a = TensorAccessor(args_a, a_addr, tile_size_bytes);
  const auto b = TensorAccessor(args_b, b_addr, tile_size_bytes);

  constexpr uint32_t block_h = BLOCK_H;
  constexpr uint32_t block_w = BLOCK_W;
  constexpr uint32_t block_size = block_h * block_w;

  uint32_t num_blocks = n_tiles / block_size;
  uint32_t remaining_tiles = n_tiles % block_size;

  uint32_t tile_offset = 0;

  // Read the broadcast tile once. It is reused for all tiles on this core.
  cb_reserve_back(cb_in1, 1);
  const uint32_t b_cb_addr = get_write_ptr(cb_in1);
  noc_async_read_tile(0, b, b_cb_addr);
  noc_async_read_barrier();
  cb_push_back(cb_in1, 1);

  // Read full blocks for input A.
  for (uint32_t block = 0; block < num_blocks; block++) {
    cb_reserve_back(cb_in0, block_size);

    for (uint32_t h = 0; h < block_h; h++) {
      for (uint32_t w = 0; w < block_w; w++) {
        const uint32_t t = h * block_w + w;
        const uint32_t tile_id = start_id + tile_offset + t;
        const uint32_t a_cb_addr = get_write_ptr(cb_in0) + t * tile_size_bytes;

        noc_async_read_tile(tile_id, a, a_cb_addr);
      }
    }

    noc_async_read_barrier();
    cb_push_back(cb_in0, block_size);
    tile_offset += block_size;
  }

  // Read remaining tiles (partial block).
  if (remaining_tiles > 0) {
    cb_reserve_back(cb_in0, remaining_tiles);

    for (uint32_t t = 0; t < remaining_tiles; t++) {
      const uint32_t tile_id = start_id + tile_offset + t;
      const uint32_t a_cb_addr = get_write_ptr(cb_in0) + t * tile_size_bytes;
      noc_async_read_tile(tile_id, a, a_cb_addr);
    }

    noc_async_read_barrier();
    cb_push_back(cb_in0, remaining_tiles);
  }
}
