// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// A-reader for the TTNN-like compute-feed probe. It matches the single-core
// large-block matmul operand-0 layout: each K block contains `mt` rows by
// `in0_block_w` columns in row-major tile order.
//
// Runtime args: 0 = A address, 1 = mt, 2 = kt, 3 = in0_block_w.

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  uint32_t a_address = get_arg_val<uint32_t>(0);
  uint32_t mt = get_arg_val<uint32_t>(1);
  uint32_t kt = get_arg_val<uint32_t>(2);
  uint32_t in0_block_w = get_arg_val<uint32_t>(3);

  constexpr uint32_t dfb_in0 = tt::CBIndex::c_0;
  const uint32_t tile_bytes = get_tile_size(dfb_in0);
  const uint32_t block_tiles = mt * in0_block_w;
  const uint32_t num_blocks = kt / in0_block_w;

  constexpr auto a_args = TensorAccessorArgs<0>();
  const auto a_accessor = TensorAccessor(a_args, a_address, tile_bytes);

  for (uint32_t block_index = 0; block_index < num_blocks; ++block_index) {
    cb_reserve_back(dfb_in0, block_tiles);
    uint32_t write_ptr = get_write_ptr(dfb_in0);
    for (uint32_t tile_row = 0; tile_row < mt; ++tile_row) {
      uint32_t tile_id = tile_row * kt + block_index * in0_block_w;
      for (uint32_t block_col = 0; block_col < in0_block_w; ++block_col) {
        noc_async_read_tile(tile_id + block_col, a_accessor, write_ptr);
        write_ptr += tile_bytes;
      }
    }
    noc_async_read_barrier();
    cb_push_back(dfb_in0, block_tiles);
  }
}
