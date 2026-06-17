// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Reader for MB2. From a single concatenated input tensor [initial | deltas],
// pushes the initial block (acc_tiles tiles) to dfb_init, then `groups` delta
// blocks (acc_tiles tiles each) to dfb_delta. groups = 1 when contributions are
// L1-resident (re-read by the compute kernel), or the contribution count when
// streamed. Tile index is row-major across the concatenated tensor.
//
// Runtime args: 0 = source DRAM address, 1 = acc_tiles, 2 = groups.

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  uint32_t src_addr = get_arg_val<uint32_t>(0);
  uint32_t acc_tiles = get_arg_val<uint32_t>(1);
  uint32_t groups = get_arg_val<uint32_t>(2);

  constexpr uint32_t dfb_init = tt::CBIndex::c_0;
  constexpr uint32_t dfb_delta = tt::CBIndex::c_1;
  const uint32_t tile_bytes = get_tile_size(dfb_init);

  constexpr auto s_args = TensorAccessorArgs<0>();
  const auto s = TensorAccessor(s_args, src_addr, tile_bytes);

  uint32_t tile = 0;

  // Initial block -> dfb_init.
  cb_reserve_back(dfb_init, acc_tiles);
  uint32_t addr = get_write_ptr(dfb_init);
  for (uint32_t u = 0; u < acc_tiles; ++u) {
    noc_async_read_tile(tile++, s, addr);
    addr += tile_bytes;
  }
  noc_async_read_barrier();
  cb_push_back(dfb_init, acc_tiles);

  // Contribution blocks -> dfb_delta.
  for (uint32_t g = 0; g < groups; ++g) {
    cb_reserve_back(dfb_delta, acc_tiles);
    addr = get_write_ptr(dfb_delta);
    for (uint32_t u = 0; u < acc_tiles; ++u) {
      noc_async_read_tile(tile++, s, addr);
      addr += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(dfb_delta, acc_tiles);
  }
}
