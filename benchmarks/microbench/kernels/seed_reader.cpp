// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Seed reader for the pack/unpack probe: read `tiles` tiles from DRAM into
// dfb_in once, before the measured zone. Idle during the zone. Runtime args: 0 =
// source DRAM address, 1 = tiles.

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  uint32_t src_addr = get_arg_val<uint32_t>(0);
  uint32_t tiles = get_arg_val<uint32_t>(1);

  constexpr uint32_t dfb_in = tt::CBIndex::c_0;
  const uint32_t tile_bytes = get_tile_size(dfb_in);

  constexpr auto s_args = TensorAccessorArgs<0>();
  const auto s = TensorAccessor(s_args, src_addr, tile_bytes);

  cb_reserve_back(dfb_in, tiles);
  uint32_t l1_write_addr = get_write_ptr(dfb_in);
  for (uint32_t t = 0; t < tiles; ++t) {
    noc_async_read_tile(t, s, l1_write_addr);
    l1_write_addr += tile_bytes;
  }
  noc_async_read_barrier();
  cb_push_back(dfb_in, tiles);
}
