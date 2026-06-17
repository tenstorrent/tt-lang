// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Drain writer for the pack/unpack probe: write `tiles` tiles from dfb_out to
// DRAM once, after the measured zone. Idle during the zone. Runtime args: 0 =
// destination DRAM address, 1 = tiles.

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  uint32_t dst_addr = get_arg_val<uint32_t>(0);
  uint32_t tiles = get_arg_val<uint32_t>(1);

  constexpr uint32_t dfb_out = tt::CBIndex::c_16;
  const uint32_t tile_bytes = get_tile_size(dfb_out);

  constexpr auto s_args = TensorAccessorArgs<0>();
  const auto s = TensorAccessor(s_args, dst_addr, tile_bytes);

  cb_wait_front(dfb_out, tiles);
  uint32_t l1_read_addr = get_read_ptr(dfb_out);
  for (uint32_t t = 0; t < tiles; ++t) {
    noc_async_write_tile(t, s, l1_read_addr);
    l1_read_addr += tile_bytes;
  }
  noc_async_write_barrier();
  cb_pop_front(dfb_out, tiles);
}
