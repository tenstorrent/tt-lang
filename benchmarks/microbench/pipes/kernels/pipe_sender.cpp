// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Sender for the naive C++ baseline (core 0,0). Per transfer: read source tile
// t from DRAM into local L1, then NoC-write it to receiver slot t, barriering
// each transfer. There is no flow control. Each tile lands in its own slot
// (c_0, block_count N), so nothing is overwritten before the receiver drains.
// The per-transfer read mirrors the tt-lang PipeNet variant. After the last
// write, increment the receiver's done semaphore once.
//
// The destination dataflow buffer (c_0) is allocated on both cores at the same
// L1 address, so get_write_ptr(c_0) here is the receiver's slot-0 base address.
//
// Runtime args: 0 = source DRAM address, 1 = receiver NoC x, 2 = receiver NoC
// y,
//               3 = transfer count, 4 = done semaphore id.

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  uint32_t src_addr = get_arg_val<uint32_t>(0);
  uint32_t recv_x = get_arg_val<uint32_t>(1);
  uint32_t recv_y = get_arg_val<uint32_t>(2);
  uint32_t n_transfers = get_arg_val<uint32_t>(3);
  uint32_t done_sem_id = get_arg_val<uint32_t>(4);

  constexpr uint32_t cb_src = tt::CBIndex::c_1; // local source scratch
  constexpr uint32_t cb_dst = tt::CBIndex::c_0; // receiver destination
  const uint32_t tile_bytes = get_tile_size(cb_src);

  constexpr auto s_args = TensorAccessorArgs<0>();
  const auto s = TensorAccessor(s_args, src_addr, tile_bytes);

  uint32_t src_l1 = get_write_ptr(cb_src);
  uint32_t dst_base = get_write_ptr(cb_dst);

  for (uint32_t t = 0; t < n_transfers; ++t) {
    noc_async_read_tile(t, s, src_l1);
    noc_async_read_barrier();
    uint32_t slot = dst_base + t * tile_bytes;
    noc_async_write(src_l1, get_noc_addr(recv_x, recv_y, slot), tile_bytes);
    noc_async_write_barrier();
  }

  uint64_t done_noc = get_noc_addr(recv_x, recv_y, get_semaphore(done_sem_id));
  noc_semaphore_inc(done_noc, 1);
}
