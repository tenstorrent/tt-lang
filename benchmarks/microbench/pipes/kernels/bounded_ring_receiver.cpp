// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Bounded-ring receiver (core 1,0). Per ring chunk: wait for the chunk, drain
// its ring half to the output DRAM tensor, then return the chunk's credits to
// the sender in one increment. The two-ring layout lets it drain the current
// chunk while the sender writes the next. Pairs with bounded_ring_sender.cpp.
//
// Runtime args: 0 = output DRAM address, 1 = sender NoC x, 2 = sender NoC y,
//               3 = transfer count, 4 = data semaphore id (on receiver),
//               5 = free semaphore id (on sender), 6 = ring depth.

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  uint32_t out_addr = get_arg_val<uint32_t>(0);
  uint32_t send_x = get_arg_val<uint32_t>(1);
  uint32_t send_y = get_arg_val<uint32_t>(2);
  uint32_t n_transfers = get_arg_val<uint32_t>(3);
  uint32_t data_sem_id = get_arg_val<uint32_t>(4);
  uint32_t free_sem_id = get_arg_val<uint32_t>(5);
  uint32_t ring = get_arg_val<uint32_t>(6);

  constexpr uint32_t cb_dst = tt::CBIndex::c_0; // two rings
  const uint32_t tile_bytes = get_tile_size(cb_dst);

  constexpr auto d_args = TensorAccessorArgs<0>();
  const auto d = TensorAccessor(d_args, out_addr, tile_bytes);

  uint32_t dst_base = get_write_ptr(cb_dst);
  volatile tt_l1_ptr uint32_t *data_sem =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
          get_semaphore(data_sem_id));
  uint64_t free_sem_noc =
      get_noc_addr(send_x, send_y, get_semaphore(free_sem_id));

  for (uint32_t base = 0; base < n_transfers; base += ring) {
    uint32_t chunk = (n_transfers - base < ring) ? (n_transfers - base) : ring;
    noc_semaphore_wait_min(data_sem, base + chunk); // chunk delivered
    uint32_t half = ((base / ring) & 1) * ring * tile_bytes;
    for (uint32_t j = 0; j < chunk; ++j) {
      noc_async_write_tile(base + j, d, dst_base + half + j * tile_bytes);
    }
    noc_async_write_barrier();
    noc_semaphore_inc(free_sem_noc, chunk); // return the ring's credits
  }
}
