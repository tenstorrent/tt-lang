// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Synced sender for the fair pipe floor (core 0,0). Unlike pipe_sender.cpp this
// does per-transfer producer-consumer flow control, matching what a bounded
// pipe must do: two destination slots (c_0, double-buffered), so before writing
// tile t (t>=2) it waits until the receiver has freed the slot that tile t-2
// used, and after each write it signals the receiver.
//
// The two semaphores are cumulative counters with a single incrementer each
// (`data` incremented only by the sender, `free` only by the receiver), so
// noc_semaphore_wait_min is race-free without an atomic barrier.
//
// Runtime args: 0 = source DRAM address, 1 = receiver NoC x, 2 = receiver NoC
// y,
//               3 = transfer count, 4 = data semaphore id (on receiver),
//               5 = free semaphore id (on sender).

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  uint32_t src_addr = get_arg_val<uint32_t>(0);
  uint32_t recv_x = get_arg_val<uint32_t>(1);
  uint32_t recv_y = get_arg_val<uint32_t>(2);
  uint32_t n_transfers = get_arg_val<uint32_t>(3);
  uint32_t data_sem_id = get_arg_val<uint32_t>(4);
  uint32_t free_sem_id = get_arg_val<uint32_t>(5);

  constexpr uint32_t cb_src = tt::CBIndex::c_1; // local source scratch
  constexpr uint32_t cb_dst = tt::CBIndex::c_0; // two-slot receiver destination
  const uint32_t tile_bytes = get_tile_size(cb_src);

  constexpr auto s_args = TensorAccessorArgs<0>();
  const auto s = TensorAccessor(s_args, src_addr, tile_bytes);

  uint32_t src_l1 = get_write_ptr(cb_src);
  uint32_t dst_base = get_write_ptr(cb_dst);
  volatile tt_l1_ptr uint32_t *free_sem =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
          get_semaphore(free_sem_id));
  uint64_t data_sem_noc =
      get_noc_addr(recv_x, recv_y, get_semaphore(data_sem_id));

  for (uint32_t t = 0; t < n_transfers; ++t) {
    if (t >= 2) {
      noc_semaphore_wait_min(free_sem, t - 1); // slot (t-2) freed by receiver
    }
    noc_async_read_tile(t, s, src_l1);
    noc_async_read_barrier();
    uint32_t slot = dst_base + (t & 1) * tile_bytes;
    noc_async_write(src_l1, get_noc_addr(recv_x, recv_y, slot), tile_bytes);
    noc_async_write_barrier();
    noc_semaphore_inc(data_sem_noc, 1);
  }
}
