// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Bounded-ring sender (core 0,0). It keeps bounded slot reuse and a cross-core
// credit handshake. It is a lower-level comparison target, not a tt-lang
// PipeNet implementation. The destination is two rings of `ring` slots (c_0,
// block_count 2*ring): chunk c is written to ring half (c & 1), so the sender
// can write the next chunk while the receiver drains the current one (lookahead
// 2). Before reusing a ring half, it waits until the receiver has freed the
// chunk that last used that half. Within a chunk the NoC write command is
// programmed once (set_state) and reused (with_state), with one read barrier
// and one write barrier.
//
// data (receiver-owned, sender increments) and free (sender-owned, receiver
// increments) are cumulative counters with one incrementer each, so wait_min is
// race-free. Correctness is the bounded-buffer invariant: a ring half is reused
// only after its previous occupant was drained.
//
// Runtime args: 0 = source DRAM address, 1 = receiver NoC x, 2 = receiver NoC
// y,
//               3 = transfer count, 4 = data semaphore id (on receiver),
//               5 = free semaphore id (on sender), 6 = ring depth.

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  uint32_t src_addr = get_arg_val<uint32_t>(0);
  uint32_t recv_x = get_arg_val<uint32_t>(1);
  uint32_t recv_y = get_arg_val<uint32_t>(2);
  uint32_t n_transfers = get_arg_val<uint32_t>(3);
  uint32_t data_sem_id = get_arg_val<uint32_t>(4);
  uint32_t free_sem_id = get_arg_val<uint32_t>(5);
  uint32_t ring = get_arg_val<uint32_t>(6);

  constexpr uint32_t cb_src = tt::CBIndex::c_1; // ring source slots (reused)
  constexpr uint32_t cb_dst = tt::CBIndex::c_0; // two rings of receiver slots
  const uint32_t tile_bytes = get_tile_size(cb_src);

  constexpr auto s_args = TensorAccessorArgs<0>();
  const auto s = TensorAccessor(s_args, src_addr, tile_bytes);

  uint32_t src_base = get_write_ptr(cb_src);
  uint32_t dst_base = get_write_ptr(cb_dst);
  volatile tt_l1_ptr uint32_t *free_sem =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
          get_semaphore(free_sem_id));
  uint64_t data_sem_noc =
      get_noc_addr(recv_x, recv_y, get_semaphore(data_sem_id));

  noc_async_write_one_packet_set_state(get_noc_addr(recv_x, recv_y, dst_base),
                                       tile_bytes);

  for (uint32_t base = 0; base < n_transfers; base += ring) {
    uint32_t chunk = (n_transfers - base < ring) ? (n_transfers - base) : ring;
    if (base >= 2 * ring) {
      // Reuse this ring half only after the receiver drains the previous chunk
      // that occupied it.
      noc_semaphore_wait_min(free_sem, base - ring);
    }
    for (uint32_t j = 0; j < chunk; ++j) {
      noc_async_read_tile(base + j, s, src_base + j * tile_bytes);
    }
    noc_async_read_barrier();
    uint32_t half = ((base / ring) & 1) * ring * tile_bytes;
    for (uint32_t j = 0; j < chunk; ++j) {
      noc_async_write_one_packet_with_state(src_base + j * tile_bytes,
                                            dst_base + half + j * tile_bytes);
    }
    noc_async_write_barrier();
    noc_semaphore_inc(data_sem_noc, chunk);
  }
}
