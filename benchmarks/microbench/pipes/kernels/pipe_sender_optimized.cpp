// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Optimized ceiling sender (core 0,0): batched, stateful NoC transfers in the
// style of hand-tuned kernels. It reads all N source tiles into L1 with a
// single read barrier, then writes them to the receiver with the NoC write
// command programmed once (noc_async_write_one_packet_set_state) and reused per
// transfer (noc_async_write_one_packet_with_state), with a single write
// barrier. There is no per-transfer barrier and no per-transfer command setup.
// This is the throughput ceiling for a core-to-core unicast, well below the
// naive per-transfer-barriered baselines.
//
// It holds all N tiles in L1 (c_1 source, c_0 destination both N slots), so the
// driver caps N for this variant to fit L1.
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

  constexpr uint32_t cb_src = tt::CBIndex::c_1; // N source slots
  constexpr uint32_t cb_dst = tt::CBIndex::c_0; // N receiver slots
  const uint32_t tile_bytes = get_tile_size(cb_src);

  constexpr auto s_args = TensorAccessorArgs<0>();
  const auto s = TensorAccessor(s_args, src_addr, tile_bytes);

  uint32_t src_base = get_write_ptr(cb_src);
  uint32_t dst_base = get_write_ptr(cb_dst);

  // Read all N tiles, one read barrier: the DRAM read latencies overlap.
  for (uint32_t t = 0; t < n_transfers; ++t) {
    noc_async_read_tile(t, s, src_base + t * tile_bytes);
  }
  noc_async_read_barrier();

  // Program the NoC write command once (receiver core + size), reuse it per
  // transfer (only the L1 addresses change), one write barrier for the batch.
  noc_async_write_one_packet_set_state(get_noc_addr(recv_x, recv_y, dst_base),
                                       tile_bytes);
  for (uint32_t t = 0; t < n_transfers; ++t) {
    noc_async_write_one_packet_with_state(src_base + t * tile_bytes,
                                          dst_base + t * tile_bytes);
  }
  noc_async_write_barrier();

  uint64_t done_noc = get_noc_addr(recv_x, recv_y, get_semaphore(done_sem_id));
  noc_semaphore_inc(done_noc, 1);
}
