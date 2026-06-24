// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Receiver for the naive C++ baseline (core 1,0). Waits on the sender's done
// semaphore, then writes each of the N tiles the sender left in local L1 slots
// (c_0, block_count N) to its output DRAM tile. The drain is outside the timed
// sender zone; it exists for the bit-exact check.
//
// Runtime args: 0 = output DRAM address, 1 = done semaphore id, 2 = transfer
// count.

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  uint32_t out_addr = get_arg_val<uint32_t>(0);
  uint32_t done_sem_id = get_arg_val<uint32_t>(1);
  uint32_t n_transfers = get_arg_val<uint32_t>(2);

  constexpr uint32_t cb_dst = tt::CBIndex::c_0;
  const uint32_t tile_bytes = get_tile_size(cb_dst);

  constexpr auto d_args = TensorAccessorArgs<0>();
  const auto d = TensorAccessor(d_args, out_addr, tile_bytes);

  volatile tt_l1_ptr uint32_t *done =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
          get_semaphore(done_sem_id));
  noc_semaphore_wait(done, 1);

  uint32_t dst_base = get_write_ptr(cb_dst);
  for (uint32_t t = 0; t < n_transfers; ++t) {
    noc_async_write_tile(t, d, dst_base + t * tile_bytes);
  }
  noc_async_write_barrier();
}
