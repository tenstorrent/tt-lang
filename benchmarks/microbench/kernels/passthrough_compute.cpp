// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Zero-compute pack/unpack probe (MB1). Measures the cost of moving a
// tile block out to L1 and back — pack DST->L1, unpack L1->DST, and the
// dataflow-buffer reserve/wait/push/pop + cross-thread semaphore sync — with no
// arithmetic, on a single compute core. The measured loop self-cycles `tiles`
// tiles through a compute-private buffer (dfb_loop): the pack thread (TRISC2)
// produces and the unpack thread (TRISC0) consumes, so the per-RISC zone split
// gives the pack and unpack times directly. No NoC/DRAM traffic occurs inside
// the zone (NCRISC/BRISC idle).
//
// Compile-time args: 0 = tiles per iteration, 1 = measured iterations,
// 2 = DST sub-block capacity. See benchmarks/microbench/RESULTS.md.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t dfb_in = 0;   // reader -> compute (seed)
constexpr uint32_t dfb_loop = 1; // compute -> compute (probe)
constexpr uint32_t dfb_out = 16; // compute -> writer (drain)

// Move `tiles` tiles from src_cb to dst_cb through DST, in DST-capacity
// sub-blocks. With src_cb == dst_cb this packs and unpacks the block once
// (rotates the buffer in place; identity copy, so content is preserved).
inline void move_tiles(uint32_t src_cb, uint32_t dst_cb, uint32_t tiles,
                       uint32_t cap) {
  cb_wait_front(src_cb, tiles);
  cb_reserve_back(dst_cb, tiles);
  for (uint32_t base = 0; base < tiles; base += cap) {
    uint32_t chunk = (tiles - base < cap) ? (tiles - base) : cap;
    tile_regs_acquire();
    for (uint32_t i = 0; i < chunk; ++i) {
      copy_tile(src_cb, base + i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < chunk; ++i) {
      pack_tile<true>(i, dst_cb, base + i);
    }
    tile_regs_release();
  }
  cb_pop_front(src_cb, tiles);
  cb_push_back(dst_cb, tiles);
}

void kernel_main() {
  const uint32_t tiles = get_compile_time_arg_val(0);
  const uint32_t iters = get_compile_time_arg_val(1);
  const uint32_t cap = get_compile_time_arg_val(2);

  init_sfpu(dfb_loop, dfb_loop);

  // Seed the private loop buffer from the reader's input tiles (outside zone).
  copy_tile_init(dfb_in);
  move_tiles(dfb_in, dfb_loop, tiles, cap);

  // Measured region: self-cycle the tiles through dfb_loop.
  copy_tile_init(dfb_loop);
  {
    DeviceZoneScopedN("pack_unpack_loop");
    for (uint32_t it = 0; it < iters; ++it) {
      move_tiles(dfb_loop, dfb_loop, tiles, cap);
    }
  }

  // Drain to the writer for the correctness check (outside zone).
  move_tiles(dfb_loop, dfb_out, tiles, cap);
}
