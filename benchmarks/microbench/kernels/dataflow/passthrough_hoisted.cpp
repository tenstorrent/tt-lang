// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Pack/unpack probe (MB1) -- hoisted-hop variant of passthrough_compute.cpp.
// Same zero-compute self-cycle through a compute-private buffer (dfb_loop), but
// the CB hop is hoisted OUTSIDE the measured loop (wait/reserve once, pop/push
// once): the timed loop body is only acquire->copy->pack, so no per-iteration
// DFB hop is measured. Read (front) and write (back) both stay in dfb_loop, so
// dfb_loop must hold >= 2*tiles (block_count >= 2) to avoid front/back aliasing.
// The pack thread (TRISC2) produces and the unpack thread (TRISC0) consumes, so
// the per-RISC zone split gives the pack and unpack times directly. No NoC/DRAM
// traffic occurs inside the zone (NCRISC/BRISC idle).
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

constexpr uint32_t dfb_in = 0;
constexpr uint32_t dfb_loop = 1;
constexpr uint32_t dfb_out = 16;

// Copy `tiles` tiles from src_cb to dst_cb through DST, in DST-capacity
// sub-blocks -- the compute-only inner work, with NO CB hop. The caller
// waits/reserves before and pops/pushes after. With src_cb == dst_cb the read
// (front) and write (back) regions stay distinct as long as the CB holds
// >= 2*tiles, so this is an identity copy that preserves content.
inline void copy_block(uint32_t src_cb, uint32_t dst_cb, uint32_t tiles,
                       uint32_t cap) {
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
}

// Full DFB move: wait/reserve, copy the block, pop/push. Used for the seed and
// drain passes (single moves outside the measured zone).
inline void move_tiles(uint32_t src_cb, uint32_t dst_cb, uint32_t tiles,
                       uint32_t cap) {
  cb_wait_front(src_cb, tiles);
  cb_reserve_back(dst_cb, tiles);
  copy_block(src_cb, dst_cb, tiles, cap);
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

  // Measured region: self-cycle the tiles through dfb_loop, but with the CB hop
  // hoisted OUTSIDE the loop -- wait/reserve once, then the loop does only
  // acquire->copy->pack (no per-iteration hop). Read (front) and write (back)
  // stay in dfb_loop; needs dfb_loop >= 2*tiles (block_count >= 2) so they don't
  // alias.
  copy_tile_init(dfb_loop);
  cb_wait_front(dfb_loop, tiles);
  cb_reserve_back(dfb_loop, tiles);
  {
    DeviceZoneScopedN("pack_unpack_loop");
    for (uint32_t it = 0; it < iters; ++it) {
      copy_block(dfb_loop, dfb_loop, tiles, cap);
    }
  }
  cb_pop_front(dfb_loop, tiles);
  cb_push_back(dfb_loop, tiles);

  // Drain to the writer for the correctness check (outside zone).
  move_tiles(dfb_loop, dfb_out, tiles, cap);
}
