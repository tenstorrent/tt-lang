// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Pack/unpack probe (MB1) -- hoisted TWO-CB variant of passthrough_hoisted.cpp.
// Like the hoisted variant (CB hop outside the loop, independent passes) but the
// loop reads one CB (dfb_loop, seeded from the reader) and writes a DISTINCT CB
// (dfb_out) instead of self-cycling one buffer -- the only change vs hoisted is
// that read and write land in different CBs (the topology a real tt-lang kernel
// uses: input CB in, output CB out). No per-iteration hop (so no deadlock even
// though dfb_out is writer-drained -- it is pushed once, after the loop). An
// untimed warmup pass primes the pack-to-dfb_out path before the measured zone.
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
// sub-blocks -- compute-only inner work, NO CB hop. Caller waits/reserves before
// and pops/pushes after. src_cb != dst_cb here (two distinct CBs).
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

// Full DFB move: wait/reserve, copy the block, pop/push. Used only for the seed
// (outside the measured zone).
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

  init_sfpu(dfb_loop, dfb_out);

  // Seed the read buffer (dfb_loop) from the reader's input tiles (outside zone).
  copy_tile_init(dfb_in);
  move_tiles(dfb_in, dfb_loop, tiles, cap);

  // Measured region: hop hoisted outside. Read the resident source (dfb_loop),
  // write a DISTINCT output CB (dfb_out). An untimed warmup pass primes the
  // pack-to-dfb_out path so the zone measures warm steady state.
  copy_tile_init(dfb_loop);
  cb_wait_front(dfb_loop, tiles);   // resident read source (dfb_loop, seeded)
  cb_reserve_back(dfb_out, tiles);  // separate write CB (dfb_out)
  copy_block(dfb_loop, dfb_out, tiles, cap);  // untimed warmup
  {
    DeviceZoneScopedN("pack_unpack_loop");
    for (uint32_t it = 0; it < iters; ++it) {
      copy_block(dfb_loop, dfb_out, tiles, cap);
    }
  }
  cb_pop_front(dfb_loop, tiles);
  cb_push_back(dfb_out, tiles);  // writer drains dfb_out directly
}
