// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MB2, L1-pack accumulation: out = initial + sum of `iters` contributions, with
// the accumulator living in L1 and re-packed every step. The initial value is
// packed once with pack_reconfig_l1_acc(0) (overwrite); each contribution is
// then copied into DST and packed with pack_reconfig_l1_acc(1) (packer
// L1-accumulate), so the accumulator round-trips through L1 every step. The
// seed and the contributions use SEPARATE DFBs, matching the DST kernel.
//
// Compile-time args: 0 = accumulator tiles, 1 = contributions, 2 = DST
// capacity, 3 = reuse (1 = contributions L1-resident: re-read one block every
// iteration;
//  0 = streamed: consume a fresh contribution block each iteration).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t dfb_init = 0;  // reader -> compute (initial value)
constexpr uint32_t dfb_delta = 1; // reader -> compute (contributions)
constexpr uint32_t dfb_out = 16;  // compute -> writer (accumulator in L1)

// Copy acc_tiles tiles from src_cb into DST in DST-capacity sub-blocks and pack
// them to dfb_out (packer L1-accumulation governed by the caller's reconfig).
inline void copy_and_pack(uint32_t src_cb, uint32_t acc_tiles, uint32_t cap) {
  for (uint32_t base = 0; base < acc_tiles; base += cap) {
    uint32_t chunk = (acc_tiles - base < cap) ? (acc_tiles - base) : cap;
    tile_regs_acquire();
    for (uint32_t i = 0; i < chunk; ++i) {
      copy_tile(src_cb, base + i, i);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t i = 0; i < chunk; ++i) {
      pack_tile<true>(i, dfb_out, base + i);
    }
    tile_regs_release();
  }
}

void kernel_main() {
  const uint32_t acc_tiles = get_compile_time_arg_val(0);
  const uint32_t iters = get_compile_time_arg_val(1);
  const uint32_t cap = get_compile_time_arg_val(2);
  const uint32_t reuse = get_compile_time_arg_val(3);

  init_sfpu(dfb_init, dfb_out);
  copy_tile_init(dfb_init);

  cb_reserve_back(dfb_out, acc_tiles);
  pack_reconfig_l1_acc(0); // initial value overwrites the L1 accumulator
  {
    DeviceZoneScopedN("acc_loop");
    cb_wait_front(dfb_init, acc_tiles);
    copy_and_pack(dfb_init, acc_tiles, cap);
    cb_pop_front(dfb_init, acc_tiles);
    pack_reconfig_l1_acc(1); // contributions accumulate in L1
    for (uint32_t it = 0; it < iters; ++it) {
      cb_wait_front(dfb_delta, acc_tiles);
      copy_and_pack(dfb_delta, acc_tiles, cap);
      if (!reuse) {
        cb_pop_front(dfb_delta, acc_tiles);
      }
    }
    if (reuse) {
      cb_pop_front(dfb_delta, acc_tiles);
    }
  }
  pack_reconfig_l1_acc(0);
  cb_push_back(dfb_out, acc_tiles);
}
