// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MB2, DST-resident accumulation: out = initial + sum of `iters` contributions.
// The accumulator lives in DST across the loop (one acquire). It is seeded from
// the initial DFB with copy_tile, then each contribution is added in place with
// binary_dest_reuse_tiles<ELWADD, DEST_TO_SRCA> — one in-place op that unpacks
// the contribution straight from the contribution DFB. The seed and the
// contributions use SEPARATE DFBs, matching tt-lang's tile_accumulate_add
// lowering (initial_dfb + delta_dfb); optimized production kernels likewise use
// separate seed and contribution DFBs. The result is packed once; no
// per-iteration pack to L1. Accumulator uses acc_tiles DST tiles, so acc_tiles
// must fit getDstCapacity.
//
// Compile-time args: 0 = accumulator tiles, 1 = contributions, 2 = reuse
// (1 = contributions L1-resident: re-read one block every iteration;
//  0 = streamed: consume a fresh contribution block each iteration).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t dfb_init = 0;  // reader -> compute (initial value)
constexpr uint32_t dfb_delta = 1; // reader -> compute (contributions)
constexpr uint32_t dfb_out = 16;  // compute -> writer (result)

void kernel_main() {
  const uint32_t acc_tiles = get_compile_time_arg_val(0);
  const uint32_t iters = get_compile_time_arg_val(1);
  const uint32_t reuse = get_compile_time_arg_val(2);

  binary_op_init_common(dfb_delta, dfb_delta,
                        dfb_out); // set up the binary datapath
  copy_tile_init(dfb_init);

  tile_regs_acquire(); // accumulator persists in DST across the whole loop
  {
    DeviceZoneScopedN("acc_loop");
    // Seed the accumulator from the initial value.
    cb_wait_front(dfb_init, acc_tiles);
    for (uint32_t u = 0; u < acc_tiles; ++u) {
      copy_tile(dfb_init, u, u);
    }
    cb_pop_front(dfb_init, acc_tiles);
    // Add each contribution in place.
    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD,
                                 EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
        dfb_delta);
    for (uint32_t it = 0; it < iters; ++it) {
      cb_wait_front(dfb_delta, acc_tiles);
      for (uint32_t u = 0; u < acc_tiles; ++u) {
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD,
                                EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
            dfb_delta, u, u);
      }
      if (!reuse) {
        cb_pop_front(dfb_delta, acc_tiles);
      }
    }
    if (reuse) {
      cb_pop_front(dfb_delta, acc_tiles);
    }
  }
  tile_regs_commit();
  tile_regs_wait();
  cb_reserve_back(dfb_out, acc_tiles);
  for (uint32_t u = 0; u < acc_tiles; ++u) {
    pack_tile<true>(u, dfb_out, u);
  }
  cb_push_back(dfb_out, acc_tiles);
  tile_regs_release();
}
