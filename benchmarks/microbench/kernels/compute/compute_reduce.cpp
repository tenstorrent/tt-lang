// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MB4 reduce compute-op probe. Row-reduces the `tiles`-tile row into a single
// output tile (the reduction lands in column 0), `iters` times, on a single
// compute core, to isolate the per-tile reduce cost the data-movement
// benchmarks omit. The input is re-read from a resident DFB each iteration and
// the single-tile result overwrites the output, so the final output is the
// row reduction (a valid PCC check) while the measured loop exercises the
// reduce `iters` times.
//
//   30 sum_row, 31 max_row    -- cb0 = tiles tiles reduced into 1 tile,
//                                cb1 = scaler (1 tile of ones)
//
// The scaler (cb1) is all ones, so it never affects timing and the PCC ref is a
// plain rowsum/rowmax. The reduce init is always hoisted (reduce_init before the
// loop, reduce_uninit after), so init_hoist is accepted for arg-layout parity
// with the unary/binary probes but ignored. The whole row reduces into DST tile
// 0, so the DST capacity arg is likewise unused.
//
// Compile-time args: 0 = op, 1 = tiles, 2 = iters, 3 = DST capacity (unused),
// 4 = init_hoist (unused -- reduce always hoists).

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t cb0 = 0;
constexpr uint32_t cb1 = 1;
constexpr uint32_t cb_out = 16;

void kernel_main() {
  const uint32_t op = get_compile_time_arg_val(0);
  const uint32_t tiles = get_compile_time_arg_val(1);
  const uint32_t iters = get_compile_time_arg_val(2);

  cb_wait_front(cb0, tiles);    // resident input, re-read each iteration
  cb_wait_front(cb1, 1u);       // scaler (ones)
  cb_reserve_back(cb_out, 1u);  // single-tile output, overwritten each iteration
  if (op == 30) {
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, false>(cb0, cb1, cb_out);
  } else {
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_ROW, false>(cb0, cb1, cb_out);
  }
  {
    DeviceZoneScopedN("compute_op_loop");
    for (uint32_t it = 0; it < iters; ++it) {
      tile_regs_acquire();
      if (op == 30) {
        for (uint32_t t = 0; t < tiles; ++t)
          reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, false>(cb0, cb1, t,
                                                                   0, 0);
      } else {
        for (uint32_t t = 0; t < tiles; ++t)
          reduce_tile<PoolType::MAX, ReduceDim::REDUCE_ROW, false>(cb0, cb1, t,
                                                                   0, 0);
      }
      tile_regs_commit();
      tile_regs_wait();
      pack_tile<true>(0, cb_out, 0);
      tile_regs_release();
    }
  }
  reduce_uninit();
  cb_pop_front(cb0, tiles);
  cb_pop_front(cb1, 1u);
  cb_push_back(cb_out, 1u);
}
