// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MB4 binary compute-op probe. Applies a selected two-operand FPU op to `tiles`
// tiles, `iters` times, on a single compute core, to isolate the per-tile
// compute-engine cost the data-movement benchmarks omit. The operands are
// re-read from resident DFBs each iteration and the result overwrites the
// output, so the final output is op(x, y) (a valid PCC check) while the measured
// loop exercises the op `iters` times.
//
// Two op families share the binary (two-operand) path:
//   10 add, 11 mul                 -- full second operand (cb1 = tiles tiles)
//   20 mul_bcast_cols, 21 sub_bcast_cols
//                                  -- broadcast operand (cb1 = 1 tile, col 0)
// The bcast variant differs only in cb1's tile count (1 vs tiles) and indexing
// (always tile 0), so it folds into the same loop. The second operand (cb1) is
// all ones, so values never affect timing and the PCC ref is trivial
// (add -> x+1, mul/mul_bcast -> x, sub_bcast -> x-1).
//
// init_hoist hoists the op init out of the loop (steady per-tile cost) vs
// re-issuing it every sub-block (init + op) -- both occur in production kernels.
//
// Compile-time args: 0 = op, 1 = tiles, 2 = iters, 3 = DST capacity,
// 4 = init_hoist.

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t cb0 = 0;
constexpr uint32_t cb1 = 1;
constexpr uint32_t cb_out = 16;

inline void op_init(uint32_t op) {
  if (op == 10) {
    add_tiles_init(cb0, cb1);
  } else if (op == 11) {
    mul_tiles_init(cb0, cb1);
  } else if (op == 20) {
    mul_bcast_cols_init_short(cb0, cb1);
  } else if (op == 21) {
    sub_bcast_cols_init_short(cb0, cb1);
  }
}

inline void op_apply(uint32_t op, uint32_t in0_idx, uint32_t dst) {
  if (op == 10) {
    add_tiles(cb0, cb1, in0_idx, in0_idx, dst);
  } else if (op == 11) {
    mul_tiles(cb0, cb1, in0_idx, in0_idx, dst);
  } else if (op == 20) {
    mul_tiles_bcast_cols(cb0, cb1, in0_idx, 0, dst);
  } else if (op == 21) {
    sub_tiles_bcast_cols(cb0, cb1, in0_idx, 0, dst);
  }
}

void kernel_main() {
  const uint32_t op = get_compile_time_arg_val(0);
  const uint32_t tiles = get_compile_time_arg_val(1);
  const uint32_t iters = get_compile_time_arg_val(2);
  const uint32_t cap = get_compile_time_arg_val(3);
  const uint32_t init_hoist = get_compile_time_arg_val(4);

  const uint32_t n1 = op >= 20 ? 1u : tiles; // bcast operand vs full operand

  binary_op_init_common(cb0, cb1, cb_out);
  cb_wait_front(cb1, n1);
  if (init_hoist) {
    op_init(op);
  }

  cb_wait_front(cb0, tiles);       // resident input, re-read each iteration
  cb_reserve_back(cb_out, tiles);  // output region, overwritten each iteration
  {
    DeviceZoneScopedN("compute_op_loop");
    for (uint32_t it = 0; it < iters; ++it) {
      for (uint32_t base = 0; base < tiles; base += cap) {
        uint32_t chunk = (tiles - base < cap) ? (tiles - base) : cap;
        if (!init_hoist) {
          binary_op_init_common(cb0, cb1, cb_out);
          op_init(op);
        }
        tile_regs_acquire();
        for (uint32_t i = 0; i < chunk; ++i) {
          op_apply(op, base + i, i);
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < chunk; ++i) {
          pack_tile<true>(i, cb_out, base + i);
        }
        tile_regs_release();
      }
    }
  }
  cb_pop_front(cb0, tiles);
  cb_pop_front(cb1, n1);
  cb_push_back(cb_out, tiles);
}
