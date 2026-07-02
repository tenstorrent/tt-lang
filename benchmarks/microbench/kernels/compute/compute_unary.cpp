// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MB4 unary compute-op probe. Applies a selected SFPU unary op (math thread,
// the thread tt-lang targets) to `tiles` tiles, `iters` times, on a single
// compute core, to isolate the per-tile math-engine cost the data-movement
// benchmarks omit. The input is re-read from a resident DFB each iteration and
// the result overwrites the output, so the final output is op(input) (a valid
// PCC check) while the measured loop exercises the op `iters` times. op = 0
// (copy) is the baseline: subtract it to get the SFPU op's marginal math cost.
//
// init_hoist selects whether the op init is hoisted out of the loop (steady
// per-tile cost) or re-issued every sub-block (init + op cost) -- both occur in
// production kernels.
//
// An untimed warmup pass runs one full copy->op->pack over the tiles before the
// measured zone, priming the math/pack pipeline so the zone captures warm steady
// state. Without it the first in-zone pass pays a one-time spin-up that dominates
// at low iters (and that the passthrough probe avoids via its out-of-zone seed).
//
// Compile-time args: 0 = op, 1 = tiles, 2 = iters, 3 = DST capacity,
// 4 = init_hoist. op ids: 0 copy, 1 exp, 2 gelu, 3 recip, 4 sqrt, 5 rsqrt.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "api/compute/eltwise_unary/sqrt.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t dfb_in = 0;
constexpr uint32_t dfb_out = 16;

inline void op_init(uint32_t op) {
  if (op == 1) {
    exp_tile_init();
  } else if (op == 2) {
    gelu_tile_init();
  } else if (op == 3) {
    recip_tile_init();
  } else if (op == 4) {
    sqrt_tile_init();
  } else if (op == 5) {
    rsqrt_tile_init();
  }
}

inline void op_apply(uint32_t op, uint32_t t) {
  if (op == 1) {
    exp_tile(t);
  } else if (op == 2) {
    gelu_tile(t);
  } else if (op == 3) {
    recip_tile(t);
  } else if (op == 4) {
    sqrt_tile(t);
  } else if (op == 5) {
    rsqrt_tile(t);
  }
  // op == 0: copy only (no SFPU), the baseline.
}

// One chunked copy->op->pack pass over `tiles` tiles in DST-capacity sub-blocks.
// Shared by the untimed warmup pass and the measured loop so both are identical.
inline void compute_pass(uint32_t op, uint32_t tiles, uint32_t cap,
                         uint32_t init_hoist) {
  for (uint32_t base = 0; base < tiles; base += cap) {
    uint32_t chunk = (tiles - base < cap) ? (tiles - base) : cap;
    if (!init_hoist) {
      copy_tile_init(dfb_in);
      op_init(op);
    }
    tile_regs_acquire();
    for (uint32_t i = 0; i < chunk; ++i) {
      copy_tile(dfb_in, base + i, i);
    }
    for (uint32_t i = 0; i < chunk; ++i) {
      op_apply(op, i);
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
  const uint32_t op = get_compile_time_arg_val(0);
  const uint32_t tiles = get_compile_time_arg_val(1);
  const uint32_t iters = get_compile_time_arg_val(2);
  const uint32_t cap = get_compile_time_arg_val(3);
  const uint32_t init_hoist = get_compile_time_arg_val(4);

  init_sfpu(dfb_in, dfb_out);
  copy_tile_init(dfb_in);
  if (init_hoist) {
    op_init(op);
  }

  cb_wait_front(dfb_in, tiles);    // resident input, re-read each iteration
  cb_reserve_back(dfb_out, tiles); // output region, overwritten each iteration

  // Untimed warmup pass: prime the math/pack pipeline before the measured zone
  // so it captures warm steady state (matching the passthrough probe's
  // out-of-zone seed). The measured loop re-reads the resident input and
  // overwrites the output, so this extra pass does not change the final result.
  compute_pass(op, tiles, cap, init_hoist);

  {
    DeviceZoneScopedN("compute_op_loop");
    for (uint32_t it = 0; it < iters; ++it) {
      compute_pass(op, tiles, cap, init_hoist);
    }
  }
  cb_pop_front(dfb_in, tiles);
  cb_push_back(dfb_out, tiles);
}
