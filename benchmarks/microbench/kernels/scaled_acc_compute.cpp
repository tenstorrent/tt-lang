// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Parameterized scaled-accumulate matmul (generalized from test_matmul_scaled_
// acc.py's `out = scale*acc + (a @ b)`), with the output subblock (sub_mt x
// sub_nt) AND the K-depth (Kt) exposed as compile-time args. Per subblock,
// mul_tiles precomputes scale*acc into the DST slots, then a Kt-step matmul_block
// loop accumulates a @ b onto them (the add vanishes), then pack. a is (Mt, Kt),
// b is (Kt, Nt), both row-major; kt_dim = Kt is the A row stride.
//
// scale*acc is a pre-seeded accumulator (a prologue), not an epilogue: it shares
// the matmul's DST slots, so it needs no scratch. A capacity heuristic that
// nonetheless reserved ~cap/2 for it would pick a smaller subblock; this kernel
// lets us measure whether that conservative pick is optimal, across K-depths.
//
// Compile-time args: 0 = Mt, 1 = Nt, 2 = Kt, 3 = sub_mt, 4 = sub_nt.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t cb_a = 0;
constexpr uint32_t cb_b = 1;
constexpr uint32_t cb_scale = 2;
constexpr uint32_t cb_acc = 3;
constexpr uint32_t cb_out = 16;

void kernel_main() {
  const uint32_t Mt = get_compile_time_arg_val(0);
  const uint32_t Nt = get_compile_time_arg_val(1);
  const uint32_t Kt = get_compile_time_arg_val(2);
  const uint32_t sub_mt = get_compile_time_arg_val(3);
  const uint32_t sub_nt = get_compile_time_arg_val(4);
  const uint32_t out_tiles = Mt * Nt;

  cb_wait_front(cb_a, Mt * Kt);
  cb_wait_front(cb_b, Kt * Nt);
  cb_wait_front(cb_scale, out_tiles);
  cb_wait_front(cb_acc, out_tiles);
  cb_reserve_back(cb_out, out_tiles);

  // Top init configures unpack for the mul operands (scale, acc) + pack to out.
  mm_block_init(cb_scale, cb_acc, cb_out, false, sub_nt, sub_mt, Kt);

  {
    DeviceZoneScopedN("scaled_acc_loop");
    for (uint32_t om = 0; om < Mt; om += sub_mt) {
      for (uint32_t on = 0; on < Nt; on += sub_nt) {
        tile_regs_acquire();

        // scale * acc -> DST subblock slots [0 .. sub_mt*sub_nt)
        mul_tiles_init(cb_scale, cb_acc);
        for (uint32_t i = 0; i < sub_mt; ++i) {
          for (uint32_t j = 0; j < sub_nt; ++j) {
            const uint32_t idx = (om + i) * Nt + (on + j);
            mul_tiles(cb_scale, cb_acc, idx, idx, i * sub_nt + j);
          }
        }

        // a @ b accumulates onto the resident scale*acc (matmul_block adds into
        // DST), so the `+` from `scale*acc + a@b` vanishes. K-loop over Kt; a
        // tile (m,k) at m*Kt+k, b tile (k,n) at k*Nt+n, kt_dim = Kt (A row stride).
        mm_block_init_short(cb_a, cb_b, false, sub_nt, sub_mt, Kt);
        for (uint32_t k = 0; k < Kt; ++k) {
          matmul_block(cb_a, cb_b, om * Kt + k, k * Nt + on, 0, false, sub_nt,
                       sub_mt, Kt);
        }

        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < sub_mt; ++i) {
          for (uint32_t j = 0; j < sub_nt; ++j) {
            pack_tile<true>(i * sub_nt + j, cb_out, (om + i) * Nt + (on + j));
          }
        }
        tile_regs_release();
      }
    }
  }

  cb_push_back(cb_out, out_tiles);
  cb_pop_front(cb_a, Mt * Kt);
  cb_pop_front(cb_b, Kt * Nt);
  cb_pop_front(cb_scale, out_tiles);
  cb_pop_front(cb_acc, out_tiles);
}
