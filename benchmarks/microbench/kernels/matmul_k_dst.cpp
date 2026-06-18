// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MB3, DST-K matmul accumulation: C[mt,nt] = sum_k A[k] @ B[k] over `kt`
// K-tiles. The output is tiled into sub_mt*sub_nt subblocks; each subblock
// stays resident in DST across the whole K loop (one acquire, matmul_block
// accumulating, pack once). When the full mt*nt output fits DST (sub == mt,nt;
// MB3.A) there is one subblock and operands are unpacked once. When it does not
// (MB3.B), each subblock re-unpacks its A rows and B cols from the resident
// operand DFBs, so total operand unpack scales by the subblock count (reuse).
//
// With fuse != 0 a GELU epilogue is applied in place on the DST subblock before
// the single pack -- no reload, because the output is already resident. The
// fast (tanh) GELU is used, the production default for fused matmul epilogues.
//
// A is resident as `kt` column-blocks of mt tiles (A tile (m,k) at k*mt+m); B
// as `kt` row-blocks of nt tiles (B tile (k,n) at k*nt+n).
//
// Compile-time args: 0 = mt, 1 = nt, 2 = kt, 3 = sub_mt, 4 = sub_nt, 5 = fuse.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t dfb_in0 = 0;
constexpr uint32_t dfb_in1 = 1;
constexpr uint32_t dfb_out = 16;

void kernel_main() {
  const uint32_t mt = get_compile_time_arg_val(0);
  const uint32_t nt = get_compile_time_arg_val(1);
  const uint32_t kt = get_compile_time_arg_val(2);
  const uint32_t sub_mt = get_compile_time_arg_val(3);
  const uint32_t sub_nt = get_compile_time_arg_val(4);
  const uint32_t fuse = get_compile_time_arg_val(5);
  const uint32_t out_tiles = mt * nt;
  const uint32_t sub_tiles = sub_mt * sub_nt;

  mm_block_init(dfb_in0, dfb_in1, dfb_out, 0, sub_nt, sub_mt, 1);

  cb_reserve_back(dfb_out, out_tiles);
  {
    DeviceZoneScopedN("matmul_k_loop");
    // Operands prefetched once and reused across all subblocks; handoff is in
    // the zone to match L1-K's per-K-step waits.
    cb_wait_front(dfb_in0, kt * mt);
    cb_wait_front(dfb_in1, kt * nt);
    for (uint32_t om = 0; om < mt; om += sub_mt) {
      for (uint32_t on = 0; on < nt; on += sub_nt) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < kt; ++k) {
          matmul_block(dfb_in0, dfb_in1, k * mt + om, k * nt + on, 0, 0, sub_nt,
                       sub_mt, 1);
        }
        if (fuse) {
          gelu_tile_init();
          for (uint32_t t = 0; t < sub_tiles; ++t) {
            gelu_tile(t);
          }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < sub_mt; ++i) {
          for (uint32_t j = 0; j < sub_nt; ++j) {
            // Out-of-order: place each subblock tile at its row-major C
            // position.
            pack_tile<true>(i * sub_nt + j, dfb_out, (om + i) * nt + (on + j));
          }
        }
        tile_regs_release();
      }
    }
    cb_pop_front(dfb_in0, kt * mt);
    cb_pop_front(dfb_in1, kt * nt);
  }
  cb_push_back(dfb_out, out_tiles);
}
