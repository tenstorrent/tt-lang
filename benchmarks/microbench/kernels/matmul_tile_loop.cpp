// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Diagnostic compute-feed matmul probe. This is intentionally closer to tt-metal's
// large-block matmul compute kernel than the DST-K/L1-K strategy benchmark:
// operands are already resident, A/B use row-major per-node block layout, and
// the K tile block is consumed inside one output-block loop.
//
// Compile-time args: 0 = mt, 1 = nt, 2 = kt, 3 = sub_mt, 4 = sub_nt.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
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
  const uint32_t out_tiles = mt * nt;

  mm_init(dfb_in0, dfb_in1, dfb_out);

  cb_wait_front(dfb_in0, mt * kt);
  cb_wait_front(dfb_in1, kt * nt);
  cb_reserve_back(dfb_out, out_tiles);

  {
    DeviceZoneScopedN("matmul_compute_loop");
    for (uint32_t output_row = 0; output_row < mt; output_row += sub_mt) {
      for (uint32_t output_col = 0; output_col < nt; output_col += sub_nt) {
        tile_regs_acquire();
        uint32_t dst_index = 0;
        for (uint32_t subblock_row = 0; subblock_row < sub_mt;
             ++subblock_row) {
          for (uint32_t subblock_col = 0; subblock_col < sub_nt;
               ++subblock_col) {
            for (uint32_t k_index = 0; k_index < kt; ++k_index) {
              matmul_tiles(dfb_in0, dfb_in1,
                           (output_row + subblock_row) * kt + k_index,
                           k_index * nt + output_col + subblock_col,
                           dst_index);
            }
            ++dst_index;
          }
        }
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t subblock_row = 0; subblock_row < sub_mt;
             ++subblock_row) {
          for (uint32_t subblock_col = 0; subblock_col < sub_nt;
               ++subblock_col) {
            pack_tile<true>(
                subblock_row * sub_nt + subblock_col, dfb_out,
                (output_row + subblock_row) * nt + (output_col + subblock_col));
          }
        }
        tile_regs_release();
      }
    }
  }

  cb_pop_front(dfb_in0, mt * kt);
  cb_pop_front(dfb_in1, kt * nt);
  cb_push_back(dfb_out, out_tiles);
}
