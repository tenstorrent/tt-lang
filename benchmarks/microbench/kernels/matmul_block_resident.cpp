// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MB3 resident-operand matmul_block roofline probe. Same matmul issue as the
// ttnn_like kernel (one matmul_block per K tile per output subblock, the whole
// sub_mt x sub_nt grid in one MOP call), but A and B are waited for OUTSIDE the
// timed zone, so the zone measures matmul + pack only and excludes the upfront
// operand DRAM load. Single K block: the full kt is resident, so kt_dim is the
// A row stride (kt) and there is no cross-block accumulation.
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
constexpr uint32_t dfb_out = 4;

void kernel_main() {
  const uint32_t mt = get_compile_time_arg_val(0);
  const uint32_t nt = get_compile_time_arg_val(1);
  const uint32_t kt = get_compile_time_arg_val(2);
  const uint32_t sub_mt = get_compile_time_arg_val(3);
  const uint32_t sub_nt = get_compile_time_arg_val(4);
  const uint32_t subblock_tiles = sub_mt * sub_nt;
  const uint32_t in0_subblock_tiles = sub_mt * kt;

  mm_block_init(dfb_in0, dfb_in1, dfb_out, false, sub_nt, sub_mt, kt);

  // Operands resident before the timed zone (load excluded from the measurement).
  cb_wait_front(dfb_in0, mt * kt);
  cb_wait_front(dfb_in1, kt * nt);

  {
    DeviceZoneScopedN("matmul_compute_loop");
    uint32_t in0_subblock_offset = 0;
    for (uint32_t output_row = 0; output_row < mt; output_row += sub_mt) {
      uint32_t in1_subblock_offset = 0;
      for (uint32_t output_col = 0; output_col < nt; output_col += sub_nt) {
        tile_regs_acquire();
        uint32_t dst_index = 0;
        uint32_t in0_index = in0_subblock_offset;
        uint32_t in1_index = in1_subblock_offset;
        for (uint32_t inner_dim_index = 0; inner_dim_index < kt;
             ++inner_dim_index) {
          matmul_block(dfb_in0, dfb_in1, in0_index, in1_index, dst_index, false,
                       sub_nt, sub_mt, kt);
          ++in0_index;
          in1_index += nt;
        }
        tile_regs_commit();
        cb_reserve_back(dfb_out, subblock_tiles);
        tile_regs_wait();
        for (uint32_t tile_index = 0; tile_index < subblock_tiles;
             ++tile_index) {
          pack_tile(tile_index, dfb_out);
        }
        tile_regs_release();
        cb_push_back(dfb_out, subblock_tiles);
        in1_subblock_offset += sub_nt;
      }
      in0_subblock_offset += in0_subblock_tiles;
    }
  }

  cb_pop_front(dfb_in0, mt * kt);
  cb_pop_front(dfb_in1, kt * nt);
}
