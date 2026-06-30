// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// mm5_block_stream_l1acc_packblock: mm4 (matmul_block + K streaming + packer L1
// accumulation) with one change -- each output subblock is packed with a single
// pack_tile_block call instead of a per-tile pack_tile loop, matching TTNN's
// block pack. This reduces pack-thread instruction overhead, which is the
// trisc_max bottleneck once the math thread is fed efficiently.
//
// Compile-time args: 0 = mt, 1 = nt, 2 = kt, 3 = in0_block_w, 4 = sub_mt,
// 5 = sub_nt.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t dfb_in0 = 0;
constexpr uint32_t dfb_in1 = 1;
constexpr uint32_t dfb_out = 4;
constexpr uint32_t dfb_intermed = 5;

// Reload one accumulated output subblock from the partials DFB into DST so the
// last K block adds its contribution on top. Used only on the last block.
static FORCE_INLINE void reload_partials(uint32_t subblock_tiles,
                                         uint32_t sub_nt, uint32_t sub_mt,
                                         uint32_t in0_block_w) {
  copy_tile_to_dst_init_short_with_dt(dfb_in1, dfb_intermed);
  cb_wait_front(dfb_intermed, subblock_tiles);
  for (uint32_t tile_index = 0; tile_index < subblock_tiles; ++tile_index) {
    copy_tile(dfb_intermed, tile_index, tile_index);
  }
  cb_pop_front(dfb_intermed, subblock_tiles);
  mm_block_init_short_with_dt(dfb_in0, dfb_in1, dfb_intermed, false, sub_nt,
                              sub_mt, in0_block_w);
}

void kernel_main() {
  const uint32_t mt = get_compile_time_arg_val(0);
  const uint32_t nt = get_compile_time_arg_val(1);
  const uint32_t kt = get_compile_time_arg_val(2);
  const uint32_t in0_block_w = get_compile_time_arg_val(3);
  const uint32_t sub_mt = get_compile_time_arg_val(4);
  const uint32_t sub_nt = get_compile_time_arg_val(5);

  const uint32_t num_blocks = kt / in0_block_w;
  const uint32_t in0_block_tiles = mt * in0_block_w;
  const uint32_t in1_block_tiles = nt * in0_block_w;
  const uint32_t out_block_tiles = mt * nt;
  const uint32_t subblock_tiles = sub_mt * sub_nt;
  const uint32_t in0_subblock_tiles = sub_mt * in0_block_w;

  mm_block_init(dfb_in0, dfb_in1, dfb_intermed, false, sub_nt, sub_mt,
                in0_block_w);

  {
    DeviceZoneScopedN("matmul_compute_loop");
    bool enable_reload = false;
    for (uint32_t block_index = 0; block_index < num_blocks; ++block_index) {
      const bool last_block = block_index == num_blocks - 1;
      cb_wait_front(dfb_in0, in0_block_tiles);
      cb_wait_front(dfb_in1, in1_block_tiles);

      uint32_t in0_subblock_offset = 0;
      for (uint32_t output_row = 0; output_row < mt; output_row += sub_mt) {
        uint32_t in1_subblock_offset = 0;
        for (uint32_t output_col = 0; output_col < nt; output_col += sub_nt) {
          tile_regs_acquire();
          if (enable_reload) {
            reload_partials(subblock_tiles, sub_nt, sub_mt, in0_block_w);
          }

          uint32_t dst_index = 0;
          uint32_t in0_index = in0_subblock_offset;
          uint32_t in1_index = in1_subblock_offset;
          for (uint32_t inner_dim_index = 0; inner_dim_index < in0_block_w;
               ++inner_dim_index) {
            matmul_block(dfb_in0, dfb_in1, in0_index, in1_index, dst_index,
                         false, sub_nt, sub_mt, in0_block_w);
            ++in0_index;
            in1_index += nt;
          }

          tile_regs_commit();
          const uint32_t pack_dfb = last_block ? dfb_out : dfb_intermed;
          cb_reserve_back(pack_dfb, subblock_tiles);
          tile_regs_wait();
          if (last_block) {
            pack_reconfig_l1_acc(0);
          } else if (block_index == 0) {
            pack_reconfig_l1_acc(0);
          } else {
            pack_reconfig_l1_acc(1);
          }
          pack_tile_block(0, pack_dfb, subblock_tiles);
          tile_regs_release();
          cb_push_back(pack_dfb, subblock_tiles);

          in1_subblock_offset += sub_nt;
        }
        in0_subblock_offset += in0_subblock_tiles;
      }

      if (block_index + 2 < num_blocks) {
        for (uint32_t tile_index = 0; tile_index < out_block_tiles;
             tile_index += subblock_tiles) {
          cb_wait_front(dfb_intermed, subblock_tiles);
          cb_pop_front(dfb_intermed, subblock_tiles);
        }
      }
      if (block_index + 2 == num_blocks) {
        enable_reload = true;
      }
      cb_pop_front(dfb_in0, in0_block_tiles);
      cb_pop_front(dfb_in1, in1_block_tiles);
    }
  }
}
