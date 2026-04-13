// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"
#include "internal/mod_div_lib.h"

namespace NAMESPACE {

void MAIN {
  constexpr uint32_t in0_block_w =
      get_compile_time_arg_val(0); // inner block size in tiles
  constexpr uint32_t in0_num_subblocks =
      get_compile_time_arg_val(1); // outer row block size (in inner row blocks)
  constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(
      2); // out_subblock_h*in0_block_w*in0_num_subblocks;
  constexpr uint32_t in0_subblock_num_tiles =
      get_compile_time_arg_val(3); // out_subblock_h*in0_block_w
  constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(
      4); // outer column block size (in inner column blocks)
  constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(
      5); // out_subblock_w*in0_block_w* in1_num_subblocks;
  constexpr uint32_t in1_block_w =
      get_compile_time_arg_val(6); // out_subblock_w*in1_num_subblocks
  constexpr uint32_t num_blocks_inner_dim =
      get_compile_time_arg_val(7); // outer inner dim (in inner dim blocks)
  constexpr uint32_t num_blocks_w_dim =
      get_compile_time_arg_val(8); // outer inner dim (in inner dim blocks)
  constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(9);
  constexpr uint32_t out_subblock_h =
      get_compile_time_arg_val(10); // inner row block size in tiles
  constexpr uint32_t out_subblock_w =
      get_compile_time_arg_val(11); // inner column block size in tiles
  constexpr uint32_t out_subblock_num_tiles =
      get_compile_time_arg_val(12); // out_subblock_h * out_subblock_w;

  constexpr uint32_t out_block_w = out_subblock_w * in1_num_subblocks;

  constexpr uint32_t in0_cb_id = tt::CBIndex::c_0;
  constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;
  constexpr uint32_t out_cb_id = tt::CBIndex::c_16;
  constexpr uint32_t mm_partials_cb_id = tt::CBIndex::c_24;

  mm_init(in0_cb_id, in1_cb_id, out_cb_id);

  bool spill = num_blocks_inner_dim > 1;

  for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
    for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
      bool enable_reload = false;
      uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;
      for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
        bool last_out = block == (num_blocks_inner_dim - 1);
        cb_wait_front(in0_cb_id, in0_block_num_tiles);
        cb_wait_front(in1_cb_id, in1_block_num_tiles);

        // start of being handled by compiler
        int a_index_subblock_offset = 0;
        for (uint32_t a_subblock = 0; a_subblock < in0_num_subblocks;
             a_subblock++) {
          int b_index_subblock_offset = 0;
          for (uint32_t b_subblock = 0; b_subblock < in1_num_subblocks;
               b_subblock++) {
            tile_regs_acquire();

            if (enable_reload) {
              copy_tile_to_dst_init_short(mm_partials_cb_id);
              cb_wait_front(mm_partials_cb_id, out_subblock_num_tiles);
              for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                copy_tile(mm_partials_cb_id, i, i);
              }
              cb_pop_front(mm_partials_cb_id, out_subblock_num_tiles);
              mm_init_short(in0_cb_id, in1_cb_id);
            }

            // Compute output sub-block from a_subblock x b_subblock
            int dst_index = 0;
            int a_index_h_offset = 0;
            for (uint32_t h = 0; h < out_subblock_h; h++) {
              for (uint32_t w = 0; w < out_subblock_w; w++) {
                int b_index_inner_dim_offset = 0;
                for (uint32_t inner_dim = 0; inner_dim < in0_block_w;
                     inner_dim++) {
                  int a_index =
                      a_index_subblock_offset + a_index_h_offset + inner_dim;
                  int b_index =
                      b_index_subblock_offset + b_index_inner_dim_offset + w;
                  matmul_tiles(in0_cb_id, in1_cb_id, a_index, b_index,
                               dst_index);
                  b_index_inner_dim_offset += in1_block_w;
                }
                dst_index++;
              }
              a_index_h_offset += in0_block_w;
            }
            tile_regs_commit();
            tile_regs_wait();

            if (last_out) {
              // Pack out to output buffer
              cb_reserve_back(out_cb_id, out_subblock_num_tiles);
              for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                pack_tile(i, out_cb_id);
              }
              cb_push_back(out_cb_id, out_subblock_num_tiles);
            } else {
              // Wait for tiles in output buffer to be written out since interm
              // and output share memory
              if (block == 0) {
                cb_reserve_back(out_cb_id, out_num_tiles_to_wait);
                out_num_tiles_to_wait += out_subblock_num_tiles;
              }
              // Move partial result to interm buffer
              cb_reserve_back(mm_partials_cb_id, out_subblock_num_tiles);
              for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
                pack_tile(i, mm_partials_cb_id);
              }
              cb_push_back(mm_partials_cb_id, out_subblock_num_tiles);
            }
            tile_regs_release();
            b_index_subblock_offset += out_subblock_w;
          }
          a_index_subblock_offset += in0_subblock_num_tiles;
        }

        if (spill) {
          enable_reload = true;
        }
        // end of compiler subblock generation
        cb_pop_front(in0_cb_id, in0_block_num_tiles);
        cb_pop_front(in1_cb_id, in1_block_num_tiles);
      }
    }
  }
}
} // namespace NAMESPACE
