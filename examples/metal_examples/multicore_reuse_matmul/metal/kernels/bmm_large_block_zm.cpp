// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/tile_move_copy.h"

namespace NAMESPACE {
void MAIN {
  // inner block size in tiles
  uint32_t K_block_size = get_compile_time_arg_val(0);
  // outer row block size (in inner row blocks)
  uint32_t a_num_subblocks = get_compile_time_arg_val(1);
  // out_subblock_h*K_block_size*a_num_subblocks;
  uint32_t a_block_num_tiles = get_compile_time_arg_val(2);
  // out_subblock_h*K_block_size
  uint32_t a_subblock_num_tiles = get_compile_time_arg_val(3);
  // outer column block size (in inner column blocks)
  uint32_t b_num_subblocks = get_compile_time_arg_val(4);
  // out_subblock_w*K_block_size* b_num_subblocks;
  uint32_t b_block_num_tiles = get_compile_time_arg_val(5);
  // out_subblock_w*b_num_subblocks
  uint32_t b_per_core_w = get_compile_time_arg_val(6);
  // outer inner dim (in inner dim blocks)
  uint32_t num_blocks = get_compile_time_arg_val(7);
  // inner row block size in tiles
  uint32_t out_subblock_h = get_compile_time_arg_val(8);
  // inner column block size in tiles
  uint32_t out_subblock_w = get_compile_time_arg_val(9);
  // out_subblock_h * out_subblock_w;
  uint32_t out_subblock_num_tiles = get_compile_time_arg_val(10);

  mm_init(tt::CBIndex::c_0, tt::CBIndex::c_1, tt::CBIndex::c_16);

  bool spill = num_blocks > 1;
  bool enable_reload = false;
  uint32_t out_num_tiles_to_wait = out_subblock_num_tiles;

  for (uint32_t block = 0; block < num_blocks; block++) {
    bool last_out = block == (num_blocks - 1);

    cb_wait_front(tt::CBIndex::c_0, a_block_num_tiles);
    cb_wait_front(tt::CBIndex::c_1, b_block_num_tiles);
    // start of being handled by compiler
    int a_index_subblock_offset = 0;
    for (uint32_t a_subblock = 0; a_subblock < a_num_subblocks; a_subblock++) {
      int b_index_subblock_offset = 0;
      for (uint32_t b_subblock = 0; b_subblock < b_num_subblocks;
           b_subblock++) {
        acquire_dst();

        if (enable_reload) {
          copy_tile_to_dst_init_short(tt::CBIndex::c_24);
          cb_wait_front(tt::CBIndex::c_24, out_subblock_num_tiles);
          for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
            copy_tile(tt::CBIndex::c_24, i, i);
          }
          cb_pop_front(tt::CBIndex::c_24, out_subblock_num_tiles);
          mm_init_short(tt::CBIndex::c_0, tt::CBIndex::c_1);
        }

        // Compute output sub-block from a_subblock x b_subblock
        int dst_index = 0;
        int a_index_h_offset = 0;
        for (uint32_t h = 0; h < out_subblock_h; h++) {
          for (uint32_t w = 0; w < out_subblock_w; w++) {
            int b_index_inner_dim_offset = 0;
            for (uint32_t inner_dim = 0; inner_dim < K_block_size;
                 inner_dim++) {
              int a_index =
                  a_index_subblock_offset + a_index_h_offset + inner_dim;
              int b_index =
                  b_index_subblock_offset + b_index_inner_dim_offset + w;
              matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_1, a_index, b_index,
                           dst_index);
              b_index_inner_dim_offset += b_per_core_w;
            }
            dst_index++;
          }
          a_index_h_offset += K_block_size;
        }

        if (last_out) {
          // Pack out to output buffer
          cb_reserve_back(tt::CBIndex::c_16, out_subblock_num_tiles);
          for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
            pack_tile(i, tt::CBIndex::c_16);
          }
          cb_push_back(tt::CBIndex::c_16, out_subblock_num_tiles);
        } else {
          // Wait for tiles in output buffer to be written out since interm
          // and output share memory
          if (block == 0) {
            cb_reserve_back(tt::CBIndex::c_16, out_num_tiles_to_wait);
            out_num_tiles_to_wait += out_subblock_num_tiles;
          }
          // Move partial result to interm buffer
          cb_reserve_back(tt::CBIndex::c_24, out_subblock_num_tiles);
          for (uint32_t i = 0; i < out_subblock_num_tiles; i++) {
            pack_tile(i, tt::CBIndex::c_24);
          }
          cb_push_back(tt::CBIndex::c_24, out_subblock_num_tiles);
        }

        release_dst();

        b_index_subblock_offset += out_subblock_w;
      }
      a_index_subblock_offset += a_subblock_num_tiles;
    }

    if (spill) {
      enable_reload = true;
    }
    // end of compiler subblock generation
    cb_pop_front(tt::CBIndex::c_0, a_block_num_tiles);
    cb_pop_front(tt::CBIndex::c_1, b_block_num_tiles);
  }
}
} // namespace NAMESPACE
