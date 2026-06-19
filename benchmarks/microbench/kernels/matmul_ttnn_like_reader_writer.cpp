// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// B-reader/output-writer for the TTNN-like compute-feed probe. This combines
// the roles TTNN assigns to the in1 sender/writer on a 1x1 grid. B is staged as
// `in0_block_w` rows by `nt` columns per K block; output subblocks are drained
// in the same order as the compute kernel packs them.
//
// Runtime args: 0 = B address, 1 = output address, 2 = mt, 3 = nt, 4 = kt,
// 5 = in0_block_w, 6 = sub_mt, 7 = sub_nt.

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  uint32_t b_address = get_arg_val<uint32_t>(0);
  uint32_t output_address = get_arg_val<uint32_t>(1);
  uint32_t mt = get_arg_val<uint32_t>(2);
  uint32_t nt = get_arg_val<uint32_t>(3);
  uint32_t kt = get_arg_val<uint32_t>(4);
  uint32_t in0_block_w = get_arg_val<uint32_t>(5);
  uint32_t sub_mt = get_arg_val<uint32_t>(6);
  uint32_t sub_nt = get_arg_val<uint32_t>(7);

  constexpr uint32_t dfb_in1 = tt::CBIndex::c_1;
  constexpr uint32_t dfb_out = tt::CBIndex::c_4;
  const uint32_t input_tile_bytes = get_tile_size(dfb_in1);
  const uint32_t output_tile_bytes = get_tile_size(dfb_out);
  const uint32_t input_block_tiles = in0_block_w * nt;
  const uint32_t output_subblock_tiles = sub_mt * sub_nt;
  const uint32_t num_blocks = kt / in0_block_w;

  constexpr auto b_args = TensorAccessorArgs<0>();
  constexpr auto output_args =
      TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
  const auto b_accessor = TensorAccessor(b_args, b_address, input_tile_bytes);
  const auto output_accessor =
      TensorAccessor(output_args, output_address, output_tile_bytes);

  for (uint32_t block_index = 0; block_index < num_blocks; ++block_index) {
    cb_reserve_back(dfb_in1, input_block_tiles);
    uint32_t write_ptr = get_write_ptr(dfb_in1);
    for (uint32_t block_row = 0; block_row < in0_block_w; ++block_row) {
      uint32_t tile_id = (block_index * in0_block_w + block_row) * nt;
      for (uint32_t tile_col = 0; tile_col < nt; ++tile_col) {
        noc_async_read_tile(tile_id + tile_col, b_accessor, write_ptr);
        write_ptr += input_tile_bytes;
      }
    }
    noc_async_read_barrier();
    cb_push_back(dfb_in1, input_block_tiles);
  }

  for (uint32_t output_row = 0; output_row < mt; output_row += sub_mt) {
    for (uint32_t output_col = 0; output_col < nt; output_col += sub_nt) {
      cb_wait_front(dfb_out, output_subblock_tiles);
      uint32_t read_ptr = get_read_ptr(dfb_out);
      for (uint32_t subblock_row = 0; subblock_row < sub_mt; ++subblock_row) {
        for (uint32_t subblock_col = 0; subblock_col < sub_nt; ++subblock_col) {
          uint32_t tile_id =
              (output_row + subblock_row) * nt + output_col + subblock_col;
          noc_async_write_tile(tile_id, output_accessor, read_ptr);
          read_ptr += output_tile_bytes;
        }
      }
      noc_async_write_barrier();
      cb_pop_front(dfb_out, output_subblock_tiles);
    }
  }
}
