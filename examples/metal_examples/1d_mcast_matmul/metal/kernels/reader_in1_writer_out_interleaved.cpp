// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"

void kernel_main() {
  // READER
  uint32_t rt_args_idx = 0;
  // in1 tensor args
  const uint32_t in1_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
  uint32_t in1_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);

  // WRITER
  // out tensor args
  const uint32_t out_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
  uint32_t out_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);

  // COMPILE TIME ARGS
  // READER
  // in1 tensor args
  constexpr uint32_t in1_tensor_stride_w = get_compile_time_arg_val(0);
  constexpr uint32_t in1_tensor_stride_h = get_compile_time_arg_val(1);
  constexpr uint32_t in1_tensor_next_block_stride = get_compile_time_arg_val(2);
  constexpr uint32_t in1_tensor_next_w_dim_block_stride =
      get_compile_time_arg_val(3);
  // in1 block args
  constexpr uint32_t in1_block_w = get_compile_time_arg_val(4);
  constexpr uint32_t in1_block_h = get_compile_time_arg_val(5);
  constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(6);
  // in0/in1 common args
  constexpr uint32_t num_blocks_inner_dim = get_compile_time_arg_val(7);
  constexpr uint32_t num_blocks_w_dim = get_compile_time_arg_val(8);
  constexpr uint32_t num_blocks_h_dim = get_compile_time_arg_val(9);

  // WRITER
  // out tensor args
  constexpr uint32_t out_tensor_stride_w = get_compile_time_arg_val(10);
  constexpr uint32_t out_tensor_stride_h = get_compile_time_arg_val(11);
  constexpr uint32_t out_tensor_next_subblock_stride_w =
      get_compile_time_arg_val(12);
  constexpr uint32_t out_tensor_next_subblock_stride_h =
      get_compile_time_arg_val(13);
  constexpr uint32_t out_tensor_next_w_dim_block_stride =
      get_compile_time_arg_val(14);
  constexpr uint32_t out_tensor_next_h_dim_block_stride =
      get_compile_time_arg_val(15);
  // out subblock args
  constexpr uint32_t out_subblock_w = get_compile_time_arg_val(16);
  constexpr uint32_t out_subblock_h = get_compile_time_arg_val(17);
  constexpr uint32_t out_num_subblocks_w = get_compile_time_arg_val(18);
  constexpr uint32_t out_num_subblocks_h = get_compile_time_arg_val(19);
  constexpr uint32_t out_subblock_tile_count = get_compile_time_arg_val(20);
  constexpr auto in1_args = TensorAccessorArgs<21>();
  constexpr auto out_args =
      TensorAccessorArgs<in1_args.next_compile_time_args_offset()>();

  constexpr uint32_t one_tile = 1;

  constexpr uint32_t cb_id_in1 = 1;
  constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
  const auto s1 =
      TensorAccessor(in1_args, in1_tensor_addr, in1_single_tile_size_bytes);

  //  WRITER
  constexpr uint32_t cb_id_out0 = tt::CBIndex::c_16;
  constexpr uint32_t output_single_tile_size_bytes = get_tile_size(cb_id_out0);
  const auto s =
      TensorAccessor(out_args, out_tensor_addr, output_single_tile_size_bytes);

  uint32_t in1_tensor_current_h_dim_block_tile_id = in1_tensor_start_tile_id;
  uint32_t out_tensor_current_h_dim_block_tile_id = out_tensor_start_tile_id;
  for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
    uint32_t in1_tensor_current_w_dim_block_tile_id =
        in1_tensor_current_h_dim_block_tile_id;
    uint32_t out_tensor_current_w_dim_block_tile_id =
        out_tensor_current_h_dim_block_tile_id;

    for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {
      uint32_t in1_tensor_current_inner_dim_block_start_tile_id =
          in1_tensor_current_w_dim_block_tile_id;
      for (uint32_t block = 0; block < num_blocks_inner_dim; ++block) {
        // Operand 1
        cb_reserve_back(cb_id_in1, in1_block_num_tiles);

        uint32_t l1_write_addr_in1 = get_write_ptr(cb_id_in1);

        // Copy in1 block into CB, as the default kernel
        uint32_t in1_tensor_row_start_tile_id =
            in1_tensor_current_inner_dim_block_start_tile_id;
        for (uint32_t h = 0; h < in1_block_h; ++h) {
          uint32_t in1_tensor_tile_id = in1_tensor_row_start_tile_id;
          for (uint32_t w = 0; w < in1_block_w; ++w) {
            noc_async_read_tile(in1_tensor_tile_id, s1, l1_write_addr_in1);
            l1_write_addr_in1 += in1_single_tile_size_bytes;
            in1_tensor_tile_id += in1_tensor_stride_w;
          }
          in1_tensor_row_start_tile_id += in1_tensor_stride_h;
        }
        in1_tensor_current_inner_dim_block_start_tile_id +=
            in1_tensor_next_block_stride;
        noc_async_read_barrier();
        cb_push_back(cb_id_in1, in1_block_num_tiles);
      }
      // WRITER

      uint32_t out_tensor_sbh_start_tile_id =
          out_tensor_current_w_dim_block_tile_id;
      for (uint32_t sbh = 0; sbh < out_num_subblocks_h; ++sbh) {
        uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
        for (uint32_t sbw = 0; sbw < out_num_subblocks_w; ++sbw) {
          uint32_t out_tensor_sb_row_start_tile_id =
              out_tensor_sbw_start_tile_id;
          cb_wait_front(cb_id_out0, out_subblock_tile_count);
          uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
          for (uint32_t h = 0; h < out_subblock_h; ++h) {
            uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
            for (uint32_t w = 0; w < out_subblock_w; ++w) {
              noc_async_write_tile(out_tensor_tile_id, s, l1_read_addr);
              l1_read_addr += output_single_tile_size_bytes;
              out_tensor_tile_id += out_tensor_stride_w;
            }
            out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
          }
          noc_async_write_barrier();
          cb_pop_front(cb_id_out0, out_subblock_tile_count);
          out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
        }
        out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
      }
      in1_tensor_current_w_dim_block_tile_id +=
          in1_tensor_next_w_dim_block_stride;
      out_tensor_current_w_dim_block_tile_id +=
          out_tensor_next_w_dim_block_stride;
    }
    out_tensor_current_h_dim_block_tile_id +=
        out_tensor_next_h_dim_block_stride;
  }
  noc_async_write_barrier();
}
