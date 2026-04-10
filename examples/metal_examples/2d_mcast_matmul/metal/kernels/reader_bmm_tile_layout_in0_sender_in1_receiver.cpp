// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Left column cores (excluding corner): reads in0 from DRAM and multicasts
// rightward along the row. Receives in1 via multicast from the top row.
// Adapted from tt-metal programming example (batch loop removed).

#include "api/dataflow/dataflow_api.h"
#include "hostdevcommon/common_values.hpp"
#include <stdint.h>

void kernel_main() {
  // in0 tensor args
  uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
  uint32_t in0_tensor_start_tile_id = get_arg_val<uint32_t>(1);
  uint32_t in0_tensor_stride_w = get_arg_val<uint32_t>(2);
  uint32_t in0_tensor_stride_h = get_arg_val<uint32_t>(3);
  uint32_t in0_tensor_next_block_stride = get_arg_val<uint32_t>(4);

  // in0 block args
  uint32_t in0_block_w = get_arg_val<uint32_t>(5);
  uint32_t in0_block_h = get_arg_val<uint32_t>(6);
  uint32_t in0_block_num_tiles = get_arg_val<uint32_t>(7);

  // in1 tensor args (unused on this core but kept for uniform arg layout)
  uint32_t in1_tensor_addr = get_arg_val<uint32_t>(8);
  uint32_t in1_tensor_start_tile_id = get_arg_val<uint32_t>(9);
  uint32_t in1_tensor_stride_w = get_arg_val<uint32_t>(10);
  uint32_t in1_tensor_stride_h = get_arg_val<uint32_t>(11);
  uint32_t in1_tensor_next_block_stride = get_arg_val<uint32_t>(12);

  // in1 block args
  uint32_t in1_block_w = get_arg_val<uint32_t>(13);
  uint32_t in1_block_h = get_arg_val<uint32_t>(14);
  uint32_t in1_block_num_tiles = get_arg_val<uint32_t>(15);

  // in0/in1 common args
  uint32_t num_blocks = get_arg_val<uint32_t>(16);

  // in0 mcast args
  uint32_t in0_mcast_dest_noc_start_x = get_arg_val<uint32_t>(17);
  uint32_t in0_mcast_dest_noc_start_y = get_arg_val<uint32_t>(18);
  uint32_t in0_mcast_dest_noc_end_x = get_arg_val<uint32_t>(19);
  uint32_t in0_mcast_dest_noc_end_y = get_arg_val<uint32_t>(20);
  uint32_t in0_mcast_num_dests = get_arg_val<uint32_t>(21);
  uint32_t in0_mcast_sender_noc_x = get_arg_val<uint32_t>(22);
  uint32_t in0_mcast_sender_noc_y = get_arg_val<uint32_t>(23);
  uint32_t in0_mcast_sender_semaphore_addr =
      get_semaphore(get_arg_val<uint32_t>(24));
  uint32_t in0_mcast_receiver_semaphore_addr =
      get_semaphore(get_arg_val<uint32_t>(25));

  // in1 mcast args
  uint32_t in1_mcast_dest_noc_start_x = get_arg_val<uint32_t>(26);
  uint32_t in1_mcast_dest_noc_start_y = get_arg_val<uint32_t>(27);
  uint32_t in1_mcast_dest_noc_end_x = get_arg_val<uint32_t>(28);
  uint32_t in1_mcast_dest_noc_end_y = get_arg_val<uint32_t>(29);
  uint32_t in1_mcast_num_dests = get_arg_val<uint32_t>(30);
  uint32_t in1_mcast_sender_noc_x = get_arg_val<uint32_t>(31);
  uint32_t in1_mcast_sender_noc_y = get_arg_val<uint32_t>(32);
  uint32_t in1_mcast_sender_semaphore_addr =
      get_semaphore(get_arg_val<uint32_t>(33));
  uint32_t in1_mcast_receiver_semaphore_addr =
      get_semaphore(get_arg_val<uint32_t>(34));

  constexpr uint32_t cb_id_in0 = 0;
  constexpr uint32_t cb_id_in1 = 1;

  const uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);

  uint32_t l1_write_addr_in0;

  volatile tt_l1_ptr uint32_t *in0_mcast_receiver_semaphore_addr_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
          in0_mcast_receiver_semaphore_addr);
  *(in0_mcast_receiver_semaphore_addr_ptr) = VALID;

  volatile tt_l1_ptr uint32_t *in0_mcast_sender_semaphore_addr_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
          in0_mcast_sender_semaphore_addr);

  volatile tt_l1_ptr uint32_t *in1_mcast_receiver_semaphore_addr_ptr =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
          in1_mcast_receiver_semaphore_addr);

  constexpr auto s0_args = TensorAccessorArgs<0>();
  const auto s0 =
      TensorAccessor(s0_args, in0_tensor_addr, single_tile_size_bytes);

  uint32_t in0_tensor_current_block_start_tile_id = in0_tensor_start_tile_id;
  for (uint32_t block = 0; block < num_blocks; block++) {
    // -- in0: read from DRAM and multicast rightward --
    cb_reserve_back(cb_id_in0, in0_block_num_tiles);
    l1_write_addr_in0 = get_write_ptr(cb_id_in0);

    uint32_t in0_start_address = l1_write_addr_in0;
    uint32_t in0_block_size_bytes = 0;

    uint32_t in0_tensor_row_start_tile_id =
        in0_tensor_current_block_start_tile_id;
    for (uint32_t h = 0; h < in0_block_h; h++) {
      uint32_t in0_tensor_tile_id = in0_tensor_row_start_tile_id;
      for (uint32_t w = 0; w < in0_block_w; w++) {
        noc_async_read_tile(in0_tensor_tile_id, s0, l1_write_addr_in0);
        l1_write_addr_in0 += single_tile_size_bytes;
        in0_tensor_tile_id += in0_tensor_stride_w;
        in0_block_size_bytes += single_tile_size_bytes;
      }
      in0_tensor_row_start_tile_id += in0_tensor_stride_h;
    }
    in0_tensor_current_block_start_tile_id += in0_tensor_next_block_stride;

    noc_async_read_barrier();

    noc_semaphore_wait(in0_mcast_sender_semaphore_addr_ptr,
                       in0_mcast_num_dests);
    noc_semaphore_set(in0_mcast_sender_semaphore_addr_ptr, 0);

    uint64_t in0_multicast_data_addr = get_noc_multicast_addr(
        in0_mcast_dest_noc_end_x, in0_mcast_dest_noc_end_y,
        in0_mcast_dest_noc_start_x, in0_mcast_dest_noc_start_y,
        in0_start_address);
    noc_async_write_multicast(in0_start_address, in0_multicast_data_addr,
                              in0_block_size_bytes, in0_mcast_num_dests);

#ifdef ARCH_BLACKHOLE
    noc_async_writes_flushed();
#endif

    uint64_t in0_mcast_receiver_semaphore_noc_addr = get_noc_multicast_addr(
        in0_mcast_dest_noc_end_x, in0_mcast_dest_noc_end_y,
        in0_mcast_dest_noc_start_x, in0_mcast_dest_noc_start_y,
        in0_mcast_receiver_semaphore_addr);
    noc_semaphore_set_multicast(in0_mcast_receiver_semaphore_addr,
                                in0_mcast_receiver_semaphore_noc_addr,
                                in0_mcast_num_dests);

    cb_push_back(cb_id_in0, in0_block_num_tiles);

    // -- in1: receive via multicast from top row --
    cb_reserve_back(cb_id_in1, in1_block_num_tiles);

    noc_semaphore_set(in1_mcast_receiver_semaphore_addr_ptr, INVALID);

    uint64_t in1_mcast_sender_semaphore_noc_addr =
        get_noc_addr(in1_mcast_sender_noc_x, in1_mcast_sender_noc_y,
                     in1_mcast_sender_semaphore_addr);
    noc_semaphore_inc(in1_mcast_sender_semaphore_noc_addr, 1);

    noc_semaphore_wait(in1_mcast_receiver_semaphore_addr_ptr, VALID);

    cb_push_back(cb_id_in1, in1_block_num_tiles);
  }
}
