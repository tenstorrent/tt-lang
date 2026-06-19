// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the diagnostic compute-feed probe. A is written row-major by tile
// (m * kt + k). B is written row-major by tile (k * nt + n). The compute kernel
// consumes the full per-node operand block after both DFBs are resident.
//
// Runtime args: 0 = A address, 1 = B address, 2 = mt, 3 = nt, 4 = kt.

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  uint32_t a_address = get_arg_val<uint32_t>(0);
  uint32_t b_address = get_arg_val<uint32_t>(1);
  uint32_t mt = get_arg_val<uint32_t>(2);
  uint32_t nt = get_arg_val<uint32_t>(3);
  uint32_t kt = get_arg_val<uint32_t>(4);

  constexpr uint32_t dfb_in0 = tt::CBIndex::c_0;
  constexpr uint32_t dfb_in1 = tt::CBIndex::c_1;
  const uint32_t tile_bytes = get_tile_size(dfb_in0);

  constexpr auto a_args = TensorAccessorArgs<0>();
  constexpr auto b_args =
      TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
  const auto a_accessor = TensorAccessor(a_args, a_address, tile_bytes);
  const auto b_accessor = TensorAccessor(b_args, b_address, tile_bytes);

  cb_reserve_back(dfb_in0, mt * kt);
  uint32_t a_write_ptr = get_write_ptr(dfb_in0);
  for (uint32_t tile_row = 0; tile_row < mt; ++tile_row) {
    for (uint32_t k_index = 0; k_index < kt; ++k_index) {
      noc_async_read_tile(tile_row * kt + k_index, a_accessor, a_write_ptr);
      a_write_ptr += tile_bytes;
    }
  }
  noc_async_read_barrier();
  cb_push_back(dfb_in0, mt * kt);

  cb_reserve_back(dfb_in1, kt * nt);
  uint32_t b_write_ptr = get_write_ptr(dfb_in1);
  for (uint32_t k_index = 0; k_index < kt; ++k_index) {
    for (uint32_t tile_col = 0; tile_col < nt; ++tile_col) {
      noc_async_read_tile(k_index * nt + tile_col, b_accessor, b_write_ptr);
      b_write_ptr += tile_bytes;
    }
  }
  noc_async_read_barrier();
  cb_push_back(dfb_in1, kt * nt);
}
