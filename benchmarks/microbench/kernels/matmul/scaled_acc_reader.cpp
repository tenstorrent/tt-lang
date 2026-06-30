// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the scaled-accumulate matmul: read a (Mt*Kt tiles), b (Kt*Nt
// tiles), scale (Mt*Nt tiles), acc (Mt*Nt tiles) into cb 0/1/2/3, each
// contiguous row-major. Operands resident for the single-pass kernel.
//
// Runtime args: 0 = a addr, 1 = b addr, 2 = scale addr, 3 = acc addr,
// 4 = Mt, 5 = Nt, 6 = Kt.

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  uint32_t a_addr = get_arg_val<uint32_t>(0);
  uint32_t b_addr = get_arg_val<uint32_t>(1);
  uint32_t scale_addr = get_arg_val<uint32_t>(2);
  uint32_t acc_addr = get_arg_val<uint32_t>(3);
  uint32_t Mt = get_arg_val<uint32_t>(4);
  uint32_t Nt = get_arg_val<uint32_t>(5);
  uint32_t Kt = get_arg_val<uint32_t>(6);
  uint32_t a_tiles = Mt * Kt;
  uint32_t b_tiles = Kt * Nt;
  uint32_t out_tiles = Mt * Nt;

  constexpr uint32_t cb_a = tt::CBIndex::c_0;
  constexpr uint32_t cb_b = tt::CBIndex::c_1;
  constexpr uint32_t cb_scale = tt::CBIndex::c_2;
  constexpr uint32_t cb_acc = tt::CBIndex::c_3;
  const uint32_t tile_bytes = get_tile_size(cb_a);

  constexpr auto a_args = TensorAccessorArgs<0>();
  constexpr auto b_args =
      TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
  constexpr auto scale_args =
      TensorAccessorArgs<b_args.next_compile_time_args_offset()>();
  constexpr auto acc_args =
      TensorAccessorArgs<scale_args.next_compile_time_args_offset()>();
  const auto sa = TensorAccessor(a_args, a_addr, tile_bytes);
  const auto sb = TensorAccessor(b_args, b_addr, tile_bytes);
  const auto ss = TensorAccessor(scale_args, scale_addr, tile_bytes);
  const auto sc = TensorAccessor(acc_args, acc_addr, tile_bytes);

  cb_reserve_back(cb_a, a_tiles);
  uint32_t wr = get_write_ptr(cb_a);
  for (uint32_t t = 0; t < a_tiles; ++t) {
    noc_async_read_tile(t, sa, wr);
    wr += tile_bytes;
  }
  noc_async_read_barrier();
  cb_push_back(cb_a, a_tiles);

  cb_reserve_back(cb_b, b_tiles);
  wr = get_write_ptr(cb_b);
  for (uint32_t t = 0; t < b_tiles; ++t) {
    noc_async_read_tile(t, sb, wr);
    wr += tile_bytes;
  }
  noc_async_read_barrier();
  cb_push_back(cb_b, b_tiles);

  cb_reserve_back(cb_scale, out_tiles);
  wr = get_write_ptr(cb_scale);
  for (uint32_t t = 0; t < out_tiles; ++t) {
    noc_async_read_tile(t, ss, wr);
    wr += tile_bytes;
  }
  noc_async_read_barrier();
  cb_push_back(cb_scale, out_tiles);

  cb_reserve_back(cb_acc, out_tiles);
  wr = get_write_ptr(cb_acc);
  for (uint32_t t = 0; t < out_tiles; ++t) {
    noc_async_read_tile(t, sc, wr);
    wr += tile_bytes;
  }
  noc_async_read_barrier();
  cb_push_back(cb_acc, out_tiles);
}
