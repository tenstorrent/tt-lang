// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// SDPA reader: read all of Q, Kt, V into cb 0/1/2 once (operands resident for
// the two resident-operand matmuls). Q is (Sq*32, HD*32), Kt is (HD*32, Sk*32),
// V is (Sk*32, HD*32), each read as contiguous row-major tiles.
//
// Runtime args: 0 = Q addr, 1 = Kt addr, 2 = V addr, 3 = Q tiles, 4 = Kt tiles,
// 5 = V tiles.

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  uint32_t q_addr = get_arg_val<uint32_t>(0);
  uint32_t kt_addr = get_arg_val<uint32_t>(1);
  uint32_t v_addr = get_arg_val<uint32_t>(2);
  uint32_t q_tiles = get_arg_val<uint32_t>(3);
  uint32_t kt_tiles = get_arg_val<uint32_t>(4);
  uint32_t v_tiles = get_arg_val<uint32_t>(5);

  constexpr uint32_t cb_q = tt::CBIndex::c_0;
  constexpr uint32_t cb_kt = tt::CBIndex::c_1;
  constexpr uint32_t cb_v = tt::CBIndex::c_2;
  const uint32_t tile_bytes = get_tile_size(cb_q);

  constexpr auto q_args = TensorAccessorArgs<0>();
  constexpr auto kt_args =
      TensorAccessorArgs<q_args.next_compile_time_args_offset()>();
  constexpr auto v_args =
      TensorAccessorArgs<kt_args.next_compile_time_args_offset()>();
  const auto sq = TensorAccessor(q_args, q_addr, tile_bytes);
  const auto skt = TensorAccessor(kt_args, kt_addr, tile_bytes);
  const auto sv = TensorAccessor(v_args, v_addr, tile_bytes);

  cb_reserve_back(cb_q, q_tiles);
  uint32_t wr = get_write_ptr(cb_q);
  for (uint32_t t = 0; t < q_tiles; ++t) {
    noc_async_read_tile(t, sq, wr);
    wr += tile_bytes;
  }
  noc_async_read_barrier();
  cb_push_back(cb_q, q_tiles);

  cb_reserve_back(cb_kt, kt_tiles);
  wr = get_write_ptr(cb_kt);
  for (uint32_t t = 0; t < kt_tiles; ++t) {
    noc_async_read_tile(t, skt, wr);
    wr += tile_bytes;
  }
  noc_async_read_barrier();
  cb_push_back(cb_kt, kt_tiles);

  cb_reserve_back(cb_v, v_tiles);
  wr = get_write_ptr(cb_v);
  for (uint32_t t = 0; t < v_tiles; ++t) {
    noc_async_read_tile(t, sv, wr);
    wr += tile_bytes;
  }
  noc_async_read_barrier();
  cb_push_back(cb_v, v_tiles);
}
