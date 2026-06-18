// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Reader for MB3. A is (mt*32, kt*32) -- tile (m,k) at index m*kt+k; B is
// (kt*32, nt*32) -- tile (k,n) at index k*nt+n. Per K step, pushes A's column k
// (mt tiles) to dfb_in0 and B's row k (nt tiles) to dfb_in1, so the compute
// kernel matmul_blocks the mt*nt subblock each iteration.
//
// Runtime args: 0 = A address, 1 = B address, 2 = mt, 3 = nt, 4 = kt.

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  uint32_t a_addr = get_arg_val<uint32_t>(0);
  uint32_t b_addr = get_arg_val<uint32_t>(1);
  uint32_t mt = get_arg_val<uint32_t>(2);
  uint32_t nt = get_arg_val<uint32_t>(3);
  uint32_t kt = get_arg_val<uint32_t>(4);

  constexpr uint32_t dfb_in0 = tt::CBIndex::c_0;
  constexpr uint32_t dfb_in1 = tt::CBIndex::c_1;
  const uint32_t tile_bytes = get_tile_size(dfb_in0);

  constexpr auto a_args = TensorAccessorArgs<0>();
  constexpr auto b_args =
      TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
  const auto sa = TensorAccessor(a_args, a_addr, tile_bytes);
  const auto sb = TensorAccessor(b_args, b_addr, tile_bytes);

  for (uint32_t k = 0; k < kt; ++k) {
    cb_reserve_back(dfb_in0, mt);
    uint32_t a_wr = get_write_ptr(dfb_in0);
    for (uint32_t m = 0; m < mt; ++m) {
      noc_async_read_tile(m * kt + k, sa, a_wr);
      a_wr += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(dfb_in0, mt);

    cb_reserve_back(dfb_in1, nt);
    uint32_t b_wr = get_write_ptr(dfb_in1);
    for (uint32_t n = 0; n < nt; ++n) {
      noc_async_read_tile(k * nt + n, sb, b_wr);
      b_wr += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(dfb_in1, nt);
  }
}
