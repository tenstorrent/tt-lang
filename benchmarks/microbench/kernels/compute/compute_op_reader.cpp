// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Reader for the MB4 compute-op probe. Reads x (n0 tiles) into cb0 and an
// optional second operand y (n1 tiles) into cb1, once, before the measured zone.
// Unary ops pass n1 = 0 (no second operand); binary ops pass n1 = tiles; bcast
// and reduce pass n1 = 1 (a broadcast operand / a reduce scaler). Idle in zone.
//
// Runtime args: 0 = x addr, 1 = y addr, 2 = n0 (tiles), 3 = n1 (second operand).

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  uint32_t x_addr = get_arg_val<uint32_t>(0);
  uint32_t y_addr = get_arg_val<uint32_t>(1);
  uint32_t n0 = get_arg_val<uint32_t>(2);
  uint32_t n1 = get_arg_val<uint32_t>(3);

  constexpr uint32_t cb0 = tt::CBIndex::c_0;
  constexpr uint32_t cb1 = tt::CBIndex::c_1;
  const uint32_t tile_bytes = get_tile_size(cb0);

  constexpr auto x_args = TensorAccessorArgs<0>();
  constexpr auto y_args =
      TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
  const auto sx = TensorAccessor(x_args, x_addr, tile_bytes);
  const auto sy = TensorAccessor(y_args, y_addr, tile_bytes);

  cb_reserve_back(cb0, n0);
  uint32_t wr = get_write_ptr(cb0);
  for (uint32_t t = 0; t < n0; ++t) {
    noc_async_read_tile(t, sx, wr);
    wr += tile_bytes;
  }
  noc_async_read_barrier();
  cb_push_back(cb0, n0);

  if (n1 > 0) {
    cb_reserve_back(cb1, n1);
    wr = get_write_ptr(cb1);
    for (uint32_t t = 0; t < n1; ++t) {
      noc_async_read_tile(t, sy, wr);
      wr += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb1, n1);
  }
}
