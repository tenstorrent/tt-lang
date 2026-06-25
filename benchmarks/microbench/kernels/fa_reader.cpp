// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Flash-attention reader: Q resident (read once); Kt and V streamed in KV
// chunks of Skc tiles. Kt is (HD*32, Sk*32) -- tile (k,n) at k*Sk+n; the chunk
// c is columns [c*Skc, c*Skc+Skc). V is (Sk*32, HD*32) -- tile (k,n) at k*HD+n;
// chunk c is rows [c*Skc, c*Skc+Skc).
//
// Runtime args: 0 = Q addr, 1 = Kt addr, 2 = V addr, 3 = Sq, 4 = Sk, 5 = HD,
// 6 = Skc.

#include "api/dataflow/dataflow_api.h"
#include <stdint.h>

void kernel_main() {
  uint32_t q_addr = get_arg_val<uint32_t>(0);
  uint32_t kt_addr = get_arg_val<uint32_t>(1);
  uint32_t v_addr = get_arg_val<uint32_t>(2);
  uint32_t Sq = get_arg_val<uint32_t>(3);
  uint32_t Sk = get_arg_val<uint32_t>(4);
  uint32_t HD = get_arg_val<uint32_t>(5);
  uint32_t Skc = get_arg_val<uint32_t>(6);
  uint32_t n_chunks = Sk / Skc;

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

  // Q resident.
  cb_reserve_back(cb_q, Sq * HD);
  uint32_t wr = get_write_ptr(cb_q);
  for (uint32_t t = 0; t < Sq * HD; ++t) {
    noc_async_read_tile(t, sq, wr);
    wr += tile_bytes;
  }
  noc_async_read_barrier();
  cb_push_back(cb_q, Sq * HD);

  for (uint32_t c = 0; c < n_chunks; ++c) {
    // Kt chunk: HD rows x Skc cols at column offset c*Skc.
    cb_reserve_back(cb_kt, HD * Skc);
    wr = get_write_ptr(cb_kt);
    for (uint32_t k = 0; k < HD; ++k) {
      for (uint32_t j = 0; j < Skc; ++j) {
        noc_async_read_tile(k * Sk + c * Skc + j, skt, wr);
        wr += tile_bytes;
      }
    }
    noc_async_read_barrier();
    cb_push_back(cb_kt, HD * Skc);

    // V chunk: Skc rows at row offset c*Skc x HD cols.
    cb_reserve_back(cb_v, Skc * HD);
    wr = get_write_ptr(cb_v);
    for (uint32_t i = 0; i < Skc; ++i) {
      for (uint32_t n = 0; n < HD; ++n) {
        noc_async_read_tile((c * Skc + i) * HD + n, sv, wr);
        wr += tile_bytes;
      }
    }
    noc_async_read_barrier();
    cb_push_back(cb_v, Skc * HD);
  }
}
