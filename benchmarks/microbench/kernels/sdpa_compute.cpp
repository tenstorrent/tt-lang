// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Hand-written single-pass (non-flash) SDPA, single head, single core, where
// EVERY compute region is subblocked by one configurable (sub_h, sub_w) -- the
// same way tt-lang subblocks each ttl.compute region for DST. Sweeping (sub_h,
// sub_w) drives the whole kernel's tiling, matching the compiler's single
// capacity-driven choice. NO max-subtraction (valid for bounded inputs).
//
//   Kt  = transpose(K)      (transpose_wh, subblock sub_h x sub_w over HD x Sk)
//   P   = exp(Q @ Kt)       (FUSED matmul + exp epilogue; S never materialized)
//   l   = rowsum(P)         (reduce; parallel dim Sq chunked by sub_h)
//   rl  = recip(l)          (elementwise, Sq chunked by sub_h)
//   rlbc= bcast_cols(rl)    (Sq chunked by sub_h)
//   O   = P @ V             (matmul, subblock sub_h x sub_w over Sq x HD)
//   out = O * rlbc          (elementwise, subblock sub_h x sub_w over Sq x HD)
//
// Requires sub_h | Sq, sub_h | HD, sub_w | Sk, sub_w | HD, sub_h*sub_w <= DST.
// K provided un-transposed ((Sk, HD)); the kernel transposes it.
//
// Compile-time args: 0 = Sq, 1 = Sk, 2 = HD, 3 = sub_h, 4 = sub_w.

#include <cstdint>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/transpose_wh.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t cb_q = 0;
constexpr uint32_t cb_k = 1;   // raw K (Sk, HD), transposed in-kernel
constexpr uint32_t cb_v = 2;
constexpr uint32_t cb_s = 3;
constexpr uint32_t cb_scaler = 4;
constexpr uint32_t cb_p = 5;
constexpr uint32_t cb_l = 6;
constexpr uint32_t cb_rl = 7;
constexpr uint32_t cb_rlbc = 8;
constexpr uint32_t cb_otmp = 9;
constexpr uint32_t cb_kt = 10;  // transpose(K) = (HD, Sk), produced in-kernel
constexpr uint32_t cb_out = 16;

void kernel_main() {
  const uint32_t Sq = get_compile_time_arg_val(0);
  const uint32_t Sk = get_compile_time_arg_val(1);
  const uint32_t HD = get_compile_time_arg_val(2);
  const uint32_t sh = get_compile_time_arg_val(3);
  const uint32_t sw = get_compile_time_arg_val(4);
  const uint32_t S_tiles = Sq * Sk;
  const uint32_t O_tiles = Sq * HD;

  cb_wait_front(cb_q, Sq * HD);
  cb_wait_front(cb_k, Sk * HD);
  cb_wait_front(cb_v, Sk * HD);

  {
    DeviceZoneScopedN("sdpa_loop");

    // reduce scaler = 1.0
    cb_reserve_back(cb_scaler, 1u);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, 1.0f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, cb_scaler, 0);
    tile_regs_release();
    cb_push_back(cb_scaler, 1u);
    cb_wait_front(cb_scaler, 1u);

    // 0. Kt = transpose(K): K tile (ks,kh) at ks*HD+kh -> Kt tile (kh,ks) at
    //    kh*Sk+ks. Subblocked (sub_h x sub_w over HD x Sk).
    cb_reserve_back(cb_kt, HD * Sk);
    transpose_wh_init(cb_k, cb_kt);
    for (uint32_t kh = 0; kh < HD; kh += sh) {
      for (uint32_t ks = 0; ks < Sk; ks += sw) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < sh; ++i)
          for (uint32_t j = 0; j < sw; ++j)
            transpose_wh_tile(cb_k, (ks + j) * HD + (kh + i), i * sw + j);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < sh; ++i)
          for (uint32_t j = 0; j < sw; ++j)
            pack_tile<true>(i * sw + j, cb_kt, (kh + i) * Sk + (ks + j));
        tile_regs_release();
      }
    }
    cb_push_back(cb_kt, HD * Sk);
    cb_pop_front(cb_k, Sk * HD);
    cb_wait_front(cb_kt, HD * Sk);

    // 1. P = exp(Q @ Kt): FUSED matmul + exp epilogue. The QKt subblock
    //    accumulates in DST, exp_tile runs in place on it, then pack P -- so S
    //    is never materialized to L1 (cb_s is unused now). Subblock (sub_h x
    //    sub_w over Sq x Sk), K-loop HD. Legal only because there is no
    //    max-subtraction; canonical exp(S - rowmax) would need the full row
    //    first (a reduction barrier).
    cb_reserve_back(cb_p, S_tiles);
    mm_block_init(cb_q, cb_kt, cb_p, false, sw, sh, HD);
    for (uint32_t om = 0; om < Sq; om += sh) {
      for (uint32_t on = 0; on < Sk; on += sw) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < HD; ++k)
          matmul_block(cb_q, cb_kt, om * HD + k, k * Sk + on, 0, false, sw, sh, HD);
        exp_tile_init();
        for (uint32_t t = 0; t < sh * sw; ++t) exp_tile(t);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < sh; ++i)
          for (uint32_t j = 0; j < sw; ++j)
            pack_tile<true>(i * sw + j, cb_p, (om + i) * Sk + (on + j));
        tile_regs_release();
      }
    }
    cb_push_back(cb_p, S_tiles);
    cb_pop_front(cb_kt, HD * Sk);

    // 3. l = rowsum(P), parallel dim Sq chunked by sub_h (full-Sk reduce/row).
    cb_wait_front(cb_p, S_tiles);
    cb_wait_front(cb_scaler, 1u);
    cb_reserve_back(cb_l, Sq);
    for (uint32_t om = 0; om < Sq; om += sh) {
      tile_regs_acquire();
      reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, false>(cb_p, cb_scaler, cb_l);
      for (uint32_t r = 0; r < sh; ++r)
        for (uint32_t k = 0; k < Sk; ++k)
          reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, false>(
              cb_p, cb_scaler, (om + r) * Sk + k, 0, r);
      reduce_uninit();
      tile_regs_commit();
      tile_regs_wait();
      for (uint32_t r = 0; r < sh; ++r) pack_tile<true>(r, cb_l, om + r);
      tile_regs_release();
    }
    cb_push_back(cb_l, Sq);

    // 4. rl = recip(l), Sq chunked by sub_h.
    cb_wait_front(cb_l, Sq);
    cb_reserve_back(cb_rl, Sq);
    init_sfpu(cb_l, cb_rl);
    copy_tile_init(cb_l);
    recip_tile_init();
    for (uint32_t om = 0; om < Sq; om += sh) {
      tile_regs_acquire();
      for (uint32_t r = 0; r < sh; ++r) copy_tile(cb_l, om + r, r);
      for (uint32_t r = 0; r < sh; ++r) recip_tile(r);
      tile_regs_commit();
      tile_regs_wait();
      for (uint32_t r = 0; r < sh; ++r) pack_tile<true>(r, cb_rl, om + r);
      tile_regs_release();
    }
    cb_push_back(cb_rl, Sq);
    cb_pop_front(cb_l, Sq);

    // 5. rlbc = bcast_cols(rl), Sq chunked by sub_h.
    cb_wait_front(cb_rl, Sq);
    cb_reserve_back(cb_rlbc, Sq);
    unary_bcast_init<BroadcastType::COL>(cb_rl, cb_rlbc);
    for (uint32_t om = 0; om < Sq; om += sh) {
      tile_regs_acquire();
      for (uint32_t r = 0; r < sh; ++r)
        unary_bcast<BroadcastType::COL>(cb_rl, om + r, r);
      tile_regs_commit();
      tile_regs_wait();
      for (uint32_t r = 0; r < sh; ++r) pack_tile<true>(r, cb_rlbc, om + r);
      tile_regs_release();
    }
    cb_push_back(cb_rlbc, Sq);
    cb_pop_front(cb_rl, Sq);

    // 6. O = P @ V, subblocked (sub_h x sub_w over Sq x HD), K-loop Sk.
    cb_reserve_back(cb_otmp, O_tiles);
    mm_block_init(cb_p, cb_v, cb_otmp, false, sw, sh, Sk);
    for (uint32_t om = 0; om < Sq; om += sh) {
      for (uint32_t on = 0; on < HD; on += sw) {
        tile_regs_acquire();
        for (uint32_t k = 0; k < Sk; ++k)
          matmul_block(cb_p, cb_v, om * Sk + k, k * HD + on, 0, false, sw, sh, Sk);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < sh; ++i)
          for (uint32_t j = 0; j < sw; ++j)
            pack_tile<true>(i * sw + j, cb_otmp, (om + i) * HD + (on + j));
        tile_regs_release();
      }
    }
    cb_push_back(cb_otmp, O_tiles);
    cb_pop_front(cb_p, S_tiles);

    // 7. out = O * rlbc, subblocked (sub_h x sub_w over Sq x HD).
    cb_wait_front(cb_otmp, O_tiles);
    cb_wait_front(cb_rlbc, Sq);
    cb_reserve_back(cb_out, O_tiles);
    binary_op_init_common(cb_otmp, cb_rlbc, cb_out);
    mul_tiles_init(cb_otmp, cb_rlbc);
    for (uint32_t om = 0; om < Sq; om += sh) {
      for (uint32_t on = 0; on < HD; on += sw) {
        tile_regs_acquire();
        for (uint32_t i = 0; i < sh; ++i)
          for (uint32_t j = 0; j < sw; ++j)
            mul_tiles(cb_otmp, cb_rlbc, (om + i) * HD + (on + j), om + i,
                      i * sw + j);
        tile_regs_commit();
        tile_regs_wait();
        for (uint32_t i = 0; i < sh; ++i)
          for (uint32_t j = 0; j < sw; ++j)
            pack_tile<true>(i * sw + j, cb_out, (om + i) * HD + (on + j));
        tile_regs_release();
      }
    }
    cb_push_back(cb_out, O_tiles);
    cb_pop_front(cb_otmp, O_tiles);
    cb_pop_front(cb_rlbc, Sq);
  }

  cb_pop_front(cb_q, Sq * HD);
  cb_pop_front(cb_v, Sk * HD);
}
