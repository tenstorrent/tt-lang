// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Block-flatten softmax compute probe, extracted from the tt-lang-generated
// kernel for the fused SDPA chain (spda_fused.py, the softmax atom). Computes a
// single softmax over a flat array of `tiles` tiles (all tiles are one softmax
// group, as in the block-flatten reference): P[i] = exp(S[i] - mx) / sum, where
// mx = max over all tiles and sum = sum over all tiles of exp(S - mx).
//
// Structure mirrors the codegen (two-pass, numerically stable):
//   1. fill a scaler tile (1.0) for the reductions
//   2. reduce_max SCALAR over all `tiles`      -> mx (1 tile)   [whole-block, fixed]
//   3. exp pass:  e[i] = exp(S[i] - mx)         -> cb_exp        [subblocked by sub]
//   4. reduce_sum SCALAR over cb_exp            -> sum (1 tile)  [whole-block, fixed]
//   5. final pass: P[i] = exp(S[i]-mx) * recip(sum) -> cb_out    [subblocked by sub]
// The elementwise passes (3, 5) recompute exp(S-mx) exactly as the codegen does.
//
// DST layout (subblocked passes): each output tile uses TWO dst slots -- the
// value at even slot 2*i (S -> S-mx -> exp -> P) and the broadcast operand at odd
// slot 2*i+1 (bcast mx, then bcast sum -> recip). So sub is bounded by
// 2*sub <= DST capacity, like the fused chain. reduce_max/reduce_sum span the
// whole array (a block reduction) and are not subblocked.
//
// CBs: cb_s=0 (S input, tiles), cb_scaler=1 (1), cb_mx=2 (1), cb_exp=3 (tiles),
// cb_sum=4 (1), cb_out=16 (P output, tiles). The scaler/mx/exp/sum CBs are
// compute-internal (produced and consumed here); the sweep only feeds cb_s and
// drains cb_out.
//
// Compile-time args: 0 = tiles, 1 = sub (output tiles per acquire in passes 3/5).

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/exp.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/recip.h"
#include "api/compute/pack.h"
#include "api/compute/reduce.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

inline uint32_t float_to_bits(const float f) {
  uint32_t r;
  __builtin_memcpy(&r, &f, sizeof(r));
  return r;
}

constexpr uint32_t cb_s = 0;       // S input (tiles)
constexpr uint32_t cb_scaler = 1;  // reduce scaler (1 tile, filled here)
constexpr uint32_t cb_mx = 2;      // scalar max (1 tile)
constexpr uint32_t cb_exp = 3;     // exp(S - mx) scratch (tiles)
constexpr uint32_t cb_sum = 4;     // scalar sum (1 tile)
constexpr uint32_t cb_out = 16;    // P output (tiles)

void kernel_main() {
  const uint32_t tiles = get_compile_time_arg_val(0);
  const uint32_t sub = get_compile_time_arg_val(1);

  cb_wait_front(cb_s, tiles);       // resident S input (flat array)
  cb_reserve_back(cb_out, tiles);
  {
    DeviceZoneScopedN("softmax_loop");

    // --- 1. scaler = 1.0 (identity multiplier for the SCALAR reductions) ---
    init_sfpu(cb_scaler, cb_scaler);
    cb_reserve_back(cb_scaler, 1);
    tile_regs_acquire();
    fill_tile_init();
    fill_tile(0, float_to_bits(1.0f));
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, cb_scaler, 0);
    tile_regs_release();
    cb_push_back(cb_scaler, 1);
    cb_wait_front(cb_scaler, 1);

    // --- 2. mx = max over all `tiles` (SCALAR reduce, whole block) ---
    cb_reserve_back(cb_mx, 1);
    tile_regs_acquire();
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_SCALAR, false>(cb_s, cb_scaler,
                                                               cb_mx);
    for (uint32_t i = 0; i < tiles; ++i) {
      reduce_tile<PoolType::MAX, ReduceDim::REDUCE_SCALAR, false>(cb_s, cb_scaler,
                                                                 i, 0, 0);
    }
    reduce_uninit();
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, cb_mx, 0);
    tile_regs_release();
    cb_push_back(cb_mx, 1);
    cb_wait_front(cb_mx, 1);

    // --- 3. e[i] = exp(S[i] - mx), subblocked by sub (2 dst slots/tile) ---
    init_sfpu(cb_s, cb_exp);
    cb_reserve_back(cb_exp, tiles);
    for (uint32_t base = 0; base < tiles; base += sub) {
      uint32_t chunk = (tiles - base < sub) ? (tiles - base) : sub;
      tile_regs_acquire();
      unary_bcast_init<BroadcastType::SCALAR>(cb_mx, cb_exp);
      for (uint32_t i = 0; i < chunk; ++i) {
        unary_bcast<BroadcastType::SCALAR>(cb_mx, 0, 2 * i + 1);  // mx -> odd
      }
      copy_tile_init(cb_s);
      for (uint32_t i = 0; i < chunk; ++i) {
        copy_tile(cb_s, base + i, 2 * i);                        // S -> even
      }
      sub_binary_tile_init();
      for (uint32_t i = 0; i < chunk; ++i) {
        sub_binary_tile(2 * i, 2 * i + 1, 2 * i);                // S - mx
      }
      exp_tile_init();
      for (uint32_t i = 0; i < chunk; ++i) {
        exp_tile(2 * i);
      }
      tile_regs_commit();
      tile_regs_wait();
      for (uint32_t i = 0; i < chunk; ++i) {
        pack_tile<true>(2 * i, cb_exp, base + i);
      }
      tile_regs_release();
    }
    cb_push_back(cb_exp, tiles);
    cb_wait_front(cb_exp, tiles);

    // --- 4. sum = sum over cb_exp (SCALAR reduce, whole block) ---
    init_sfpu(cb_exp, cb_sum);
    cb_reserve_back(cb_sum, 1);
    tile_regs_acquire();
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR, false>(cb_exp, cb_scaler,
                                                               cb_sum);
    for (uint32_t i = 0; i < tiles; ++i) {
      reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR, false>(cb_exp,
                                                                 cb_scaler, i, 0,
                                                                 0);
    }
    reduce_uninit();
    tile_regs_commit();
    tile_regs_wait();
    pack_tile<true>(0, cb_sum, 0);
    tile_regs_release();
    cb_push_back(cb_sum, 1);
    cb_wait_front(cb_sum, 1);
    cb_pop_front(cb_exp, tiles);

    // --- 5. P[i] = exp(S[i]-mx) * recip(sum), subblocked by sub ---
    init_sfpu(cb_s, cb_out);
    for (uint32_t base = 0; base < tiles; base += sub) {
      uint32_t chunk = (tiles - base < sub) ? (tiles - base) : sub;
      tile_regs_acquire();
      unary_bcast_init<BroadcastType::SCALAR>(cb_mx, cb_out);
      for (uint32_t i = 0; i < chunk; ++i) {
        unary_bcast<BroadcastType::SCALAR>(cb_mx, 0, 2 * i + 1);  // mx -> odd
      }
      copy_tile_init(cb_s);
      for (uint32_t i = 0; i < chunk; ++i) {
        copy_tile(cb_s, base + i, 2 * i);                        // S -> even
      }
      sub_binary_tile_init();
      for (uint32_t i = 0; i < chunk; ++i) {
        sub_binary_tile(2 * i, 2 * i + 1, 2 * i);                // S - mx
      }
      unary_bcast_init<BroadcastType::SCALAR>(cb_sum, cb_out);
      for (uint32_t i = 0; i < chunk; ++i) {
        unary_bcast<BroadcastType::SCALAR>(cb_sum, 0, 2 * i + 1);  // sum -> odd
      }
      exp_tile_init();
      for (uint32_t i = 0; i < chunk; ++i) {
        exp_tile(2 * i);                                         // exp(S - mx)
      }
      recip_tile_init();
      for (uint32_t i = 0; i < chunk; ++i) {
        recip_tile(2 * i + 1);                                   // 1 / sum
      }
      mul_binary_tile_init();
      for (uint32_t i = 0; i < chunk; ++i) {
        mul_binary_tile(2 * i, 2 * i + 1, 2 * i);                // exp * (1/sum)
      }
      tile_regs_commit();
      tile_regs_wait();
      for (uint32_t i = 0; i < chunk; ++i) {
        pack_tile<true>(2 * i, cb_out, base + i);
      }
      tile_regs_release();
    }
  }
  cb_pop_front(cb_s, tiles);
  cb_push_back(cb_out, tiles);
  cb_pop_front(cb_mx, 1);
  cb_pop_front(cb_sum, 1);
  cb_pop_front(cb_scaler, 1);
}
