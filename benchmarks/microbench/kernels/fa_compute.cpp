// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Hand-written FLASH attention (streaming KV), single head, single core, with
// configurable subblock sizes for the two matmuls. The KV sequence is streamed
// in chunks of Skc key-tiles; only one chunk's scores (Sq x Skc) are ever
// materialized, and the running denominator `l` and output `O` accumulate
// across chunks via packer L1-accumulation (pack_reconfig_l1_acc), exactly like
// matmul_k_l1's cross-K accumulation -- here the "K steps" are the KV chunks.
//
// Softmax has NO max-subtraction (valid for the bounded inputs; the result is
// mathematically identical to the non-flash sdpa_compute.cpp). Per chunk c:
//   S   = Q @ Kt_chunk                    (qk_sub_h x qk_sub_w; output Sq x Skc)
//   P   = exp(S)
//   l  += rowsum(P)                        (L1-acc across chunks)
//   O  += P @ V_chunk                      (out_sub_h x out_sub_w; L1-acc)
// After all chunks: out = O * recip(l).
//
// Compile-time args: 0 = Sq, 1 = Sk, 2 = HD, 3 = Skc (KV chunk tiles),
// 4 = qk_sub_h, 5 = qk_sub_w, 6 = out_sub_h, 7 = out_sub_w.

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
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t cb_q = 0;
constexpr uint32_t cb_kt = 1;     // Kt chunk (HD, Skc), streamed
constexpr uint32_t cb_v = 2;      // V chunk (Skc, HD), streamed
constexpr uint32_t cb_scaler = 3;
constexpr uint32_t cb_s = 4;      // S chunk (Sq, Skc)
constexpr uint32_t cb_p = 5;      // P chunk (Sq, Skc)
constexpr uint32_t cb_l = 6;      // running rowsum (Sq), L1-accumulated
constexpr uint32_t cb_rl = 7;     // recip(l)
constexpr uint32_t cb_rlbc = 8;   // recip(l) bcast
constexpr uint32_t cb_o = 9;      // running output (Sq, HD), L1-accumulated
constexpr uint32_t cb_out = 16;

void kernel_main() {
  const uint32_t Sq = get_compile_time_arg_val(0);
  const uint32_t Sk = get_compile_time_arg_val(1);
  const uint32_t HD = get_compile_time_arg_val(2);
  const uint32_t Skc = get_compile_time_arg_val(3);
  const uint32_t qk_h = get_compile_time_arg_val(4);
  const uint32_t qk_w = get_compile_time_arg_val(5);
  const uint32_t o_h = get_compile_time_arg_val(6);
  const uint32_t o_w = get_compile_time_arg_val(7);
  const uint32_t n_chunks = Sk / Skc;
  const uint32_t S_tiles = Sq * Skc;
  const uint32_t O_tiles = Sq * HD;

  cb_wait_front(cb_q, Sq * HD);  // Q resident across all chunks

  {
    DeviceZoneScopedN("fa_loop");

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

    // Reserve the running accumulators once; they persist (not pushed) so the
    // packer keeps accumulating into the same L1 region across chunks.
    cb_reserve_back(cb_l, Sq);
    cb_reserve_back(cb_o, O_tiles);

    for (uint32_t c = 0; c < n_chunks; ++c) {
      cb_wait_front(cb_kt, HD * Skc);
      cb_wait_front(cb_v, Skc * HD);

      // ---- S = Q @ Kt_chunk (overwrite) ----
      cb_reserve_back(cb_s, S_tiles);
      mm_block_init(cb_q, cb_kt, cb_s, false, qk_w, qk_h, HD);
      pack_reconfig_l1_acc(0);  // set AFTER init (init resets the packer mode)
      for (uint32_t om = 0; om < Sq; om += qk_h) {
        for (uint32_t on = 0; on < Skc; on += qk_w) {
          tile_regs_acquire();
          for (uint32_t k = 0; k < HD; ++k) {
            matmul_block(cb_q, cb_kt, om * HD + k, k * Skc + on, 0, false, qk_w,
                         qk_h, HD);
          }
          tile_regs_commit();
          tile_regs_wait();
          for (uint32_t i = 0; i < qk_h; ++i)
            for (uint32_t j = 0; j < qk_w; ++j)
              pack_tile<true>(i * qk_w + j, cb_s, (om + i) * Skc + (on + j));
          tile_regs_release();
        }
      }
      cb_push_back(cb_s, S_tiles);

      // ---- P = exp(S) (overwrite) ----
      cb_wait_front(cb_s, S_tiles);
      cb_reserve_back(cb_p, S_tiles);
      init_sfpu(cb_s, cb_p);
      copy_tile_init(cb_s);
      exp_tile_init();
      pack_reconfig_l1_acc(0);
      for (uint32_t t = 0; t < S_tiles; ++t) {
        tile_regs_acquire();
        copy_tile(cb_s, t, 0);
        exp_tile(0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_p, t);
        tile_regs_release();
      }
      cb_push_back(cb_p, S_tiles);
      cb_pop_front(cb_s, S_tiles);

      // ---- l += rowsum(P) ; O += P @ V_chunk  (L1-accumulate across chunks) ----
      cb_wait_front(cb_p, S_tiles);
      for (uint32_t om = 0; om < Sq; ++om) {
        tile_regs_acquire();
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW, false>(cb_p, cb_scaler,
                                                                cb_l);
        for (uint32_t k = 0; k < Skc; ++k)
          reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW, false>(
              cb_p, cb_scaler, om * Skc + k, 0, 0);
        reduce_uninit();
        tile_regs_commit();
        tile_regs_wait();
        pack_reconfig_l1_acc(c == 0 ? 0 : 1);  // chunk 0 overwrites, rest accumulate
        pack_tile<true>(0, cb_l, om);
        tile_regs_release();
      }

      mm_block_init(cb_p, cb_v, cb_o, false, o_w, o_h, Skc);
      pack_reconfig_l1_acc(c == 0 ? 0 : 1);  // set AFTER mm_block_init
      for (uint32_t om = 0; om < Sq; om += o_h) {
        for (uint32_t on = 0; on < HD; on += o_w) {
          tile_regs_acquire();
          for (uint32_t k = 0; k < Skc; ++k) {
            matmul_block(cb_p, cb_v, om * Skc + k, k * HD + on, 0, false, o_w,
                         o_h, Skc);
          }
          tile_regs_commit();
          tile_regs_wait();
          for (uint32_t i = 0; i < o_h; ++i)
            for (uint32_t j = 0; j < o_w; ++j)
              pack_tile<true>(i * o_w + j, cb_o, (om + i) * HD + (on + j));
          tile_regs_release();
        }
      }

      cb_pop_front(cb_p, S_tiles);
      cb_pop_front(cb_kt, HD * Skc);
      cb_pop_front(cb_v, Skc * HD);
    }

    pack_reconfig_l1_acc(0);
    cb_push_back(cb_l, Sq);
    cb_push_back(cb_o, O_tiles);

    // ---- normalize: out = O * recip(l) ----
    cb_wait_front(cb_l, Sq);
    cb_reserve_back(cb_rl, Sq);
    init_sfpu(cb_l, cb_rl);
    copy_tile_init(cb_l);
    recip_tile_init();
    for (uint32_t om = 0; om < Sq; ++om) {
      tile_regs_acquire();
      copy_tile(cb_l, om, 0);
      recip_tile(0);
      tile_regs_commit();
      tile_regs_wait();
      pack_tile<true>(0, cb_rl, om);
      tile_regs_release();
    }
    cb_push_back(cb_rl, Sq);
    cb_pop_front(cb_l, Sq);

    cb_wait_front(cb_rl, Sq);
    cb_reserve_back(cb_rlbc, Sq);
    unary_bcast_init<BroadcastType::COL>(cb_rl, cb_rlbc);
    for (uint32_t om = 0; om < Sq; ++om) {
      tile_regs_acquire();
      unary_bcast<BroadcastType::COL>(cb_rl, om, 0);
      tile_regs_commit();
      tile_regs_wait();
      pack_tile<true>(0, cb_rlbc, om);
      tile_regs_release();
    }
    cb_push_back(cb_rlbc, Sq);
    cb_pop_front(cb_rl, Sq);

    cb_wait_front(cb_o, O_tiles);
    cb_wait_front(cb_rlbc, Sq);
    cb_reserve_back(cb_out, O_tiles);
    binary_op_init_common(cb_o, cb_rlbc, cb_out);
    mul_tiles_init(cb_o, cb_rlbc);
    for (uint32_t om = 0; om < Sq; ++om) {
      for (uint32_t hd = 0; hd < HD; ++hd) {
        tile_regs_acquire();
        mul_tiles(cb_o, cb_rlbc, om * HD + hd, om, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_out, om * HD + hd);
        tile_regs_release();
      }
    }
    cb_push_back(cb_out, O_tiles);
    cb_pop_front(cb_o, O_tiles);
    cb_pop_front(cb_rlbc, Sq);
  }

  cb_pop_front(cb_q, Sq * HD);
}
