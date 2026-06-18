// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MB3, L1-K matmul accumulation: C[mt,nt] = sum_k A[k] @ B[k] over `kt`
// K-tiles, with the mt*nt accumulator living in L1. Each K step matmuls every
// sub_mt*sub_nt output subblock into a fresh DST and packs it to L1 with packer
// L1-accumulation. K=0 overwrites (pack_reconfig_l1_acc(0)); later K steps
// accumulate (pack_reconfig_l1_acc(1)). With sub == mt,nt this is the single-
// block L1-pack form (MB3.A); subblocking (MB3.B) lets it run when the output
// exceeds DST. The output is repacked to L1 `kt` times -- the cost L1-K trades
// against DST-K's resident-output pack-once.
//
// With fuse != 0 a GELU epilogue is applied. Because the accumulator lives in
// L1, applying it costs a reload: the partials are accumulated into dfb_acc,
// then read back into DST, GELU'd, and packed to dfb_out -- the round trip
// DST-K avoids by keeping the output resident.
//
// A column k (mt tiles) and B row k (nt tiles) are streamed per K step.
//
// Compile-time args: 0 = mt, 1 = nt, 2 = kt, 3 = sub_mt, 4 = sub_nt, 5 = fuse.

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/gelu.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t dfb_in0 = 0;
constexpr uint32_t dfb_in1 = 1;
constexpr uint32_t dfb_acc = 2;
constexpr uint32_t dfb_out = 16;

void kernel_main() {
  const uint32_t mt = get_compile_time_arg_val(0);
  const uint32_t nt = get_compile_time_arg_val(1);
  const uint32_t kt = get_compile_time_arg_val(2);
  const uint32_t sub_mt = get_compile_time_arg_val(3);
  const uint32_t sub_nt = get_compile_time_arg_val(4);
  const uint32_t fuse = get_compile_time_arg_val(5);
  const uint32_t out_tiles = mt * nt;
  const uint32_t sub_tiles = sub_mt * sub_nt;
  const uint32_t acc_cb = fuse ? dfb_acc : dfb_out;

  mm_block_init(dfb_in0, dfb_in1, acc_cb, 0, sub_nt, sub_mt, 1);

  cb_reserve_back(acc_cb, out_tiles);
  pack_reconfig_l1_acc(0); // K=0 overwrites the L1 accumulator
  {
    DeviceZoneScopedN("matmul_k_loop");
    for (uint32_t k = 0; k < kt; ++k) {
      cb_wait_front(dfb_in0, mt);
      cb_wait_front(dfb_in1, nt);
      for (uint32_t om = 0; om < mt; om += sub_mt) {
        for (uint32_t on = 0; on < nt; on += sub_nt) {
          tile_regs_acquire();
          matmul_block(dfb_in0, dfb_in1, om, on, 0, 0, sub_nt, sub_mt, 1);
          tile_regs_commit();
          tile_regs_wait();
          for (uint32_t i = 0; i < sub_mt; ++i) {
            for (uint32_t j = 0; j < sub_nt; ++j) {
              pack_tile<true>(i * sub_nt + j, acc_cb, (om + i) * nt + (on + j));
            }
          }
          tile_regs_release();
        }
      }
      cb_pop_front(dfb_in0, mt);
      cb_pop_front(dfb_in1, nt);
      if (k == 0) {
        pack_reconfig_l1_acc(1);
      }
    }
    pack_reconfig_l1_acc(0);
    if (fuse) {
      // Reload the L1 accumulator into DST, apply GELU, pack to the output.
      cb_push_back(dfb_acc, out_tiles);
      cb_wait_front(dfb_acc, out_tiles);
      cb_reserve_back(dfb_out, out_tiles);
      copy_tile_init(dfb_acc);
      gelu_tile_init();
      for (uint32_t om = 0; om < mt; om += sub_mt) {
        for (uint32_t on = 0; on < nt; on += sub_nt) {
          tile_regs_acquire();
          for (uint32_t i = 0; i < sub_mt; ++i) {
            for (uint32_t j = 0; j < sub_nt; ++j) {
              copy_tile(dfb_acc, (om + i) * nt + (on + j), i * sub_nt + j);
            }
          }
          for (uint32_t t = 0; t < sub_tiles; ++t) {
            gelu_tile(t);
          }
          tile_regs_commit();
          tile_regs_wait();
          for (uint32_t i = 0; i < sub_mt; ++i) {
            for (uint32_t j = 0; j < sub_nt; ++j) {
              pack_tile<true>(i * sub_nt + j, dfb_out,
                              (om + i) * nt + (on + j));
            }
          }
          tile_regs_release();
        }
      }
      cb_pop_front(dfb_acc, out_tiles);
    }
  }
  cb_push_back(dfb_out, out_tiles);
}
