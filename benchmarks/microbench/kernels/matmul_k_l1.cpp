// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MB3, L1-K matmul accumulation (production-representative): C[mt,nt] = sum_k
// A[k] @ B[k] over `kt` K-tiles, with each K step's mt*nt subblock matmul'd
// into a fresh DST and packed to L1 with packer L1-accumulation. K=0 overwrites
// (pack_reconfig_l1_acc(0)), later steps accumulate (pack_reconfig_l1_acc(1)).
// This is the strategy when the output does not fit DST across the K loop.
//
// Compile-time args: 0 = mt (output rows, tiles), 1 = nt (output cols, tiles),
// 2 = kt (K-depth, tiles).

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/dataflow/circular_buffer.h"
#include "tools/profiler/kernel_profiler.hpp"

constexpr uint32_t dfb_in0 = 0;
constexpr uint32_t dfb_in1 = 1;
constexpr uint32_t dfb_out = 16;

void kernel_main() {
  const uint32_t mt = get_compile_time_arg_val(0);
  const uint32_t nt = get_compile_time_arg_val(1);
  const uint32_t kt = get_compile_time_arg_val(2);
  const uint32_t out_tiles = mt * nt;

  mm_block_init(dfb_in0, dfb_in1, dfb_out, 0, nt, mt, 1);

  cb_reserve_back(dfb_out, out_tiles);
  pack_reconfig_l1_acc(0); // K=0 overwrites the L1 accumulator
  {
    DeviceZoneScopedN("matmul_k_loop");
    for (uint32_t k = 0; k < kt; ++k) {
      cb_wait_front(dfb_in0, mt);
      cb_wait_front(dfb_in1, nt);
      tile_regs_acquire();
      matmul_block(dfb_in0, dfb_in1, 0, 0, 0, 0, nt, mt, 1);
      tile_regs_commit();
      tile_regs_wait();
      for (uint32_t i = 0; i < out_tiles; ++i) {
        pack_tile<true>(i, dfb_out, i); // packer adds into L1 after K=0
      }
      tile_regs_release();
      cb_pop_front(dfb_in0, mt);
      cb_pop_front(dfb_in1, nt);
      if (k == 0) {
        pack_reconfig_l1_acc(1);
      }
    }
  }
  pack_reconfig_l1_acc(0);
  cb_push_back(dfb_out, out_tiles);
}
