// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// MB3, DST-K matmul accumulation (production-representative): C[mt,nt] = sum_k
// A[k] @ B[k] over `kt` K-tiles. The mt*nt output subblock is held in DST
// across the whole K loop (one acquire) and packed once with pack_tile_block.
// Each K step is one matmul_block over the mt*nt subblock (rt_dim=mt,
// ct_dim=nt, kt_dim=1), accumulating into DST. A and B are prefetched a block
// at a time. Legal only while mt*nt <= getDstCapacity (the P > capacity case is
// the spill/reload variant).
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

  tile_regs_acquire();
  {
    DeviceZoneScopedN("matmul_k_loop");
    for (uint32_t k = 0; k < kt; ++k) {
      cb_wait_front(dfb_in0, mt);
      cb_wait_front(dfb_in1, nt);
      matmul_block(dfb_in0, dfb_in1, 0, 0, 0, 0, nt, mt,
                   1);
      cb_pop_front(dfb_in0, mt);
      cb_pop_front(dfb_in1, nt);
    }
  }
  tile_regs_commit();
  tile_regs_wait();
  cb_reserve_back(dfb_out, out_tiles);
  pack_tile_block(0, dfb_out, out_tiles);
  cb_push_back(dfb_out, out_tiles);
  tile_regs_release();
}
