// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#endif

template <typename Lhs, typename Rhs, typename Result, uint32_t Rows,
          uint32_t Inner, uint32_t Columns>
static inline void ttl_external_matmul() {
#if defined(COMPILE_FOR_TRISC)
  static_assert(Rows > 0 && Inner > 0 && Columns > 0);
  static_assert(Lhs::pages_per_block == Rows * Inner);
  static_assert(Rhs::pages_per_block == Inner * Columns);
  static_assert(Result::pages_per_block == Rows * Columns);
#if TTLANG_DFB_STORAGE_COMPILER_L1
  Lhs lhs = Lhs::bind();
  Rhs rhs = Rhs::bind();
  Result result = Result::bind();
  result.reserve_back(Result::pages_per_block);
  lhs.wait_front(Lhs::pages_per_block);
  rhs.wait_front(Rhs::pages_per_block);
  ttlang::l1::target::ComputeContext computeContext;
  computeContext.matmulBlockInit(lhs, rhs, result, 0, Columns, Rows, Inner);
  tile_regs_acquire();
  ttlang::l1::target::matmul_block_strided(lhs, rhs, 0, 0, 0, 0, Columns, Rows,
                                           Inner, Columns);
  tile_regs_commit();
  tile_regs_wait();
  for (uint32_t tile = 0; tile < Result::pages_per_block; ++tile) {
    ttlang::l1::target::pack_tile<true>(tile, result, tile);
  }
  tile_regs_release();
  result.push_back(Result::pages_per_block);
  lhs.pop_front(Lhs::pages_per_block);
  rhs.pop_front(Rhs::pages_per_block);
#else
  using namespace ckernel;

  // TT-Metal requires one hardware startup call before the first compute API.
  compute_kernel_hw_startup<SrcOrder::Reverse>(Lhs::index, Rhs::index,
                                               Result::index);
  cb_reserve_back(Result::index, Result::pages_per_block);
  cb_wait_front(Lhs::index, Lhs::pages_per_block);
  cb_wait_front(Rhs::index, Rhs::pages_per_block);
  matmul_block_init(Lhs::index, Rhs::index, 0, Columns, Rows, Inner);
  tile_regs_acquire();
  for (uint32_t reduction = 0; reduction < Inner; ++reduction) {
    matmul_block(Lhs::index, Rhs::index, reduction, reduction * Columns, 0, 0,
                 Columns, Rows, Inner);
  }
  tile_regs_commit();
  tile_regs_wait();
  for (uint32_t tile = 0; tile < Result::pages_per_block; ++tile) {
    pack_tile<true>(tile, Result::index, tile);
  }
  tile_regs_release();
  cb_push_back(Result::index, Result::pages_per_block);
  cb_pop_front(Lhs::index, Lhs::pages_per_block);
  cb_pop_front(Rhs::index, Rhs::pages_per_block);
#endif
#endif
}
