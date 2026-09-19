// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// External compute operation used to validate typed DFB descriptors. The
// external function owns the compute-thread DFB protocol.
#pragma once

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#endif

template <typename Lhs, typename Rhs, typename Result>
static inline void ttl_external_eltwise_mul() {
#if defined(COMPILE_FOR_TRISC)
  static_assert(Lhs::pages_per_block == Rhs::pages_per_block);
  static_assert(Lhs::pages_per_block == Result::pages_per_block);
  // TT-Metal compute APIs require descriptor indices, while compiler-managed
  // storage uses the target's address-based compute interface.
#if TTLANG_DFB_STORAGE_COMPILER_L1
  Lhs lhs = Lhs::bind();
  Rhs rhs = Rhs::bind();
  Result result = Result::bind();
  result.reserve_back(Result::pages_per_block);
  lhs.wait_front(Lhs::pages_per_block);
  rhs.wait_front(Rhs::pages_per_block);
  ttlang::l1::target::ComputeContext computeContext;
  computeContext.configure(lhs, rhs, result);
  tile_regs_acquire();
  ttlang::l1::target::mul_tiles_init(lhs, rhs);
  ttlang::l1::target::mul_tiles(lhs, rhs, 0, 0, 0);
  tile_regs_commit();
  tile_regs_wait();
  ttlang::l1::target::pack_tile<true>(0, result, 0);
  tile_regs_release();
  result.push_back(Result::pages_per_block);
  lhs.pop_front(Lhs::pages_per_block);
  rhs.pop_front(Rhs::pages_per_block);
#else
  using namespace ckernel;

  cb_reserve_back(Result::index, Result::pages_per_block);
  cb_wait_front(Lhs::index, Lhs::pages_per_block);
  cb_wait_front(Rhs::index, Rhs::pages_per_block);
  binary_op_init_common(Lhs::index, Rhs::index, Result::index);
  mul_tiles_init(Lhs::index, Rhs::index);
  tile_regs_acquire();
  mul_tiles(Lhs::index, Rhs::index, 0, 0, 0);
  tile_regs_commit();
  cb_pop_front(Lhs::index, Lhs::pages_per_block);
  cb_pop_front(Rhs::index, Rhs::pages_per_block);
  tile_regs_wait();
  pack_tile(0, Result::index);
  cb_push_back(Result::index, Result::pages_per_block);
  tile_regs_release();
#endif
#endif
}
