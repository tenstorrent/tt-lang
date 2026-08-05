// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#endif

template <typename Lhs, typename Rhs, typename Result>
inline void external_dfb_multiply() {
  static_assert(Lhs::pages_per_block == Rhs::pages_per_block);
  static_assert(Lhs::pages_per_block == Result::pages_per_block);
  static_assert(Lhs::block_count == 2);
  static_assert(Rhs::block_count == 2);
  static_assert(Result::block_count == 2);
  static_assert(Lhs::page_size_bytes == Rhs::page_size_bytes);
  static_assert(Lhs::page_size_bytes == Result::page_size_bytes);
#if defined(COMPILE_FOR_TRISC)
  using namespace ckernel;

  binary_op_init_common(Lhs::index, Rhs::index, Result::index);
  mul_tiles_init(Lhs::index, Rhs::index);
  tile_regs_acquire();
  mul_tiles(Lhs::index, Rhs::index, 0, 0, 0);
  tile_regs_commit();
  tile_regs_wait();
  pack_tile(0, Result::index);
  tile_regs_release();
#endif
}
