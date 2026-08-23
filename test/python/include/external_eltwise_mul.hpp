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
  using namespace ckernel;

  static_assert(Lhs::pages_per_block == Rhs::pages_per_block);
  static_assert(Lhs::pages_per_block == Result::pages_per_block);
#if COMPILE_FOR_TRISC == 0
  volatile uint32_t observed_lhs_read_pointer =
      get_local_cb_interface(Lhs::index).fifo_rd_ptr;
  (void)observed_lhs_read_pointer;
#endif
#if COMPILE_FOR_TRISC == 2
  volatile uint32_t observed_result_write_pointer =
      get_local_cb_interface(Result::index).fifo_wr_ptr;
  (void)observed_result_write_pointer;
#endif
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
}
