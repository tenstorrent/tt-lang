// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// External compute operation used to validate direct DFB operands. The
// external function owns the compute-thread DFB protocol.
#pragma once

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#endif

static inline void ttl_external_eltwise_mul(int lhs, int rhs, int result) {
#if defined(COMPILE_FOR_TRISC)
  using namespace ckernel;

  cb_reserve_back(result, 1);
  cb_wait_front(lhs, 1);
  cb_wait_front(rhs, 1);
  binary_op_init_common(lhs, rhs, result);
  mul_tiles_init(lhs, rhs);
  tile_regs_acquire();
  mul_tiles(lhs, rhs, 0, 0, 0);
  tile_regs_commit();
  cb_pop_front(lhs, 1);
  cb_pop_front(rhs, 1);
  tile_regs_wait();
  pack_tile(0, result);
  cb_push_back(result, 1);
  tile_regs_release();
#else
  (void)lhs;
  (void)rhs;
  (void)result;
#endif
}
