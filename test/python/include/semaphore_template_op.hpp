// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Accept the same semaphore capture in both argument forms so the test detects
// signature drift without requiring synchronization in the compute body.

#pragma once

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#endif

template <uint32_t InCB, uint32_t OutCB, uint32_t SemaphoreAddr>
void semaphore_template_negate_shim(uint32_t, uint32_t,
                                    uint32_t semaphore_address) {
  static_assert(SemaphoreAddr != 0);
  (void)semaphore_address;
#if defined(COMPILE_FOR_TRISC)
  using namespace ckernel;

  unary_op_init_common(InCB, OutCB);
  copy_tile_init(InCB);

  tile_regs_acquire();
  copy_tile(InCB, 0, 0);
  negative_tile_init();
  negative_tile(0);
  tile_regs_commit();
  tile_regs_wait();
  pack_tile(0, OutCB);
  tile_regs_release();
#endif
}
