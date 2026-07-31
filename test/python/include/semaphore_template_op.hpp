// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// External compute shim that accepts a semaphore address as a template arg.
// The semaphore value is intentionally unused in the kernel body; this test
// validates frontend/lowering handling of real ttnn GlobalSemaphore objects
// passed through ttl.call_extern_func template_args.

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
void semaphore_template_negate_shim() {
#if defined(COMPILE_FOR_TRISC)
  using namespace ckernel;
  (void)SemaphoreAddr;

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
#else
  (void)SemaphoreAddr;
#endif
}
