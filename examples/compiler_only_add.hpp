// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"

namespace ckernel {

// Each DFB holds one acquired or reserved tile; TTL owns the DFB transactions.
inline void compiler_only_add(uint32_t aDfb, uint32_t bDfb, uint32_t outDfb) {
  binary_op_init_common(aDfb, bDfb, outDfb);
  add_tiles_init(aDfb, bDfb);
  tile_regs_acquire();
  add_tiles(aDfb, bDfb, 0, 0, 0);
  tile_regs_commit();
  tile_regs_wait();
  pack_tile(0, outDfb);
  tile_regs_release();
}

} // namespace ckernel
#endif
