// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/negative.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#endif

template <uint32_t OutCB>
void tensor_address_alias_shim(uint32_t first_address, uint32_t second_address,
                               uint32_t in_cb, uint32_t out_cb) {
  (void)out_cb;
#if defined(COMPILE_FOR_TRISC)
  using namespace ckernel;
  unary_op_init_common(in_cb, OutCB);
  copy_tile_init(in_cb);
  tile_regs_acquire();
  copy_tile(in_cb, 0, 0);
  if (first_address != second_address) {
    negative_tile_init();
    negative_tile(0);
  }
  tile_regs_commit();
  tile_regs_wait();
  pack_tile(0, OutCB);
  tile_regs_release();
#else
  (void)first_address;
  (void)second_address;
  (void)in_cb;
#endif
}
