// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#endif

template <typename Source, typename Completion>
inline void consume_repeated_dfb_and_signal() {
#if defined(COMPILE_FOR_TRISC)
  using namespace ckernel;
  for (uint32_t transaction = 0; transaction < 4; ++transaction) {
    cb_wait_front(Source::index, Source::pages_per_block);
    cb_pop_front(Source::index, Source::pages_per_block);
  }
  cb_reserve_back(Completion::index, Completion::pages_per_block);
  cb_push_back(Completion::index, Completion::pages_per_block);
#endif
}

template <typename Source, typename Destination>
inline void copy_repeated_dfb() {
#if defined(COMPILE_FOR_TRISC)
  using namespace ckernel;
  unary_op_init_common(Source::index, Destination::index);
  copy_tile_init(Source::index);
  for (uint32_t transaction = 0; transaction < 4; ++transaction) {
    cb_wait_front(Source::index, Source::pages_per_block);
    cb_reserve_back(Destination::index, Destination::pages_per_block);
    for (uint32_t tile = 0; tile < Source::pages_per_block; ++tile) {
      tile_regs_acquire();
      copy_tile(Source::index, tile, 0);
      tile_regs_commit();
      tile_regs_wait();
      pack_tile(0, Destination::index);
      tile_regs_release();
    }
    cb_pop_front(Source::index, Source::pages_per_block);
    cb_push_back(Destination::index, Destination::pages_per_block);
  }
#endif
}
