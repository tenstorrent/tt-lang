// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/tile_move_copy.h"
#endif

template <typename SourceDescriptor, typename Destination>
inline void copy_raw_tile_without_consuming_source(uint32_t sourceAddress) {
#if defined(COMPILE_FOR_TRISC)
  using namespace ckernel;

  uint32_t savedReadPointer = 0;
  UNPACK(({
    auto &sourceInterface = get_local_cb_interface(SourceDescriptor::index);
    savedReadPointer = sourceInterface.fifo_rd_ptr;
    sourceInterface.fifo_rd_ptr = sourceAddress >> cb_addr_shift;
  }));

  cb_reserve_back(Destination::index, Destination::pages_per_block);
  unary_op_init_common(SourceDescriptor::index, Destination::index);
  copy_tile_init(SourceDescriptor::index);
  tile_regs_acquire();
  copy_tile(SourceDescriptor::index, 0, 0);
  tile_regs_commit();
  tile_regs_wait();
  pack_tile(0, Destination::index);
  tile_regs_release();
  cb_push_back(Destination::index, Destination::pages_per_block);

  UNPACK(({
    get_local_cb_interface(SourceDescriptor::index).fifo_rd_ptr =
        savedReadPointer;
  }));
#else
  (void)sourceAddress;
#endif
}
