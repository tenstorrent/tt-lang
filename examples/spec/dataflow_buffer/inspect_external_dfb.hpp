// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/common.h"
#endif

template <typename Descriptor>
inline void inspect_dfb() {
#if defined(COMPILE_FOR_TRISC)
  using namespace ckernel;

  uint32_t readPointer = 0;
  UNPACK(({
    readPointer = get_local_cb_interface(Descriptor::index).fifo_rd_ptr;
  }));
  (void)readPointer;
#endif
}
