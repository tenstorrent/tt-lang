// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "internal/hw_thread.h"

__attribute__((noinline)) static void
ttl_dfb_reset_barrier(volatile uint32_t tt_l1_ptr *arrivalSemaphore,
                      volatile uint32_t tt_l1_ptr *releaseSemaphore) {
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) ||               \
    defined(TRISC_UNPACK) || defined(TRISC_MATH) || defined(TRISC_PACK)
  constexpr uint32_t participantCount = 5;
  auto *arrivalEpochs =
      reinterpret_cast<volatile uint8_t tt_l1_ptr *>(arrivalSemaphore);
  const uint32_t hardwareThread = internal_::get_hw_thread_idx();
  const uint8_t nextEpoch = static_cast<uint8_t>(*releaseSemaphore) + 1;

  if (hardwareThread == 0) {
    for (uint32_t participant = 1; participant < participantCount;
         ++participant) {
      while (arrivalEpochs[participant - 1] != nextEpoch) {
        invalidate_l1_cache();
      }
    }
    *releaseSemaphore = nextEpoch;
  } else {
    arrivalEpochs[hardwareThread - 1] = nextEpoch;
    while (static_cast<uint8_t>(*releaseSemaphore) != nextEpoch) {
      invalidate_l1_cache();
    }
  }
#endif
}

static inline void ttl_reset_one_dfb_state(uint32_t dfb) {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC) ||               \
    defined(TRISC_UNPACK) || defined(TRISC_PACK)
  LocalCBInterface &interface = get_local_cb_interface(dfb);
  const uint32_t base = interface.fifo_limit - interface.fifo_size;
#if defined(COMPILE_FOR_NCRISC)
  interface.fifo_rd_ptr = base;
  interface.fifo_wr_ptr = base;
  interface.tiles_acked_received_init = 0;
  *get_cb_tiles_received_ptr(dfb) = 0;
  *get_cb_tiles_acked_ptr(dfb) = 0;
#elif defined(COMPILE_FOR_BRISC)
  interface.fifo_rd_ptr = base;
  interface.fifo_wr_ptr = base;
  interface.tiles_acked_received_init = 0;
#elif defined(TRISC_UNPACK)
  interface.fifo_rd_ptr = base;
  interface.tiles_acked_received_init = 0;
#elif defined(TRISC_PACK)
  interface.fifo_wr_ptr = base;
  interface.fifo_wr_tile_ptr = 0;
  interface.tiles_acked_received_init = 0;
#endif
#endif
}

static inline void ttl_reset_dfb_state(uint32_t dfb,
                                       uint32_t enterSemaphoreAddress,
                                       uint32_t exitSemaphoreAddress) {
  auto *enterSemaphore =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(enterSemaphoreAddress);
  auto *exitSemaphore =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(exitSemaphoreAddress);
  ttl_dfb_reset_barrier(enterSemaphore, exitSemaphore);
  ttl_reset_one_dfb_state(dfb);
  ttl_dfb_reset_barrier(enterSemaphore, exitSemaphore);
}

static inline void ttl_reset_all_dfb_state(uint32_t enterSemaphoreAddress,
                                           uint32_t exitSemaphoreAddress) {
  auto *enterSemaphore =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(enterSemaphoreAddress);
  auto *exitSemaphore =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(exitSemaphoreAddress);
  ttl_dfb_reset_barrier(enterSemaphore, exitSemaphore);
  for (uint32_t dfb = 0; dfb < NUM_CIRCULAR_BUFFERS; ++dfb) {
    ttl_reset_one_dfb_state(dfb);
  }
  ttl_dfb_reset_barrier(enterSemaphore, exitSemaphore);
}
