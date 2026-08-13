// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#pragma once

__attribute__((noinline)) static void
ttl_dfb_reset_barrier(volatile uint32_t tt_l1_ptr *semaphore) {
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) ||               \
    defined(TRISC_UNPACK) || defined(TRISC_MATH) || defined(TRISC_PACK)
  uint32_t generationEnd;
#if defined(COMPILE_FOR_NCRISC)
  const uint32_t generationBase =
      __atomic_load_n(semaphore, __ATOMIC_ACQUIRE) & ~7u;
  while (__atomic_load_n(semaphore, __ATOMIC_ACQUIRE) < generationBase + 4u) {
  }
  __atomic_fetch_add(semaphore, 4u, __ATOMIC_ACQ_REL);
  generationEnd = generationBase + 8u;
#else
  const uint32_t ticket = __atomic_fetch_add(semaphore, 1u, __ATOMIC_ACQ_REL);
  generationEnd = (ticket & ~7u) + 8u;
#endif
  while (__atomic_load_n(semaphore, __ATOMIC_ACQUIRE) < generationEnd) {
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
  ttl_dfb_reset_barrier(enterSemaphore);
  ttl_reset_one_dfb_state(dfb);
  ttl_dfb_reset_barrier(exitSemaphore);
}

static inline void ttl_reset_all_dfb_state(uint32_t enterSemaphoreAddress,
                                           uint32_t exitSemaphoreAddress) {
  auto *enterSemaphore =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(enterSemaphoreAddress);
  auto *exitSemaphore =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(exitSemaphoreAddress);
  ttl_dfb_reset_barrier(enterSemaphore);
  for (uint32_t dfb = 0; dfb < 64; ++dfb) {
    ttl_reset_one_dfb_state(dfb);
  }
  ttl_dfb_reset_barrier(exitSemaphore);
}
