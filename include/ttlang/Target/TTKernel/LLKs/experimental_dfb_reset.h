// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_DFB_RESET_H
#define TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_DFB_RESET_H

#include <cstdint>

namespace experimental {
namespace dfb_reset_detail {

#if defined(COMPILE_FOR_BRISC) ||                                              \
    (defined(COMPILE_FOR_DM) && COMPILE_FOR_DM == 0)
#define TTL_DFB_RESET_DM0
#endif

#if defined(COMPILE_FOR_NCRISC) ||                                             \
    (defined(COMPILE_FOR_DM) && COMPILE_FOR_DM == 1)
#define TTL_DFB_RESET_DM1
#endif

#if defined(UCK_CHLKC_UNPACK) || defined(TRISC_UNPACK)
#define TTL_DFB_RESET_UNPACK
#endif

#if defined(UCK_CHLKC_PACK) || defined(TRISC_PACK)
#define TTL_DFB_RESET_PACK
#endif

constexpr uint32_t dm0StateWord = 0;
constexpr uint32_t unpackStateWord = 1;
constexpr uint32_t packStateWord = 2;
constexpr uint32_t releaseWord = 3;
constexpr uint32_t stateWordCount = 4;
constexpr uint32_t participantCount = 3;
constexpr uint32_t entryComplete = 1;
constexpr uint32_t exitComplete = 2;

static_assert(releaseWord + 1 == stateWordCount);

FORCE_INLINE uint32_t loadStateWord(volatile uint32_t tt_l1_ptr *stateWord) {
  // Blackhole RISC caches are not coherent across processors on the core.
  asm volatile("fence" ::: "memory");
  uint32_t value;
  asm volatile("lw %[value], (%[address])\n\t"
               "and x0, x0, %[value]"
               : [value] "=r"(value)
               : [address] "r"(stateWord)
               : "memory");
  return value;
}

FORCE_INLINE void storeStateWord(volatile uint32_t tt_l1_ptr *stateWord,
                                 uint32_t value) {
  // The dependent load waits until the store is visible to the other RISCs.
  asm volatile("sw %[value], (%[address])\n\t"
               "lw %[value], (%[address])\n\t"
               "and x0, x0, %[value]"
               : [value] "+r"(value)
               : [address] "r"(stateWord)
               : "memory");
}

FORCE_INLINE bool
participantsHaveState(volatile uint32_t tt_l1_ptr *synchronizationState,
                      uint32_t state) {
  for (uint32_t participant = 0; participant < participantCount;
       ++participant) {
    if (loadStateWord(&synchronizationState[participant]) != state) {
      return false;
    }
  }
  return true;
}

// Every interface owner must complete earlier asynchronous work before a
// reset participant publishes its arrival.
FORCE_INLINE void completeInterfaceWork() {
#if defined(TTL_DFB_RESET_DM0) || defined(TTL_DFB_RESET_DM1)
  noc_async_full_barrier();
#endif
#if defined(TTL_DFB_RESET_UNPACK)
  constexpr uint32_t waitResources = p_stall::UNPACK;
#elif defined(TTL_DFB_RESET_PACK)
  constexpr uint32_t waitResources = p_stall::PACK;
#endif
#if defined(TTL_DFB_RESET_UNPACK) || defined(TTL_DFB_RESET_PACK)
  TTI_STALLWAIT(p_stall::STALL_TDMA, waitResources);
  tensix_sync();
#endif
}

FORCE_INLINE void enter(volatile uint32_t tt_l1_ptr *synchronizationState) {
#if defined(TTL_DFB_RESET_DM0)
  constexpr uint32_t arrivalWord = dm0StateWord;
#elif defined(TTL_DFB_RESET_UNPACK)
  constexpr uint32_t arrivalWord = unpackStateWord;
#elif defined(TTL_DFB_RESET_PACK)
  constexpr uint32_t arrivalWord = packStateWord;
#endif
  completeInterfaceWork();
#if defined(TTL_DFB_RESET_DM0) || defined(TTL_DFB_RESET_UNPACK) ||             \
    defined(TTL_DFB_RESET_PACK)
  storeStateWord(&synchronizationState[arrivalWord], entryComplete);
  while (loadStateWord(&synchronizationState[releaseWord]) != entryComplete) {
  }
#elif defined(TTL_DFB_RESET_DM1)
  while (!participantsHaveState(synchronizationState, entryComplete)) {
  }
  storeStateWord(&synchronizationState[releaseWord], entryComplete);
#endif
}

// DM1 cannot begin later work until every owner completes its interface reset.
FORCE_INLINE void exit(volatile uint32_t tt_l1_ptr *synchronizationState) {
#if defined(TTL_DFB_RESET_DM0)
  constexpr uint32_t arrivalWord = dm0StateWord;
#elif defined(TTL_DFB_RESET_UNPACK)
  constexpr uint32_t arrivalWord = unpackStateWord;
#elif defined(TTL_DFB_RESET_PACK)
  constexpr uint32_t arrivalWord = packStateWord;
#endif
#if defined(TTL_DFB_RESET_DM0) || defined(TTL_DFB_RESET_UNPACK) ||             \
    defined(TTL_DFB_RESET_PACK)
  storeStateWord(&synchronizationState[arrivalWord], exitComplete);
  while (loadStateWord(&synchronizationState[releaseWord]) != exitComplete) {
  }
  storeStateWord(&synchronizationState[arrivalWord], 0);
  while (loadStateWord(&synchronizationState[releaseWord]) != 0) {
  }
#elif defined(TTL_DFB_RESET_DM1)
  while (!participantsHaveState(synchronizationState, exitComplete)) {
  }
  storeStateWord(&synchronizationState[releaseWord], exitComplete);
  while (!participantsHaveState(synchronizationState, 0)) {
  }
  storeStateWord(&synchronizationState[releaseWord], 0);
#endif
}

FORCE_INLINE void applyMask(uint32_t activeMask, uint32_t firstDFBIndex) {
#if defined(TTL_DFB_RESET_DM1) || defined(TTL_DFB_RESET_DM0) ||                \
    defined(TTL_DFB_RESET_UNPACK) || defined(TTL_DFB_RESET_PACK)
  uint32_t dfbIndex = firstDFBIndex;
  while (activeMask != 0) {
    if ((activeMask & 1U) != 0) {
      LocalCBInterface &interface = get_local_cb_interface(dfbIndex);
      const uint32_t base = interface.fifo_limit - interface.fifo_size;
#if defined(TTL_DFB_RESET_DM1)
      interface.fifo_rd_ptr = base;
      interface.fifo_wr_ptr = base;
      interface.tiles_acked_received_init = 0;
      *get_cb_tiles_received_ptr(dfbIndex) = 0;
      *get_cb_tiles_acked_ptr(dfbIndex) = 0;
#elif defined(TTL_DFB_RESET_DM0)
      interface.fifo_rd_ptr = base;
      interface.fifo_wr_ptr = base;
      interface.tiles_acked_received_init = 0;
#elif defined(TTL_DFB_RESET_UNPACK)
      interface.fifo_rd_ptr = base;
      interface.tiles_acked_received_init = 0;
#elif defined(TTL_DFB_RESET_PACK)
      interface.fifo_wr_ptr = base;
      interface.fifo_wr_tile_ptr = 0;
      interface.tiles_acked_received_init = 0;
#endif
    }
    activeMask >>= 1;
    ++dfbIndex;
  }
#else
  (void)activeMask;
  (void)firstDFBIndex;
#endif
}

} // namespace dfb_reset_detail

// Custom resets must complete interface work before publishing arrival.
FORCE_INLINE void complete_dfb_interface_work() {
  dfb_reset_detail::completeInterfaceWork();
}

FORCE_INLINE void reset_dfb_interfaces(uint32_t synchronizationAddress,
                                       uint32_t lowMask, uint32_t highMask) {
#if defined(TTL_DFB_RESET_DM1) || defined(TTL_DFB_RESET_DM0) ||                \
    defined(TTL_DFB_RESET_UNPACK) || defined(TTL_DFB_RESET_PACK)
  auto *synchronizationState =
      reinterpret_cast<volatile uint32_t tt_l1_ptr *>(synchronizationAddress);
  dfb_reset_detail::enter(synchronizationState);
  dfb_reset_detail::applyMask(lowMask, 0);
  dfb_reset_detail::applyMask(highMask, 32);
  dfb_reset_detail::exit(synchronizationState);
#else
  (void)synchronizationAddress;
  (void)lowMask;
  (void)highMask;
#endif
}

#undef TTL_DFB_RESET_DM0
#undef TTL_DFB_RESET_DM1
#undef TTL_DFB_RESET_UNPACK
#undef TTL_DFB_RESET_PACK

} // namespace experimental

#endif
