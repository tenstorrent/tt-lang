// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_DFB_RESET_H
#define TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_DFB_RESET_H

#include <cstdint>

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_RISC_BARRIER_H
#include "ttlang/Target/TTKernel/LLKs/risc_barrier.h"
#endif

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

#if defined(UCK_CHLKC_MATH) || defined(TRISC_MATH)
#define TTL_DFB_RESET_MATH
#endif

#if defined(UCK_CHLKC_PACK) || defined(TRISC_PACK)
#define TTL_DFB_RESET_PACK
#endif

constexpr uint32_t stateWordCount = 4;

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
    defined(TTL_DFB_RESET_UNPACK) || defined(TTL_DFB_RESET_MATH) ||            \
    defined(TTL_DFB_RESET_PACK)
  auto *synchronizationState =
      reinterpret_cast<volatile uint32_t tt_l1_ptr *>(synchronizationAddress);
  const uint32_t participantStateAddresses[] = {
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(
          &synchronizationState[0])),
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(
          &synchronizationState[1])),
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(
          &synchronizationState[2])),
      static_cast<uint32_t>(reinterpret_cast<uintptr_t>(
          &synchronizationState[3])),
  };
  dfb_reset_detail::completeInterfaceWork();
  ttlang::detail::riscBarrierEnter(
      participantStateAddresses[0], participantStateAddresses[1],
      participantStateAddresses[2], participantStateAddresses[3]);
  dfb_reset_detail::applyMask(lowMask, 0);
  dfb_reset_detail::applyMask(highMask, 32);
  ttlang::detail::riscBarrierExit(
      participantStateAddresses[0], participantStateAddresses[1],
      participantStateAddresses[2], participantStateAddresses[3]);
#else
  (void)synchronizationAddress;
  (void)lowMask;
  (void)highMask;
#endif
}

#undef TTL_DFB_RESET_DM0
#undef TTL_DFB_RESET_DM1
#undef TTL_DFB_RESET_UNPACK
#undef TTL_DFB_RESET_MATH
#undef TTL_DFB_RESET_PACK

} // namespace experimental

#endif
