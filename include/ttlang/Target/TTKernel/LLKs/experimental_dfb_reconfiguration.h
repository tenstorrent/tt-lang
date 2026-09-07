// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_DFB_RECONFIGURATION_H
#define TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_DFB_RECONFIGURATION_H

#include <cstdint>

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_RISC_BARRIER_H
#include "ttlang/Target/TTKernel/LLKs/risc_barrier.h"
#endif

namespace experimental {
namespace dfb_reconfiguration_detail {

#if defined(COMPILE_FOR_BRISC) ||                                              \
    (defined(COMPILE_FOR_DM) && COMPILE_FOR_DM == 0)
#define TTL_DFB_RECONFIGURATION_DM0
#endif

#if defined(COMPILE_FOR_NCRISC) ||                                             \
    (defined(COMPILE_FOR_DM) && COMPILE_FOR_DM == 1)
#define TTL_DFB_RECONFIGURATION_DM1
#endif

#if defined(UCK_CHLKC_UNPACK) || defined(TRISC_UNPACK)
#define TTL_DFB_RECONFIGURATION_UNPACK
#endif

#if defined(UCK_CHLKC_MATH) || defined(TRISC_MATH)
#define TTL_DFB_RECONFIGURATION_MATH
#endif

#if defined(UCK_CHLKC_PACK) || defined(TRISC_PACK)
#define TTL_DFB_RECONFIGURATION_PACK
#endif

// Each core stores 64 four-word interface records, two active masks, four
// participant state words, and two padding words in shared L1.
constexpr uint32_t lowMaskWord = 256;
constexpr uint32_t highMaskWord = 257;
constexpr uint32_t synchronizationWord = 258;
constexpr uint32_t completionMarker = 0xD1FB;
constexpr uint32_t preserveFifoAddress = 0;

FORCE_INLINE void drainComputeEngine() {
#if defined(TTL_DFB_RECONFIGURATION_UNPACK)
  constexpr uint32_t waitResources = p_stall::UNPACK;
  constexpr uint32_t completionGpr = p_gpr_unpack::TMP0;
#elif defined(TTL_DFB_RECONFIGURATION_PACK)
  constexpr uint32_t waitResources = p_stall::PACK;
  constexpr uint32_t completionGpr = p_gpr_pack::TMP0;
#endif
#if defined(TTL_DFB_RECONFIGURATION_UNPACK) ||                                 \
    defined(TTL_DFB_RECONFIGURATION_PACK)
  TTI_STALLWAIT(p_stall::STALL_TDMA, waitResources);
  TTI_SETDMAREG(0, completionMarker, 0, LO_16(completionGpr));
  sync_regfile_write(completionGpr);
#endif
}

FORCE_INLINE void completeInterfaceWork() {
#if defined(TTL_DFB_RECONFIGURATION_DM0) ||                                   \
    defined(TTL_DFB_RECONFIGURATION_DM1)
  noc_async_full_barrier();
#elif defined(TTL_DFB_RECONFIGURATION_UNPACK) ||                               \
    defined(TTL_DFB_RECONFIGURATION_PACK)
  drainComputeEngine();
#endif
}

template <bool updateReadPointer, bool updateWritePointer,
          bool updateWriteTilePointer, bool resetStreamCounters>
FORCE_INLINE void applyMask(uint32_t tt_l1_ptr *configuration,
                            uint32_t activeMask, uint32_t firstDfbIndex) {
  uint32_t dfbIndex = firstDfbIndex;
  while (activeMask != 0) {
    if ((activeMask & 1U) != 0) {
      uint32_t configurationOffset = dfbIndex * 4;
      uint32_t configuredAddress = configuration[configurationOffset];
      uint32_t fifoSize =
          configuration[configurationOffset + 1] >> cb_addr_shift;
      uint32_t fifoNumPages = configuration[configurationOffset + 2];
      uint32_t fifoPageSize =
          configuration[configurationOffset + 3] >> cb_addr_shift;

      LocalCBInterface &interface = get_local_cb_interface(dfbIndex);
      uint32_t fifoAddress =
          configuredAddress == preserveFifoAddress
              ? interface.fifo_limit - interface.fifo_size
              : configuredAddress >> cb_addr_shift;
      if constexpr (updateReadPointer) {
        interface.fifo_rd_ptr = fifoAddress;
      }
      if constexpr (updateWritePointer) {
        interface.fifo_wr_ptr = fifoAddress;
        interface.fifo_num_pages = fifoNumPages;
      }
      if constexpr (updateWriteTilePointer) {
        interface.fifo_wr_tile_ptr = 0;
      }
      interface.fifo_size = fifoSize;
      interface.fifo_limit = fifoAddress + fifoSize;
      interface.fifo_page_size = fifoPageSize;
      interface.tiles_acked_received_init = 0;

      if constexpr (resetStreamCounters) {
        *get_cb_tiles_received_ptr(dfbIndex) = 0;
        *get_cb_tiles_acked_ptr(dfbIndex) = 0;
      }
    }
    activeMask >>= 1;
    ++dfbIndex;
  }
}

} // namespace dfb_reconfiguration_detail

FORCE_INLINE void reconfigure_dfb_interfaces(uint32_t configurationAddress) {
#if defined(TTL_DFB_RECONFIGURATION_DM1) ||                                    \
    defined(TTL_DFB_RECONFIGURATION_DM0) ||                                    \
    defined(TTL_DFB_RECONFIGURATION_UNPACK) ||                                 \
    defined(TTL_DFB_RECONFIGURATION_MATH) ||                                   \
    defined(TTL_DFB_RECONFIGURATION_PACK)
#if !defined(TTL_DFB_RECONFIGURATION_MATH)
#if defined(TTL_DFB_RECONFIGURATION_DM1)
  constexpr bool updateReadPointer = true;
  constexpr bool updateWritePointer = true;
  constexpr bool updateWriteTilePointer = false;
  constexpr bool resetStreamCounters = true;
#elif defined(TTL_DFB_RECONFIGURATION_DM0)
  constexpr bool updateReadPointer = true;
  constexpr bool updateWritePointer = true;
  constexpr bool updateWriteTilePointer = false;
  constexpr bool resetStreamCounters = false;
#elif defined(TTL_DFB_RECONFIGURATION_UNPACK)
  constexpr bool updateReadPointer = true;
  constexpr bool updateWritePointer = false;
  constexpr bool updateWriteTilePointer = false;
  constexpr bool resetStreamCounters = false;
#elif defined(TTL_DFB_RECONFIGURATION_PACK)
  constexpr bool updateReadPointer = false;
  constexpr bool updateWritePointer = true;
  constexpr bool updateWriteTilePointer = true;
  constexpr bool resetStreamCounters = false;
#endif
#endif

  auto *configuration =
      reinterpret_cast<uint32_t tt_l1_ptr *>(configurationAddress);
  auto *synchronizationState = reinterpret_cast<volatile uint32_t tt_l1_ptr *>(
      &configuration[dfb_reconfiguration_detail::synchronizationWord]);
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
  dfb_reconfiguration_detail::completeInterfaceWork();
  ttlang::detail::riscBarrierEnter(
      participantStateAddresses[0], participantStateAddresses[1],
      participantStateAddresses[2], participantStateAddresses[3]);
#if !defined(TTL_DFB_RECONFIGURATION_MATH)
  dfb_reconfiguration_detail::applyMask<updateReadPointer, updateWritePointer,
                                        updateWriteTilePointer,
                                        resetStreamCounters>(
      configuration, configuration[dfb_reconfiguration_detail::lowMaskWord], 0);
  dfb_reconfiguration_detail::applyMask<updateReadPointer, updateWritePointer,
                                        updateWriteTilePointer,
                                        resetStreamCounters>(
      configuration, configuration[dfb_reconfiguration_detail::highMaskWord],
      32);
#endif
  ttlang::detail::riscBarrierExit(
      participantStateAddresses[0], participantStateAddresses[1],
      participantStateAddresses[2], participantStateAddresses[3]);
#else
  (void)configurationAddress;
#endif
}

#undef TTL_DFB_RECONFIGURATION_DM0
#undef TTL_DFB_RECONFIGURATION_DM1
#undef TTL_DFB_RECONFIGURATION_UNPACK
#undef TTL_DFB_RECONFIGURATION_MATH
#undef TTL_DFB_RECONFIGURATION_PACK

} // namespace experimental

#endif
