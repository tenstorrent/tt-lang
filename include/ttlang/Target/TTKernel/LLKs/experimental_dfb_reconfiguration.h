// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_DFB_RECONFIGURATION_H
#define TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_DFB_RECONFIGURATION_H

#include <cstdint>

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

#if defined(UCK_CHLKC_PACK) || defined(TRISC_PACK)
#define TTL_DFB_RECONFIGURATION_PACK
#endif

#if defined(UCK_CHLKC_MATH) || defined(TRISC_MATH)
#define TTL_DFB_RECONFIGURATION_MATH
#endif

// Each core stores 64 four-word interface records, two active masks, and two
// synchronization words in one shared L1 allocation.
constexpr uint32_t lowMaskWord = 256;
constexpr uint32_t highMaskWord = 257;
constexpr uint32_t synchronizationWord = 258;
constexpr uint32_t dm0ArrivalBit = 1U << 0;
constexpr uint32_t unpackArrivalBit = 1U << 1;
constexpr uint32_t packArrivalBit = 1U << 2;
constexpr uint32_t mathArrivalBit = 1U << 3;
constexpr uint32_t allArrivalBits =
    dm0ArrivalBit | unpackArrivalBit | packArrivalBit | mathArrivalBit;
constexpr uint32_t entryReleaseBit = 1U << 31;
constexpr uint32_t exitWaitingParticipantCount = 4;

// Unpack and math handshake so pack completes before any TRISC polls L1.
FORCE_INLINE void quiesceComputePipeline() {
#if defined(TTL_DFB_RECONFIGURATION_UNPACK)
  tensix_sync();
  mailbox_write(ThreadId::MathThreadId, 1);
  (void)mailbox_read(ThreadId::MathThreadId);
#elif defined(TTL_DFB_RECONFIGURATION_MATH)
  tensix_sync();
  (void)mailbox_read(ThreadId::UnpackThreadId);
  while (semaphore_read(semaphore::MATH_PACK) > 0) {
  }
  mailbox_write(ThreadId::UnpackThreadId, 1);
#elif defined(TTL_DFB_RECONFIGURATION_PACK)
  tensix_sync();
#endif
}

// DM1 releases the other RISCs only after all prior-epoch work is complete
// because reconfiguration changes DFB state shared across those RISCs.
FORCE_INLINE void enter(volatile uint32_t tt_l1_ptr *synchronizationState) {
#if defined(TTL_DFB_RECONFIGURATION_DM0) ||                                    \
    defined(TTL_DFB_RECONFIGURATION_UNPACK) ||                                 \
    defined(TTL_DFB_RECONFIGURATION_PACK) ||                                   \
    defined(TTL_DFB_RECONFIGURATION_MATH)
  quiesceComputePipeline();
#if defined(TTL_DFB_RECONFIGURATION_DM0)
  constexpr uint32_t arrivalBit = dm0ArrivalBit;
#elif defined(TTL_DFB_RECONFIGURATION_UNPACK)
  constexpr uint32_t arrivalBit = unpackArrivalBit;
#elif defined(TTL_DFB_RECONFIGURATION_PACK)
  constexpr uint32_t arrivalBit = packArrivalBit;
#elif defined(TTL_DFB_RECONFIGURATION_MATH)
  constexpr uint32_t arrivalBit = mathArrivalBit;
#endif
  __atomic_fetch_or(&synchronizationState[0], arrivalBit, __ATOMIC_RELEASE);
  while ((__atomic_load_n(&synchronizationState[0], __ATOMIC_RELAXED) &
          entryReleaseBit) == 0) {
  }
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  __atomic_fetch_and(&synchronizationState[0], ~arrivalBit, __ATOMIC_RELEASE);
#elif defined(TTL_DFB_RECONFIGURATION_DM1)
  while ((__atomic_load_n(&synchronizationState[0], __ATOMIC_RELAXED) &
          allArrivalBits) != allArrivalBits) {
  }
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  __atomic_fetch_or(&synchronizationState[0], entryReleaseBit,
                    __ATOMIC_RELEASE);
  while ((__atomic_load_n(&synchronizationState[0], __ATOMIC_RELAXED) &
          allArrivalBits) != 0) {
  }
  __atomic_store_n(&synchronizationState[0], 0, __ATOMIC_RELAXED);
#endif
}

// DM1 releases the other participants only after updating shared stream state.
FORCE_INLINE void exit(volatile uint32_t tt_l1_ptr *synchronizationState) {
#if defined(TTL_DFB_RECONFIGURATION_DM1)
  __atomic_fetch_add(&synchronizationState[1], exitWaitingParticipantCount,
                     __ATOMIC_RELEASE);
#elif defined(TTL_DFB_RECONFIGURATION_DM0) ||                                  \
    defined(TTL_DFB_RECONFIGURATION_UNPACK) ||                                 \
    defined(TTL_DFB_RECONFIGURATION_PACK) ||                                   \
    defined(TTL_DFB_RECONFIGURATION_MATH)
  while (__atomic_load_n(&synchronizationState[1], __ATOMIC_RELAXED) == 0) {
  }
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
  __atomic_fetch_sub(&synchronizationState[1], 1, __ATOMIC_RELEASE);
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
      uint32_t fifoAddress =
          configuration[configurationOffset] >> cb_addr_shift;
      uint32_t fifoSize =
          configuration[configurationOffset + 1] >> cb_addr_shift;
      uint32_t fifoNumPages = configuration[configurationOffset + 2];
      uint32_t fifoPageSize =
          configuration[configurationOffset + 3] >> cb_addr_shift;

      LocalCBInterface &interface = get_local_cb_interface(dfbIndex);
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
    defined(TTL_DFB_RECONFIGURATION_PACK) ||                                   \
    defined(TTL_DFB_RECONFIGURATION_MATH)
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
  dfb_reconfiguration_detail::enter(synchronizationState);
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
  dfb_reconfiguration_detail::exit(synchronizationState);
#else
  (void)configurationAddress;
#endif
}

#undef TTL_DFB_RECONFIGURATION_DM0
#undef TTL_DFB_RECONFIGURATION_DM1
#undef TTL_DFB_RECONFIGURATION_UNPACK
#undef TTL_DFB_RECONFIGURATION_PACK
#undef TTL_DFB_RECONFIGURATION_MATH

} // namespace experimental

#endif
