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

// Each core stores 64 four-word interface records, two active masks, three
// arrival words, one release word, and two padding words in shared L1.
constexpr uint32_t lowMaskWord = 256;
constexpr uint32_t highMaskWord = 257;
constexpr uint32_t synchronizationWord = 258;
constexpr uint32_t dm0StateWord = 0;
constexpr uint32_t unpackStateWord = 1;
constexpr uint32_t packStateWord = 2;
constexpr uint32_t releaseWord = 3;
constexpr uint32_t participantCount = 3;
constexpr uint32_t entryComplete = 1;
constexpr uint32_t exitComplete = 2;
constexpr uint32_t completionMarker = 0xD1FB;

FORCE_INLINE uint32_t
loadSynchronizationWord(volatile uint32_t tt_l1_ptr *synchronizationWord) {
  // Blackhole RISC caches are not coherent across processors on the core.
  asm volatile("fence" ::: "memory");
  uint32_t value;
  asm volatile("lw %[value], (%[address])\n\t"
               "and x0, x0, %[value]"
               : [value] "=r"(value)
               : [address] "r"(synchronizationWord)
               : "memory");
  return value;
}

FORCE_INLINE void
storeSynchronizationWord(volatile uint32_t tt_l1_ptr *synchronizationWord,
                         uint32_t value) {
  // The dependent load waits until the store is visible to the other RISCs.
  asm volatile("sw %[value], (%[address])\n\t"
               "lw %[value], (%[address])\n\t"
               "and x0, x0, %[value]"
               : [value] "+r"(value)
               : [address] "r"(synchronizationWord)
               : "memory");
}

FORCE_INLINE bool
participantsHaveState(volatile uint32_t tt_l1_ptr *synchronizationState,
                      uint32_t state) {
  for (uint32_t participant = 0; participant < participantCount;
       ++participant) {
    if (loadSynchronizationWord(&synchronizationState[participant]) != state) {
      return false;
    }
  }
  return true;
}

// The completion marker is ordered after prior engine work. Its GPR readback
// prevents arrival before retirement; TMP0 is temporary across LLK calls.
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

FORCE_INLINE void enter(volatile uint32_t tt_l1_ptr *synchronizationState) {
#if defined(TTL_DFB_RECONFIGURATION_DM0)
  constexpr uint32_t arrivalWord = dm0StateWord;
#elif defined(TTL_DFB_RECONFIGURATION_UNPACK)
  constexpr uint32_t arrivalWord = unpackStateWord;
#elif defined(TTL_DFB_RECONFIGURATION_PACK)
  constexpr uint32_t arrivalWord = packStateWord;
#endif
#if defined(TTL_DFB_RECONFIGURATION_DM0)
  noc_async_full_barrier();
#elif defined(TTL_DFB_RECONFIGURATION_UNPACK) ||                               \
    defined(TTL_DFB_RECONFIGURATION_PACK)
  drainComputeEngine();
#endif
#if defined(TTL_DFB_RECONFIGURATION_DM0) ||                                    \
    defined(TTL_DFB_RECONFIGURATION_UNPACK) ||                                 \
    defined(TTL_DFB_RECONFIGURATION_PACK)
  storeSynchronizationWord(&synchronizationState[arrivalWord], entryComplete);
  while (loadSynchronizationWord(&synchronizationState[releaseWord]) !=
         entryComplete) {
  }
#elif defined(TTL_DFB_RECONFIGURATION_DM1)
  noc_async_full_barrier();
  while (!participantsHaveState(synchronizationState, entryComplete)) {
  }
  storeSynchronizationWord(&synchronizationState[releaseWord], entryComplete);
#endif
}

// DM1 cannot begin next-epoch work until every other RISC has completed its
// interface updates.
FORCE_INLINE void exit(volatile uint32_t tt_l1_ptr *synchronizationState) {
#if defined(TTL_DFB_RECONFIGURATION_DM0)
  constexpr uint32_t arrivalWord = dm0StateWord;
#elif defined(TTL_DFB_RECONFIGURATION_UNPACK)
  constexpr uint32_t arrivalWord = unpackStateWord;
#elif defined(TTL_DFB_RECONFIGURATION_PACK)
  constexpr uint32_t arrivalWord = packStateWord;
#endif
#if defined(TTL_DFB_RECONFIGURATION_DM0) ||                                    \
    defined(TTL_DFB_RECONFIGURATION_UNPACK) ||                                 \
    defined(TTL_DFB_RECONFIGURATION_PACK)
  storeSynchronizationWord(&synchronizationState[arrivalWord], exitComplete);
  while (loadSynchronizationWord(&synchronizationState[releaseWord]) !=
         exitComplete) {
  }
  storeSynchronizationWord(&synchronizationState[arrivalWord], 0);
  while (loadSynchronizationWord(&synchronizationState[releaseWord]) != 0) {
  }
#elif defined(TTL_DFB_RECONFIGURATION_DM1)
  while (!participantsHaveState(synchronizationState, exitComplete)) {
  }
  storeSynchronizationWord(&synchronizationState[releaseWord], exitComplete);
  while (!participantsHaveState(synchronizationState, 0)) {
  }
  storeSynchronizationWord(&synchronizationState[releaseWord], 0);
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
    defined(TTL_DFB_RECONFIGURATION_PACK)
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

  auto *configuration =
      reinterpret_cast<uint32_t tt_l1_ptr *>(configurationAddress);
  auto *synchronizationState = reinterpret_cast<volatile uint32_t tt_l1_ptr *>(
      &configuration[dfb_reconfiguration_detail::synchronizationWord]);
  dfb_reconfiguration_detail::enter(synchronizationState);
  dfb_reconfiguration_detail::applyMask<updateReadPointer, updateWritePointer,
                                        updateWriteTilePointer,
                                        resetStreamCounters>(
      configuration, configuration[dfb_reconfiguration_detail::lowMaskWord], 0);
  dfb_reconfiguration_detail::applyMask<updateReadPointer, updateWritePointer,
                                        updateWriteTilePointer,
                                        resetStreamCounters>(
      configuration, configuration[dfb_reconfiguration_detail::highMaskWord],
      32);
  dfb_reconfiguration_detail::exit(synchronizationState);
#else
  (void)configurationAddress;
#endif
}

#undef TTL_DFB_RECONFIGURATION_DM0
#undef TTL_DFB_RECONFIGURATION_DM1
#undef TTL_DFB_RECONFIGURATION_UNPACK
#undef TTL_DFB_RECONFIGURATION_PACK

} // namespace experimental

#endif
