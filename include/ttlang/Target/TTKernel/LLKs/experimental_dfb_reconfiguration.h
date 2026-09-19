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

#if defined(UCK_CHLKC_MATH) || defined(TRISC_MATH)
#define TTL_DFB_RECONFIGURATION_MATH
#endif

#if defined(UCK_CHLKC_PACK) || defined(TRISC_PACK)
#define TTL_DFB_RECONFIGURATION_PACK
#endif

// Each boundary stores one address per reconfigured DFB, two active masks, and
// six synchronization words. Interface geometry is encoded in template data.
constexpr uint32_t activeMaskWordCount = 2;
constexpr uint32_t synchronizationWordCount = 6;
constexpr uint32_t recordWordCount = 12;
constexpr uint32_t specializedRecordWordCount = recordWordCount + 1;
constexpr uint32_t dm0StateWord = 0;
constexpr uint32_t unpackStateWord = 1;
constexpr uint32_t packStateWord = 2;
constexpr uint32_t releaseWord = 3;
constexpr uint32_t mathStateWord = 4;
constexpr uint32_t participantCount = 3;
constexpr uint32_t entryComplete = 1;
constexpr uint32_t exitComplete = 2;
constexpr uint32_t completionMarker = 0xD1FB;
constexpr uint32_t preserveFifoAddress = 0;

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

template <bool includeMath>
FORCE_INLINE bool
participantsHaveState(volatile uint32_t tt_l1_ptr *synchronizationState,
                      uint32_t state) {
  for (uint32_t participant = 0; participant < participantCount;
       ++participant) {
    if (loadSynchronizationWord(&synchronizationState[participant]) != state) {
      return false;
    }
  }
  if constexpr (includeMath) {
    return loadSynchronizationWord(&synchronizationState[mathStateWord]) ==
           state;
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
#elif defined(TTL_DFB_RECONFIGURATION_MATH)
  TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
  tensix_sync();
#endif
#if defined(TTL_DFB_RECONFIGURATION_UNPACK) ||                                 \
    defined(TTL_DFB_RECONFIGURATION_PACK)
  TTI_STALLWAIT(p_stall::STALL_TDMA, waitResources);
  TTI_SETDMAREG(0, completionMarker, 0, LO_16(completionGpr));
  sync_regfile_write(completionGpr);
#endif
}

template <bool includeMath>
FORCE_INLINE void enter(volatile uint32_t tt_l1_ptr *synchronizationState) {
#if defined(TTL_DFB_RECONFIGURATION_DM0)
  constexpr uint32_t arrivalWord = dm0StateWord;
#elif defined(TTL_DFB_RECONFIGURATION_UNPACK)
  constexpr uint32_t arrivalWord = unpackStateWord;
#elif defined(TTL_DFB_RECONFIGURATION_PACK)
  constexpr uint32_t arrivalWord = packStateWord;
#elif defined(TTL_DFB_RECONFIGURATION_MATH)
  constexpr uint32_t arrivalWord = mathStateWord;
#endif
#if defined(TTL_DFB_RECONFIGURATION_DM0)
  noc_async_full_barrier();
#elif defined(TTL_DFB_RECONFIGURATION_UNPACK) ||                               \
    defined(TTL_DFB_RECONFIGURATION_PACK)
  drainComputeEngine();
#elif defined(TTL_DFB_RECONFIGURATION_MATH)
  if constexpr (includeMath) {
    drainComputeEngine();
  }
#endif
#if defined(TTL_DFB_RECONFIGURATION_DM0) ||                                    \
    defined(TTL_DFB_RECONFIGURATION_UNPACK) ||                                 \
    defined(TTL_DFB_RECONFIGURATION_PACK)
  storeSynchronizationWord(&synchronizationState[arrivalWord], entryComplete);
  while (loadSynchronizationWord(&synchronizationState[releaseWord]) !=
         entryComplete) {
  }
#elif defined(TTL_DFB_RECONFIGURATION_MATH)
  if constexpr (includeMath) {
    storeSynchronizationWord(&synchronizationState[arrivalWord], entryComplete);
    while (loadSynchronizationWord(&synchronizationState[releaseWord]) !=
           entryComplete) {
    }
  }
#elif defined(TTL_DFB_RECONFIGURATION_DM1)
  noc_async_full_barrier();
  while (!participantsHaveState<includeMath>(synchronizationState,
                                             entryComplete)) {
  }
  storeSynchronizationWord(&synchronizationState[releaseWord], entryComplete);
#endif
}

// DM1 cannot begin next-epoch work until every other RISC has completed its
// interface updates.
template <bool includeMath>
FORCE_INLINE void exit(volatile uint32_t tt_l1_ptr *synchronizationState) {
#if defined(TTL_DFB_RECONFIGURATION_DM0)
  constexpr uint32_t arrivalWord = dm0StateWord;
#elif defined(TTL_DFB_RECONFIGURATION_UNPACK)
  constexpr uint32_t arrivalWord = unpackStateWord;
#elif defined(TTL_DFB_RECONFIGURATION_PACK)
  constexpr uint32_t arrivalWord = packStateWord;
#elif defined(TTL_DFB_RECONFIGURATION_MATH)
  constexpr uint32_t arrivalWord = mathStateWord;
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
#elif defined(TTL_DFB_RECONFIGURATION_MATH)
  if constexpr (includeMath) {
    storeSynchronizationWord(&synchronizationState[arrivalWord], exitComplete);
    while (loadSynchronizationWord(&synchronizationState[releaseWord]) !=
           exitComplete) {
    }
    storeSynchronizationWord(&synchronizationState[arrivalWord], 0);
    while (loadSynchronizationWord(&synchronizationState[releaseWord]) != 0) {
    }
  }
#elif defined(TTL_DFB_RECONFIGURATION_DM1)
  while (
      !participantsHaveState<includeMath>(synchronizationState, exitComplete)) {
  }
  storeSynchronizationWord(&synchronizationState[releaseWord], exitComplete);
  while (!participantsHaveState<includeMath>(synchronizationState, 0)) {
  }
  storeSynchronizationWord(&synchronizationState[releaseWord], 0);
#endif
}

template <bool updateReadPointer, bool updateWritePointer,
          bool updateWriteTilePointer, bool resetStreamCounters>
FORCE_INLINE void applyInterface(uint32_t dfbIndex, uint32_t fifoAddress,
                                 uint32_t fifoSize, uint32_t fifoNumPages,
                                 uint32_t fifoPageSize) {
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

#if defined(TTLANG_RUNTIME_DFB_RECONFIGURATION)
template <uint32_t dfbIndex, uint32_t pageBytes, uint32_t l1Format,
          uint32_t tileHeight, uint32_t tileWidth, uint32_t faceHeight,
          uint32_t numFaces, uint32_t unpackDstFormat, uint32_t packSrcFormat>
FORCE_INLINE void applyDescriptor() {
#if defined(TTL_DFB_RECONFIGURATION_DM0) ||                                    \
    defined(TTL_DFB_RECONFIGURATION_DM1) ||                                    \
    defined(TTL_DFB_RECONFIGURATION_UNPACK) ||                                 \
    defined(TTL_DFB_RECONFIGURATION_MATH)
  unpack_src_format[dfbIndex] = l1Format;
  unpack_dst_format[dfbIndex] = unpackDstFormat;
  unpack_tile_num_faces[dfbIndex] = numFaces;
  unpack_partial_face[dfbIndex] = tileHeight < 32;
  unpack_tile_face_r_dim[dfbIndex] = faceHeight;
  unpack_narrow_tile[dfbIndex] = tileWidth < 32;
  unpack_tile_r_dim[dfbIndex] = tileHeight;
  unpack_tile_c_dim[dfbIndex] = tileWidth;
  unpack_tile_size[dfbIndex] = pageBytes;
  unpack_num_faces_c_dim[dfbIndex] =
      numFaces < tileWidth / 16 ? numFaces : tileWidth / 16;
  unpack_num_faces_r_dim[dfbIndex] =
      numFaces / unpack_num_faces_c_dim[dfbIndex];
#endif

#if defined(TTL_DFB_RECONFIGURATION_DM0) ||                                    \
    defined(TTL_DFB_RECONFIGURATION_DM1) ||                                    \
    defined(TTL_DFB_RECONFIGURATION_PACK)
  pack_src_format[dfbIndex] = packSrcFormat;
  pack_dst_format[dfbIndex] = l1Format;
#if defined(TTL_DFB_RECONFIGURATION_PACK)
  unpack_src_format[dfbIndex] = l1Format;
#endif
  pack_tile_num_faces[dfbIndex] = numFaces;
  pack_partial_face[dfbIndex] = tileHeight < 32;
  pack_tile_face_r_dim[dfbIndex] = faceHeight;
  pack_narrow_tile[dfbIndex] = tileWidth < 32;
  pack_tile_r_dim[dfbIndex] = tileHeight;
  pack_tile_c_dim[dfbIndex] = tileWidth;
  pack_tile_size[dfbIndex] = pageBytes;
  pack_num_faces_c_dim[dfbIndex] =
      numFaces < tileWidth / 16 ? numFaces : tileWidth / 16;
  pack_num_faces_r_dim[dfbIndex] = numFaces / pack_num_faces_c_dim[dfbIndex];
#endif
}

#endif

template <bool updateReadPointer, bool updateWritePointer,
          bool updateWriteTilePointer, bool resetStreamCounters,
          uint32_t dfbIndex, uint32_t totalBytes, uint32_t numPages,
          uint32_t pageBytes, uint32_t updateDescriptor, uint32_t l1Format,
          uint32_t tileHeight, uint32_t tileWidth, uint32_t faceHeight,
          uint32_t numFaces, uint32_t unpackDstFormat, uint32_t packSrcFormat>
FORCE_INLINE void applyRecord(uint32_t configuredAddress) {
#if !defined(TTL_DFB_RECONFIGURATION_MATH)
  LocalCBInterface &interface = get_local_cb_interface(dfbIndex);
  uint32_t fifoAddress = configuredAddress == preserveFifoAddress
                             ? interface.fifo_limit - interface.fifo_size
                             : configuredAddress >> cb_addr_shift;
  applyInterface<updateReadPointer, updateWritePointer, updateWriteTilePointer,
                 resetStreamCounters>(dfbIndex, fifoAddress,
                                      totalBytes >> cb_addr_shift, numPages,
                                      pageBytes >> cb_addr_shift);
#endif
#if defined(TTLANG_RUNTIME_DFB_RECONFIGURATION)
  if constexpr (updateDescriptor != 0) {
    applyDescriptor<dfbIndex, pageBytes, l1Format, tileHeight, tileWidth,
                    faceHeight, numFaces, unpackDstFormat, packSrcFormat>();
  }
#else
  static_assert(updateDescriptor == 0,
                "descriptor updates require runtime descriptor storage");
#endif
}

template <uint32_t recordOffset, uint32_t... recordWords>
struct ApplyRecords;

template <uint32_t recordOffset>
struct ApplyRecords<recordOffset> {
  template <bool updateReadPointer, bool updateWritePointer,
            bool updateWriteTilePointer, bool resetStreamCounters>
  static FORCE_INLINE void run(uint32_t tt_l1_ptr *, uint32_t, uint32_t) {}
};

template <uint32_t recordOffset, uint32_t dfbIndex, uint32_t totalBytes,
          uint32_t numPages, uint32_t pageBytes, uint32_t updateDescriptor,
          uint32_t l1Format, uint32_t tileHeight, uint32_t tileWidth,
          uint32_t faceHeight, uint32_t numFaces, uint32_t unpackDstFormat,
          uint32_t packSrcFormat, uint32_t... remainingWords>
struct ApplyRecords<recordOffset, dfbIndex, totalBytes, numPages, pageBytes,
                    updateDescriptor, l1Format, tileHeight, tileWidth,
                    faceHeight, numFaces, unpackDstFormat, packSrcFormat,
                    remainingWords...> {
  template <bool updateReadPointer, bool updateWritePointer,
            bool updateWriteTilePointer, bool resetStreamCounters>
  static FORCE_INLINE void run(uint32_t tt_l1_ptr *configuration,
                               uint32_t lowMask, uint32_t highMask) {
    uint32_t activeMask = dfbIndex < 32 ? lowMask : highMask;
    if ((activeMask & (1U << (dfbIndex % 32))) != 0) {
      applyRecord<updateReadPointer, updateWritePointer,
                  updateWriteTilePointer, resetStreamCounters, dfbIndex,
                  totalBytes, numPages, pageBytes, updateDescriptor, l1Format,
                  tileHeight, tileWidth, faceHeight, numFaces, unpackDstFormat,
                  packSrcFormat>(configuration[recordOffset]);
    }
    ApplyRecords<recordOffset + 1, remainingWords...>::template run<
        updateReadPointer, updateWritePointer, updateWriteTilePointer,
        resetStreamCounters>(configuration, lowMask, highMask);
  }
};

template <uint32_t... recordWords>
struct ApplySpecializedRecords;

template <>
struct ApplySpecializedRecords<> {
  template <bool updateReadPointer, bool updateWritePointer,
            bool updateWriteTilePointer, bool resetStreamCounters>
  static FORCE_INLINE void run(uint32_t tt_l1_ptr *) {}
};

template <uint32_t runtimeAddressOffset, uint32_t dfbIndex,
          uint32_t totalBytes, uint32_t numPages, uint32_t pageBytes,
          uint32_t updateDescriptor, uint32_t l1Format, uint32_t tileHeight,
          uint32_t tileWidth, uint32_t faceHeight, uint32_t numFaces,
          uint32_t unpackDstFormat, uint32_t packSrcFormat,
          uint32_t... remainingWords>
struct ApplySpecializedRecords<
    runtimeAddressOffset, dfbIndex, totalBytes, numPages, pageBytes,
    updateDescriptor, l1Format, tileHeight, tileWidth, faceHeight, numFaces,
    unpackDstFormat, packSrcFormat, remainingWords...> {
  template <bool updateReadPointer, bool updateWritePointer,
            bool updateWriteTilePointer, bool resetStreamCounters>
  static FORCE_INLINE void run(uint32_t tt_l1_ptr *configuration) {
    applyRecord<updateReadPointer, updateWritePointer,
                updateWriteTilePointer, resetStreamCounters, dfbIndex,
                totalBytes, numPages, pageBytes, updateDescriptor, l1Format,
                tileHeight, tileWidth, faceHeight, numFaces, unpackDstFormat,
                packSrcFormat>(configuration[runtimeAddressOffset]);
    ApplySpecializedRecords<remainingWords...>::template run<
        updateReadPointer, updateWritePointer, updateWriteTilePointer,
        resetStreamCounters>(configuration);
  }
};

template <uint32_t runtimeRecordCount, uint32_t... recordWords>
struct ApplyMaskedRecords {
  template <bool updateReadPointer, bool updateWritePointer,
            bool updateWriteTilePointer, bool resetStreamCounters>
  static FORCE_INLINE void run(uint32_t tt_l1_ptr *configuration) {
    ApplyRecords<0, recordWords...>::template run<
        updateReadPointer, updateWritePointer, updateWriteTilePointer,
        resetStreamCounters>(configuration, configuration[runtimeRecordCount],
                             configuration[runtimeRecordCount + 1]);
  }
};

template <bool includeMath, uint32_t runtimeRecordCount,
          typename RecordApplication>
FORCE_INLINE void applyReconfiguration(uint32_t configurationAddress) {
#if defined(TTL_DFB_RECONFIGURATION_DM1) ||                                    \
    defined(TTL_DFB_RECONFIGURATION_DM0) ||                                    \
    defined(TTL_DFB_RECONFIGURATION_UNPACK) ||                                 \
    defined(TTL_DFB_RECONFIGURATION_MATH) ||                                   \
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
#elif defined(TTL_DFB_RECONFIGURATION_MATH)
  constexpr bool updateReadPointer = false;
  constexpr bool updateWritePointer = false;
  constexpr bool updateWriteTilePointer = false;
  constexpr bool resetStreamCounters = false;
#endif

  auto *configuration =
      reinterpret_cast<uint32_t tt_l1_ptr *>(configurationAddress);
  constexpr uint32_t synchronizationWord =
      runtimeRecordCount + activeMaskWordCount;
  auto *synchronizationState = reinterpret_cast<volatile uint32_t tt_l1_ptr *>(
      &configuration[synchronizationWord]);
  enter<includeMath>(synchronizationState);
  RecordApplication::template run<
      updateReadPointer, updateWritePointer, updateWriteTilePointer,
      resetStreamCounters>(configuration);
  asm volatile("" ::: "memory");
  exit<includeMath>(synchronizationState);
#else
  (void)configurationAddress;
#endif
}

} // namespace dfb_reconfiguration_detail

template <uint32_t recordCount, uint32_t... recordWords>
FORCE_INLINE void reconfigure_dfb_interfaces(uint32_t configurationAddress) {
  static_assert(sizeof...(recordWords) == recordCount *
                                              dfb_reconfiguration_detail::
                                                  recordWordCount);
  dfb_reconfiguration_detail::applyReconfiguration<
      false, recordCount,
      dfb_reconfiguration_detail::ApplyMaskedRecords<recordCount,
                                                     recordWords...>>(
      configurationAddress);
}

template <uint32_t runtimeRecordCount, uint32_t selectedRecordCount,
          uint32_t... recordWords>
FORCE_INLINE void
reconfigure_dfb_interfaces_specialized(uint32_t configurationAddress) {
  static_assert(sizeof...(recordWords) ==
                selectedRecordCount *
                    dfb_reconfiguration_detail::specializedRecordWordCount);
  dfb_reconfiguration_detail::applyReconfiguration<
      false, runtimeRecordCount,
      dfb_reconfiguration_detail::ApplySpecializedRecords<recordWords...>>(
      configurationAddress);
}

#if defined(TTLANG_RUNTIME_DFB_RECONFIGURATION)
template <uint32_t recordCount, uint32_t... recordWords>
FORCE_INLINE void reconfigure_dfb_descriptors(uint32_t configurationAddress) {
  static_assert(sizeof...(recordWords) == recordCount *
                                              dfb_reconfiguration_detail::
                                                  recordWordCount);
  dfb_reconfiguration_detail::applyReconfiguration<
      true, recordCount,
      dfb_reconfiguration_detail::ApplyMaskedRecords<recordCount,
                                                     recordWords...>>(
      configurationAddress);
}

template <uint32_t runtimeRecordCount, uint32_t selectedRecordCount,
          uint32_t... recordWords>
FORCE_INLINE void
reconfigure_dfb_descriptors_specialized(uint32_t configurationAddress) {
  static_assert(sizeof...(recordWords) ==
                selectedRecordCount *
                    dfb_reconfiguration_detail::specializedRecordWordCount);
  dfb_reconfiguration_detail::applyReconfiguration<
      true, runtimeRecordCount,
      dfb_reconfiguration_detail::ApplySpecializedRecords<recordWords...>>(
      configurationAddress);
}
#endif

#undef TTL_DFB_RECONFIGURATION_DM0
#undef TTL_DFB_RECONFIGURATION_DM1
#undef TTL_DFB_RECONFIGURATION_UNPACK
#undef TTL_DFB_RECONFIGURATION_MATH
#undef TTL_DFB_RECONFIGURATION_PACK

} // namespace experimental

#endif
