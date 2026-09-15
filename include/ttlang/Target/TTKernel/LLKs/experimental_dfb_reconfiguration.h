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

// Each core stores 64 four-word interface records, two active masks, three
// arrival words, one release word, and two padding words in shared L1.
constexpr uint32_t lowMaskWord = 256;
constexpr uint32_t highMaskWord = 257;
constexpr uint32_t synchronizationWord = 258;
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
      uint32_t fifoAddress = configuredAddress == preserveFifoAddress
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

template <uint32_t... descriptorWords>
struct ApplyDescriptors;

template <>
struct ApplyDescriptors<> {
  static FORCE_INLINE void run() {}
};

template <uint32_t dfbIndex, uint32_t pageBytes, uint32_t l1Format,
          uint32_t tileHeight, uint32_t tileWidth, uint32_t faceHeight,
          uint32_t numFaces, uint32_t unpackDstFormat, uint32_t packSrcFormat,
          uint32_t... remainingWords>
struct ApplyDescriptors<dfbIndex, pageBytes, l1Format, tileHeight, tileWidth,
                        faceHeight, numFaces, unpackDstFormat, packSrcFormat,
                        remainingWords...> {
  static FORCE_INLINE void run() {
    applyDescriptor<dfbIndex, pageBytes, l1Format, tileHeight, tileWidth,
                    faceHeight, numFaces, unpackDstFormat, packSrcFormat>();
    ApplyDescriptors<remainingWords...>::run();
  }
};
#endif

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
  dfb_reconfiguration_detail::enter<false>(synchronizationState);
  dfb_reconfiguration_detail::applyMask<updateReadPointer, updateWritePointer,
                                        updateWriteTilePointer,
                                        resetStreamCounters>(
      configuration, configuration[dfb_reconfiguration_detail::lowMaskWord], 0);
  dfb_reconfiguration_detail::applyMask<updateReadPointer, updateWritePointer,
                                        updateWriteTilePointer,
                                        resetStreamCounters>(
      configuration, configuration[dfb_reconfiguration_detail::highMaskWord],
      32);
  dfb_reconfiguration_detail::exit<false>(synchronizationState);
#else
  (void)configurationAddress;
#endif
}

#if defined(TTLANG_RUNTIME_DFB_RECONFIGURATION)
template <uint32_t recordCount, uint32_t... descriptorWords>
FORCE_INLINE void reconfigure_dfb_descriptors(uint32_t configurationAddress) {
  static_assert(sizeof...(descriptorWords) == recordCount * 9);
#if defined(TTL_DFB_RECONFIGURATION_DM1) ||                                    \
    defined(TTL_DFB_RECONFIGURATION_DM0) ||                                    \
    defined(TTL_DFB_RECONFIGURATION_UNPACK) ||                                 \
    defined(TTL_DFB_RECONFIGURATION_MATH) ||                                   \
    defined(TTL_DFB_RECONFIGURATION_PACK)
  auto *configuration =
      reinterpret_cast<uint32_t tt_l1_ptr *>(configurationAddress);
  auto *synchronizationState = reinterpret_cast<volatile uint32_t tt_l1_ptr *>(
      &configuration[dfb_reconfiguration_detail::synchronizationWord]);
  dfb_reconfiguration_detail::enter<true>(synchronizationState);
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
  dfb_reconfiguration_detail::ApplyDescriptors<descriptorWords...>::run();
  asm volatile("" ::: "memory");
  dfb_reconfiguration_detail::exit<true>(synchronizationState);
#else
  (void)configurationAddress;
#endif
}
#endif

#undef TTL_DFB_RECONFIGURATION_DM0
#undef TTL_DFB_RECONFIGURATION_DM1
#undef TTL_DFB_RECONFIGURATION_UNPACK
#undef TTL_DFB_RECONFIGURATION_MATH
#undef TTL_DFB_RECONFIGURATION_PACK

} // namespace experimental

#endif
