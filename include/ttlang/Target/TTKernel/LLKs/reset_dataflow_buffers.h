// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC) ||               \
    defined(COMPILE_FOR_DM)
#include "api/dataflow/dataflow_api.h"
#elif defined(UCK_CHLKC_UNPACK) || defined(UCK_CHLKC_MATH) ||                  \
    defined(UCK_CHLKC_PACK) || defined(TRISC_UNPACK) || defined(TRISC_MATH) || \
    defined(TRISC_PACK)
#include "api/compute/common.h"
#endif

#if !defined(ARCH_BLACKHOLE)
#error "reset_dataflow_buffers currently supports Blackhole only"
#endif

namespace ttlang {
namespace detail {

static constexpr uint32_t kConfigWords = 11;
static constexpr uint32_t kCompletionMarker = 0xd1fb;
static constexpr uint32_t kParticipantCount = 4;
static constexpr uint32_t kEntryArrived = 1;
static constexpr uint32_t kEntryReleased = 2;
static constexpr uint32_t kExitArrived = 3;
static constexpr uint32_t kExitReleased = 4;

#if defined(COMPILE_FOR_BRISC) ||                                              \
    (defined(COMPILE_FOR_DM) && COMPILE_FOR_DM == 0)
#define TTLANG_DFB_DM0
#endif

#if defined(COMPILE_FOR_NCRISC) ||                                             \
    (defined(COMPILE_FOR_DM) && COMPILE_FOR_DM == 1)
#define TTLANG_DFB_DM1
#endif

#if defined(UCK_CHLKC_UNPACK) || defined(TRISC_UNPACK)
#define TTLANG_DFB_UNPACK
#endif

#if defined(UCK_CHLKC_MATH) || defined(TRISC_MATH)
#define TTLANG_DFB_MATH
#endif

#if defined(UCK_CHLKC_PACK) || defined(TRISC_PACK)
#define TTLANG_DFB_PACK
#endif

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

template <uint32_t BaseSemaphore>
FORCE_INLINE volatile uint32_t tt_l1_ptr *
synchronizationWord(uint32_t participant) {
  uint32_t semaphore = BaseSemaphore + participant;
#if defined(TTLANG_DFB_UNPACK) || defined(TTLANG_DFB_MATH) ||                  \
    defined(TTLANG_DFB_PACK)
  auto *mailboxes = reinterpret_cast<tt_l1_ptr mailboxes_t *>(MEM_MAILBOX_BASE);
  auto &kernelConfig =
      mailboxes->launch[mailboxes->launch_msg_rd_ptr].kernel_config;
  uintptr_t address =
      kernelConfig.kernel_config_base[static_cast<int>(
          ProgrammableCoreType::TENSIX)] +
      kernelConfig.sem_offset[static_cast<int>(ProgrammableCoreType::TENSIX)] +
      semaphore * L1_ALIGNMENT;
#else
  uintptr_t address = get_semaphore(semaphore);
#endif
  return reinterpret_cast<volatile uint32_t tt_l1_ptr *>(address);
}

template <uint32_t BaseSemaphore>
FORCE_INLINE bool participantsHaveState(uint32_t state) {
  for (uint32_t participant = 0; participant < kParticipantCount;
       ++participant) {
    if (loadSynchronizationWord(
            synchronizationWord<BaseSemaphore>(participant)) != state) {
      return false;
    }
  }
  return true;
}

template <uint32_t BaseSemaphore>
FORCE_INLINE void setParticipantStates(uint32_t state) {
  for (uint32_t participant = 0; participant < kParticipantCount;
       ++participant) {
    storeSynchronizationWord(synchronizationWord<BaseSemaphore>(participant),
                             state);
  }
}

static constexpr uint32_t participantWord() {
#if defined(TTLANG_DFB_DM0)
  return 0;
#elif defined(TTLANG_DFB_UNPACK)
  return 1;
#elif defined(TTLANG_DFB_MATH)
  return 2;
#elif defined(TTLANG_DFB_PACK)
  return 3;
#else
  return 0;
#endif
}

FORCE_INLINE void drain() {
#if defined(TTLANG_DFB_DM0) || defined(TTLANG_DFB_DM1)
  noc_async_full_barrier();
#elif defined(TTLANG_DFB_UNPACK)
  TTI_STALLWAIT(p_stall::STALL_TDMA, p_stall::UNPACK);
  TTI_SETDMAREG(0, kCompletionMarker, 0, LO_16(p_gpr_unpack::TMP0));
  sync_regfile_write(p_gpr_unpack::TMP0);
#elif defined(TTLANG_DFB_MATH)
  TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::MATH | p_stall::WAIT_SFPU);
  tensix_sync();
#elif defined(TTLANG_DFB_PACK)
  TTI_STALLWAIT(p_stall::STALL_TDMA, p_stall::PACK);
  TTI_SETDMAREG(0, kCompletionMarker, 0, LO_16(p_gpr_pack::TMP0));
  sync_regfile_write(p_gpr_pack::TMP0);
#endif
}

template <uint32_t BaseSemaphore>
FORCE_INLINE void enter() {
#if defined(TTLANG_DFB_DM0) || defined(TTLANG_DFB_UNPACK) ||                   \
    defined(TTLANG_DFB_MATH) || defined(TTLANG_DFB_PACK)
  constexpr uint32_t word = participantWord();
  auto *state = synchronizationWord<BaseSemaphore>(word);
  storeSynchronizationWord(state, kEntryArrived);
  while (loadSynchronizationWord(state) != kEntryReleased) {
  }
#elif defined(TTLANG_DFB_DM1)
  while (!participantsHaveState<BaseSemaphore>(kEntryArrived)) {
  }
  setParticipantStates<BaseSemaphore>(kEntryReleased);
#endif
}

template <uint32_t BaseSemaphore>
FORCE_INLINE void exit() {
#if defined(TTLANG_DFB_DM0) || defined(TTLANG_DFB_UNPACK) ||                   \
    defined(TTLANG_DFB_MATH) || defined(TTLANG_DFB_PACK)
  constexpr uint32_t word = participantWord();
  auto *state = synchronizationWord<BaseSemaphore>(word);
  storeSynchronizationWord(state, kExitArrived);
  while (loadSynchronizationWord(state) != kExitReleased) {
  }
#elif defined(TTLANG_DFB_DM1)
  while (!participantsHaveState<BaseSemaphore>(kExitArrived)) {
  }
  setParticipantStates<BaseSemaphore>(kExitReleased);
#endif
}

template <uint32_t Id, uint32_t TotalBytes, uint32_t NumPages,
          uint32_t PageBytes>
FORCE_INLINE void reconfigureInterface() {
#if defined(TTLANG_DFB_DM0) || defined(TTLANG_DFB_DM1) ||                      \
    defined(TTLANG_DFB_UNPACK) || defined(TTLANG_DFB_PACK)
  LocalCBInterface &iface = get_local_cb_interface(Id);
  const uint32_t base = iface.fifo_limit - iface.fifo_size;
  const uint32_t size = TotalBytes >> cb_addr_shift;
  const uint32_t pageSize = PageBytes >> cb_addr_shift;

#if defined(TTLANG_DFB_DM0) || defined(TTLANG_DFB_DM1)
  iface.fifo_rd_ptr = base;
  iface.fifo_wr_ptr = base;
  iface.fifo_num_pages = NumPages;
#elif defined(TTLANG_DFB_UNPACK)
  iface.fifo_rd_ptr = base;
#elif defined(TTLANG_DFB_PACK)
  iface.fifo_wr_ptr = base;
  iface.fifo_wr_tile_ptr = 0;
  iface.fifo_num_pages = NumPages;
#endif
  iface.fifo_size = size;
  iface.fifo_limit = base + size;
  iface.fifo_page_size = pageSize;
  iface.tiles_acked_received_init = 0;

#if defined(TTLANG_DFB_DM1)
  *get_cb_tiles_received_ptr(Id) = 0;
  *get_cb_tiles_acked_ptr(Id) = 0;
#endif
#endif
}

template <uint32_t Id, uint32_t PageBytes, uint32_t L1Format,
          uint32_t TileHeight, uint32_t TileWidth, uint32_t FaceHeight,
          uint32_t NumFaces, uint32_t UnpackDstFormat, uint32_t PackSrcFormat>
FORCE_INLINE void reconfigureFormat() {
#if defined(TTLANG_DFB_DM0) || defined(TTLANG_DFB_DM1) ||                      \
    defined(TTLANG_DFB_UNPACK) || defined(TTLANG_DFB_MATH)
  unpack_src_format[Id] = L1Format;
  unpack_dst_format[Id] = UnpackDstFormat;
  unpack_tile_num_faces[Id] = NumFaces;
  unpack_partial_face[Id] = TileHeight < 32;
  unpack_tile_face_r_dim[Id] = FaceHeight;
  unpack_narrow_tile[Id] = TileWidth < 32;
  unpack_tile_r_dim[Id] = TileHeight;
  unpack_tile_c_dim[Id] = TileWidth;
  unpack_tile_size[Id] = PageBytes;
  unpack_num_faces_c_dim[Id] =
      NumFaces < TileWidth / 16 ? NumFaces : TileWidth / 16;
  unpack_num_faces_r_dim[Id] = NumFaces / unpack_num_faces_c_dim[Id];
#endif

#if defined(TTLANG_DFB_DM0) || defined(TTLANG_DFB_DM1) ||                      \
    defined(TTLANG_DFB_PACK)
  pack_src_format[Id] = PackSrcFormat;
  pack_dst_format[Id] = L1Format;
#if defined(TTLANG_DFB_PACK)
  unpack_src_format[Id] = L1Format;
#endif
  pack_tile_num_faces[Id] = NumFaces;
  pack_partial_face[Id] = TileHeight < 32;
  pack_tile_face_r_dim[Id] = FaceHeight;
  pack_narrow_tile[Id] = TileWidth < 32;
  pack_tile_r_dim[Id] = TileHeight;
  pack_tile_c_dim[Id] = TileWidth;
  pack_tile_size[Id] = PageBytes;
  pack_num_faces_c_dim[Id] =
      NumFaces < TileWidth / 16 ? NumFaces : TileWidth / 16;
  pack_num_faces_r_dim[Id] = NumFaces / pack_num_faces_c_dim[Id];
#endif
}

template <uint32_t... Config>
struct ApplyConfigurations;

template <>
struct ApplyConfigurations<> {
  static FORCE_INLINE void run() {}
};

template <uint32_t Id, uint32_t TotalBytes, uint32_t NumPages,
          uint32_t PageBytes, uint32_t L1Format, uint32_t TileHeight,
          uint32_t TileWidth, uint32_t FaceHeight, uint32_t NumFaces,
          uint32_t UnpackDstFormat, uint32_t PackSrcFormat,
          uint32_t... Remaining>
struct ApplyConfigurations<Id, TotalBytes, NumPages, PageBytes, L1Format,
                           TileHeight, TileWidth, FaceHeight, NumFaces,
                           UnpackDstFormat, PackSrcFormat, Remaining...> {
  static FORCE_INLINE void run() {
    reconfigureInterface<Id, TotalBytes, NumPages, PageBytes>();
    reconfigureFormat<Id, PageBytes, L1Format, TileHeight, TileWidth,
                      FaceHeight, NumFaces, UnpackDstFormat, PackSrcFormat>();
    ApplyConfigurations<Remaining...>::run();
  }
};

} // namespace detail

template <uint32_t BaseSemaphore, uint32_t RecordCount, uint32_t... Config>
FORCE_INLINE void reset_dataflow_buffers() {
  static_assert(sizeof...(Config) == RecordCount * detail::kConfigWords);

  detail::drain();
  detail::enter<BaseSemaphore>();
  detail::ApplyConfigurations<Config...>::run();
  asm volatile("" ::: "memory");
  detail::exit<BaseSemaphore>();
}

} // namespace ttlang

#undef TTLANG_DFB_DM0
#undef TTLANG_DFB_DM1
#undef TTLANG_DFB_UNPACK
#undef TTLANG_DFB_MATH
#undef TTLANG_DFB_PACK
