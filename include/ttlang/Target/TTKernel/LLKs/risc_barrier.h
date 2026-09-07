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
#error "risc_barrier currently supports Blackhole only"
#endif

#if defined(COMPILE_FOR_BRISC) ||                                              \
    (defined(COMPILE_FOR_DM) && COMPILE_FOR_DM == 0)
#define TTLANG_RISC_BARRIER_DM0
#endif

#if defined(COMPILE_FOR_NCRISC) ||                                             \
    (defined(COMPILE_FOR_DM) && COMPILE_FOR_DM == 1)
#define TTLANG_RISC_BARRIER_DM1
#endif

#if defined(UCK_CHLKC_UNPACK) || defined(TRISC_UNPACK)
#define TTLANG_RISC_BARRIER_UNPACK
#endif

#if defined(UCK_CHLKC_MATH) || defined(TRISC_MATH)
#define TTLANG_RISC_BARRIER_MATH
#endif

#if defined(UCK_CHLKC_PACK) || defined(TRISC_PACK)
#define TTLANG_RISC_BARRIER_PACK
#endif

namespace ttlang::detail {

static constexpr uint32_t kRiscBarrierParticipantCount = 4;
static constexpr uint32_t kRiscBarrierEntryArrived = 1;
static constexpr uint32_t kRiscBarrierEntryReleased = 2;
static constexpr uint32_t kRiscBarrierExitArrived = 3;
static constexpr uint32_t kRiscBarrierExitReleased = 4;

FORCE_INLINE uint32_t
loadSynchronizationWord(volatile uint32_t tt_l1_ptr *synchronizationWord) {
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
  asm volatile("sw %[value], (%[address])\n\t"
               "lw %[value], (%[address])\n\t"
               "and x0, x0, %[value]"
               : [value] "+r"(value)
               : [address] "r"(synchronizationWord)
               : "memory");
}

FORCE_INLINE volatile uint32_t tt_l1_ptr *
riscBarrierSynchronizationWord(uint32_t participant, uint32_t word0,
                               uint32_t word1, uint32_t word2,
                               uint32_t word3) {
  uintptr_t address = word0;
  if (participant == 1) {
    address = word1;
  } else if (participant == 2) {
    address = word2;
  } else if (participant == 3) {
    address = word3;
  }
  return reinterpret_cast<volatile uint32_t tt_l1_ptr *>(address);
}

FORCE_INLINE bool riscBarrierParticipantsHaveState(uint32_t state,
                                                   uint32_t word0,
                                                   uint32_t word1,
                                                   uint32_t word2,
                                                   uint32_t word3) {
  for (uint32_t participant = 0;
       participant < kRiscBarrierParticipantCount; ++participant) {
    if (loadSynchronizationWord(riscBarrierSynchronizationWord(
            participant, word0, word1, word2, word3)) != state) {
      return false;
    }
  }
  return true;
}

FORCE_INLINE void riscBarrierSetParticipantStates(uint32_t state,
                                                  uint32_t word0,
                                                  uint32_t word1,
                                                  uint32_t word2,
                                                  uint32_t word3) {
  for (uint32_t participant = 0;
       participant < kRiscBarrierParticipantCount; ++participant) {
    storeSynchronizationWord(riscBarrierSynchronizationWord(
                                 participant, word0, word1, word2, word3),
                             state);
  }
}

static constexpr uint32_t riscBarrierParticipantWord() {
#if defined(TTLANG_RISC_BARRIER_DM0)
  return 0;
#elif defined(TTLANG_RISC_BARRIER_UNPACK)
  return 1;
#elif defined(TTLANG_RISC_BARRIER_MATH)
  return 2;
#elif defined(TTLANG_RISC_BARRIER_PACK)
  return 3;
#else
  return 0;
#endif
}

FORCE_INLINE void riscBarrierEnter(uint32_t word0, uint32_t word1,
                                   uint32_t word2, uint32_t word3) {
#if defined(TTLANG_RISC_BARRIER_DM0) ||                                       \
    defined(TTLANG_RISC_BARRIER_UNPACK) ||                                    \
    defined(TTLANG_RISC_BARRIER_MATH) ||                                      \
    defined(TTLANG_RISC_BARRIER_PACK)
  constexpr uint32_t word = riscBarrierParticipantWord();
  auto *state =
      riscBarrierSynchronizationWord(word, word0, word1, word2, word3);
  storeSynchronizationWord(state, kRiscBarrierEntryArrived);
  while (loadSynchronizationWord(state) != kRiscBarrierEntryReleased) {
  }
#elif defined(TTLANG_RISC_BARRIER_DM1)
  while (!riscBarrierParticipantsHaveState(kRiscBarrierEntryArrived, word0,
                                           word1, word2, word3)) {
  }
  riscBarrierSetParticipantStates(kRiscBarrierEntryReleased, word0, word1,
                                  word2, word3);
#endif
}

FORCE_INLINE void riscBarrierExit(uint32_t word0, uint32_t word1,
                                  uint32_t word2, uint32_t word3) {
#if defined(TTLANG_RISC_BARRIER_DM0) ||                                       \
    defined(TTLANG_RISC_BARRIER_UNPACK) ||                                    \
    defined(TTLANG_RISC_BARRIER_MATH) ||                                      \
    defined(TTLANG_RISC_BARRIER_PACK)
  constexpr uint32_t word = riscBarrierParticipantWord();
  auto *state =
      riscBarrierSynchronizationWord(word, word0, word1, word2, word3);
  storeSynchronizationWord(state, kRiscBarrierExitArrived);
  while (loadSynchronizationWord(state) != kRiscBarrierExitReleased) {
  }
#elif defined(TTLANG_RISC_BARRIER_DM1)
  while (!riscBarrierParticipantsHaveState(kRiscBarrierExitArrived, word0,
                                           word1, word2, word3)) {
  }
  riscBarrierSetParticipantStates(kRiscBarrierExitReleased, word0, word1,
                                  word2, word3);
#endif
}

} // namespace ttlang::detail

#undef TTLANG_RISC_BARRIER_DM0
#undef TTLANG_RISC_BARRIER_DM1
#undef TTLANG_RISC_BARRIER_UNPACK
#undef TTLANG_RISC_BARRIER_MATH
#undef TTLANG_RISC_BARRIER_PACK
