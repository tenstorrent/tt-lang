// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#ifndef TTLANG_COMPILER_L1_TARGET_H
#define TTLANG_COMPILER_L1_TARGET_H
#include <cstdint>
namespace ttlang::l1::target {
#if !defined(ARCH_WORMHOLE) && !defined(ARCH_WORMHOLE_B0) &&                   \
    !defined(ARCH_BLACKHOLE)
#error "compiler-l1 requires Wormhole or Blackhole"
#endif
inline uint32_t load(uint32_t address) {
  asm volatile("fence" ::: "memory");
  uint32_t value;
  asm volatile("lw %[value], (%[address])\n\tand x0, x0, %[value]"
               : [value] "=r"(value)
               : [address] "r"(address)
               : "memory");
  return value;
}
inline void store(uint32_t address, uint32_t value) {
  asm volatile("sw %[value], (%[address])\n\tlw %[value], (%[address])\n\tand "
               "x0, x0, %[value]"
               : [value] "+r"(value)
               : [address] "r"(address)
               : "memory");
}
inline void complete() { noc_async_full_barrier(); }
} // namespace ttlang::l1::target
#endif
