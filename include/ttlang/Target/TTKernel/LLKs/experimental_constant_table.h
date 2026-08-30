// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_CONSTANT_TABLE_H
#define TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_CONSTANT_TABLE_H

#include <cstddef>
#include <cstdint>

namespace experimental {

template <std::size_t BitsPerValue>
FORCE_INLINE std::size_t constant_table_lookup_word(std::size_t index,
                                                    std::uint32_t packed_word) {
  static_assert(BitsPerValue > 0 && BitsPerValue <= 32);
  constexpr std::uint32_t value_mask = ~std::uint32_t{0} >> (32 - BitsPerValue);
  return (packed_word >> (index * BitsPerValue)) & value_mask;
}

// Wormhole NCRISC shares the complex lookup to fit its 16 KiB instruction
// memory.
#if defined(ARCH_WORMHOLE) && defined(COMPILE_FOR_NCRISC)
#define TTLANG_CONSTANT_TABLE_INLINE __attribute__((noinline)) inline
#else
#define TTLANG_CONSTANT_TABLE_INLINE FORCE_INLINE
#endif

template <std::size_t BitsPerValue>
TTLANG_CONSTANT_TABLE_INLINE std::size_t
constant_table_lookup(std::size_t index, const std::uint64_t *packed_table) {
  static_assert(BitsPerValue > 0 && BitsPerValue < 64);
  constexpr std::uint64_t value_mask = (std::uint64_t{1} << BitsPerValue) - 1;
  const std::size_t bit_offset = index * BitsPerValue;
  const std::size_t word_index = bit_offset / 64;
  const std::size_t offset_in_word = bit_offset % 64;
  std::uint64_t value = packed_table[word_index] >> offset_in_word;
  if (offset_in_word + BitsPerValue > 64) {
    value |= packed_table[word_index + 1] << (64 - offset_in_word);
  }
  return value & value_mask;
}

#undef TTLANG_CONSTANT_TABLE_INLINE

} // namespace experimental

#endif
