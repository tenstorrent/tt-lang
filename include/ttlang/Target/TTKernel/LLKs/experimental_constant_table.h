// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_CONSTANT_TABLE_H
#define TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_CONSTANT_TABLE_H

#include <cstddef>
#include <cstdint>

namespace experimental {

template <std::size_t BitsPerValue>
FORCE_INLINE std::size_t
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

} // namespace experimental

#endif
