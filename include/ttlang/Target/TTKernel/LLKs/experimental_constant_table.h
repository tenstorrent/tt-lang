// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_CONSTANT_TABLE_H
#define TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_CONSTANT_TABLE_H

#include <cstddef>

namespace experimental {

template <std::size_t... Values>
FORCE_INLINE std::size_t constant_table_lookup(std::size_t index) {
  static constexpr std::size_t table[] = {Values...};
  return table[index];
}

} // namespace experimental

#endif
