// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "compiler_l1_external.hpp"

namespace compiler_sram_test {
template <typename Descriptor>
struct IsCompilerSRAMDescriptor {
  static constexpr bool value = false;
};

template <uint32_t PageBytes, uint32_t PagesPerBlock, uint32_t BlockCount,
          uint32_t StorageCapacityPages, uint32_t StateOffset,
          uint32_t PayloadOffset, int32_t PayloadCommonArgIndex>
struct IsCompilerSRAMDescriptor<ttlang::l1::DFBDescriptor<
    PageBytes, PagesPerBlock, BlockCount, StorageCapacityPages, StateOffset,
    PayloadOffset, PayloadCommonArgIndex>> {
  static constexpr bool value = true;
};

template <typename Source, typename Destination>
static inline void copy_typed_dfb() {
  static_assert(IsCompilerSRAMDescriptor<Source>::value);
  static_assert(IsCompilerSRAMDescriptor<Destination>::value);
  compiler_l1_copy_dfb<Source, Destination>();
}
} // namespace compiler_sram_test
