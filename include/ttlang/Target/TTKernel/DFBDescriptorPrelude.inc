// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

namespace ttlang {
template <uint32_t Index, uint32_t PagesPerBlock, uint32_t BlockCount,
          uint32_t PageSizeBytes>
struct DFBDescriptor {
  static constexpr uint32_t index = Index;
  static constexpr uint32_t pages_per_block = PagesPerBlock;
  static constexpr uint32_t block_count = BlockCount;
  static constexpr uint32_t page_size_bytes = PageSizeBytes;
};
} // namespace ttlang
