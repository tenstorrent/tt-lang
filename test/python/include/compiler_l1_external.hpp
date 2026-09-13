// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

template <typename Source, typename Destination>
static inline void compiler_l1_copy_dfb() {
  static_assert(Source::pages_per_block == Destination::pages_per_block);
  static_assert(Source::page_size_bytes == Destination::page_size_bytes);
  Source source = Source::bind();
  Destination destination = Destination::bind();
  destination.reserve_back(Destination::pages_per_block);
  source.wait_front(Source::pages_per_block);
  auto *sourceWords = reinterpret_cast<volatile std::uint32_t tt_l1_ptr *>(
      source.get_read_ptr());
  auto *destinationWords = reinterpret_cast<volatile std::uint32_t tt_l1_ptr *>(
      destination.get_write_ptr());
  constexpr std::uint32_t wordCount =
      Source::pages_per_block * Source::page_size_bytes / sizeof(std::uint32_t);
  for (std::uint32_t wordIndex = 0; wordIndex < wordCount; ++wordIndex) {
    destinationWords[wordIndex] = sourceWords[wordIndex];
  }
  destination.push_back(Destination::pages_per_block);
  source.pop_front(Source::pages_per_block);
}
