// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/circular_buffer.h"

template <typename Destination>
inline void raw_address_capture(uint32_t tensor_address) {
  static_assert(Destination::page_size_bytes % sizeof(uint32_t) == 0);
  cb_reserve_back(Destination::index, Destination::pages_per_block);
  auto *destination = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
      get_write_ptr(Destination::index));
  constexpr uint32_t wordCount = Destination::pages_per_block *
                                 Destination::page_size_bytes /
                                 sizeof(uint32_t);
  for (uint32_t wordIndex = 0; wordIndex < wordCount; ++wordIndex) {
    destination[wordIndex] = tensor_address;
  }
  cb_push_back(Destination::index, Destination::pages_per_block);
}
