// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/compute_kernel_api.h"
#else
#include "api/dataflow/circular_buffer.h"
#endif

#if defined(COMPILE_FOR_TRISC)
template <uint32_t WordCount>
inline void raw_address_capture_compute(uint32_t tensor_address,
                                        uint32_t output_address) {
#ifdef TRISC_MATH
  auto *output =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(output_address);
  for (uint32_t word_index = 0; word_index < WordCount; ++word_index) {
    output[word_index] = tensor_address;
  }
#endif
}
#else
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
#endif

template <uint32_t WordCount>
inline void raw_address_capture_unified(uint32_t tensor_address,
                                        uint32_t output_address) {
#if defined(COMPILE_FOR_TRISC)
  raw_address_capture_compute<WordCount>(tensor_address, output_address);
#else
  (void)tensor_address;
  (void)output_address;
#endif
}
