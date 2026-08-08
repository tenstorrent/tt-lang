// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_api.h"

inline void write_operation_runtime_value(uint32_t outputAddress) {
  const uint32_t outputValue = get_arg_val<uint32_t>(0);
  const uint32_t semaphoreId = get_arg_val<uint32_t>(1);
  const uint32_t generation = get_arg_val<uint32_t>(2);
  auto *semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
      get_semaphore(semaphoreId));

  noc_semaphore_set(semaphore, generation);
  noc_semaphore_wait(semaphore, generation);

  auto *output = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(outputAddress);
#if defined(OUTPUT_BF16)
  constexpr uint32_t wordCount = 32 * 32 / 2;
  const uint32_t packedValue = outputValue | (outputValue << 16);
  for (uint32_t wordIndex = 0; wordIndex < wordCount; ++wordIndex) {
    output[wordIndex] = packedValue;
  }
#else
  constexpr uint32_t wordCount = 32 * 32;
  for (uint32_t wordIndex = 0; wordIndex < wordCount; ++wordIndex) {
    output[wordIndex] = outputValue;
  }
#endif
}
