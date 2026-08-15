// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"

inline void write_operation_runtime_value(uint32_t outputDfb) {
  uint32_t outputValue = get_arg_val<uint32_t>(0);
  const uint32_t semaphoreId = get_arg_val<uint32_t>(1);
  const uint32_t generation = get_arg_val<uint32_t>(2);
  auto *semaphore = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
      get_semaphore(semaphoreId));

  noc_semaphore_set(semaphore, generation);
  noc_semaphore_wait(semaphore, generation);

#if defined(OUTPUT_BF16)
  constexpr uint32_t wordCount = 32 * 32 / 2;
#if defined(OUTPUT_ALTERNATE)
  outputValue = get_absolute_logical_x() == 0 ? 0x40E0 : 0x4100;
#endif
  outputValue |= outputValue << 16;
#else
  constexpr uint32_t wordCount = 32 * 32;
#if defined(OUTPUT_ALTERNATE)
  outputValue = get_absolute_logical_x() == 0 ? 0x40E00000 : 0x41000000;
#endif
#endif

  CircularBuffer output(outputDfb);
  output.reserve_back(1);
  auto *outputWords =
      reinterpret_cast<volatile tt_l1_ptr uint32_t *>(output.get_write_ptr());
  for (uint32_t wordIndex = 0; wordIndex < wordCount; ++wordIndex) {
    outputWords[wordIndex] = outputValue;
  }
  output.push_back(1);
}
