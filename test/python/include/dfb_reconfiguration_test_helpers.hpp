// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#if defined(COMPILE_FOR_TRISC)
#include "api/compute/cb_api.h"
#include "api/compute/compute_kernel_api.h"
#elif defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#endif

template <typename OutputDFB>
inline void write_reconfiguration_runtime_value() {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
  uint32_t firstValue = get_arg_val<uint32_t>(0);
  uint32_t outputValue = firstValue;
  if (get_absolute_logical_x() != 0) {
    uint32_t secondValue = get_arg_val<uint32_t>(1);
    outputValue = firstValue == 0x4000 && secondValue == 0x4040 ? 0x4080 : 0;
  }
  outputValue |= outputValue << 16;

  CircularBuffer outputDfb(OutputDFB::index);
  outputDfb.reserve_back(1);
  auto *outputWords = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(
      outputDfb.get_write_ptr());
  constexpr uint32_t wordCount = 32 * 32 / 2;
  for (uint32_t wordIndex = 0; wordIndex < wordCount; ++wordIndex) {
    outputWords[wordIndex] = outputValue;
  }
  outputDfb.push_back(1);
#endif
}

template <typename OutputDFB>
inline void publish_unread_tile() {
#if defined(COMPILE_FOR_NCRISC) || defined(COMPILE_FOR_BRISC)
  CircularBuffer outputDfb(OutputDFB::index);
  outputDfb.reserve_back(1);
  outputDfb.push_back(1);
#endif
}

template <typename InputDFB>
inline void wait_without_pop() {
#if defined(COMPILE_FOR_TRISC)
  using namespace ckernel;
  cb_wait_front(InputDFB::index, 1);
#endif
}
