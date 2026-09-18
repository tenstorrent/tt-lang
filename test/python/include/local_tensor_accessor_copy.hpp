// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/tensor/local_tensor_accessor.h"

template <std::uint32_t ByteCount>
inline void local_tensor_accessor_copy(
    const LocalTensorAccessor<std::uint8_t> &source,
    const LocalTensorAccessor<std::uint8_t> &destination) {
  for (std::uint32_t byteIndex = 0; byteIndex < ByteCount; ++byteIndex) {
    destination[byteIndex] = source[byteIndex];
  }
}
