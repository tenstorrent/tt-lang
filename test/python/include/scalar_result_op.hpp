// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

template <std::uint32_t BitWidth>
struct ScalarResult;

template <>
struct ScalarResult<32> {
  using Type = std::int32_t;
};

template <>
struct ScalarResult<64> {
  using Type = std::int64_t;
};

template <std::uint32_t BitWidth>
inline typename ScalarResult<BitWidth>::Type scalar_result() {
  if constexpr (BitWidth == 64) {
    return std::int64_t{1} << 40;
  }
  return 1;
}

template <std::uint32_t BitWidth, typename Coordinate>
inline typename ScalarResult<BitWidth>::Type
scalar_result_from_coordinate(Coordinate) {
  return 1;
}

template <bool Value>
inline std::int32_t scalar_predicate() {
  return Value ? 1 : 0;
}
