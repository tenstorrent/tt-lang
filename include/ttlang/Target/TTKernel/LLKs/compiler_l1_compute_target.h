// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
#ifndef TTLANG_COMPILER_L1_COMPUTE_TARGET_H
#define TTLANG_COMPILER_L1_COMPUTE_TARGET_H
#include "api/compute/bcast.h"
#include "api/compute/matmul.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include <cstdint>
namespace ttlang::l1::target {
/// Encodes a byte address in the 16-byte, preincremented form used by raw LLKs.
inline uint32_t toLlkTileAddress(uint32_t byteAddress, uint32_t tile,
                                 uint32_t pageWords) {
  return (byteAddress >> 4) + tile * pageWords - 1;
}

inline void resetMatmulThrottleState() {
#if defined(ARCH_BLACKHOLE)
  MATH((ckernel::throttled_mop_status = 0));
#endif
}

template <ckernel::DataCopyType CopyType, ckernel::BroadcastType Broadcast,
          typename Source>
inline void initializeUnaryDataCopy() {
#if defined(ARCH_BLACKHOLE)
  MATH((_llk_math_eltwise_unary_datacopy_init_<CopyType, DST_ACCUM_MODE,
                                               Broadcast>(
      Source::faceCount, Source::unpackFormat, false)));
#else
  MATH((_llk_math_eltwise_unary_datacopy_init_<CopyType, DST_ACCUM_MODE,
                                               Broadcast>(
      Source::faceCount, Source::unpackFormat)));
#endif
}

template <typename Output>
inline void initializePack() {
#if defined(ARCH_BLACKHOLE)
  PACK((_llk_pack_hw_configure_<DST_ACCUM_MODE, ckernel::PackMode::Default>(
      Output::format, Output::format, Output::pageWords, Output::faceRowHeight,
      Output::width, Output::faceCount, Output::partialFace, 0)));
  PACK((_llk_pack_init_<ckernel::PackMode::Default>(
      Output::format, Output::faceRowHeight, Output::width, Output::faceCount,
      1, false)));
  PACK((_llk_pack_dest_init_<DST_SYNC_MODE, DST_ACCUM_MODE>()));
#else
  PACK((_llk_pack_hw_configure_<DST_ACCUM_MODE, ckernel::PackMode::Default>(
      Output::format, Output::format, Output::pageWords, Output::faceRowHeight,
      Output::faceCount, Output::partialFace, Output::narrowTile, 0)));
  PACK((_llk_pack_init_<ckernel::PackMode::Default>(
      Output::format, Output::faceRowHeight, Output::faceCount,
      Output::partialFace, Output::narrowTile, 1)));
  PACK((_llk_pack_dest_init_<DST_SYNC_MODE, DST_ACCUM_MODE,
                             ckernel::PackMode::Default>(Output::faceRowHeight,
                                                         Output::narrowTile)));
#endif
}

template <typename Output, bool ReconfigureTileDimensions = false>
inline void reconfigurePack() {
#if defined(ARCH_BLACKHOLE)
  PACK((_llk_pack_reconfig_data_format_<DST_ACCUM_MODE>(
      Output::format, Output::format, Output::pageWords, Output::width,
      Output::faceCount, Output::partialFace)));
  if constexpr (ReconfigureTileDimensions) {
    PACK((_llk_pack_init_<ckernel::PackMode::Default, false, true>(
        Output::format, Output::faceRowHeight, Output::width, Output::faceCount,
        1, false)));
  }
#else
  PACK((_llk_pack_reconfig_data_format_<DST_ACCUM_MODE>(
      Output::format, Output::format, Output::pageWords, Output::faceRowHeight,
      Output::faceCount, Output::partialFace, Output::narrowTile)));
  if constexpr (ReconfigureTileDimensions) {
    PACK((_llk_pack_init_<ckernel::PackMode::Default, false, true>(
        Output::format, Output::faceRowHeight, Output::faceCount,
        Output::partialFace, Output::narrowTile, 1)));
  }
#endif
}

template <typename Lhs, typename Rhs>
inline void executeMatmul(uint32_t destination, uint32_t transpose,
                          uint32_t columns, uint32_t rows) {
#if defined(ARCH_BLACKHOLE) && defined(TRISC_MATH)
  bool throttled = (*ckernel::throttle_ptr % 2) != 0;
  if (throttled) {
    if (ckernel::throttled_mop_status != 1) {
      _llk_math_matmul_init_<MATH_FIDELITY, MM_THROTTLE_MAX>(
          Lhs::height, Lhs::width, Rhs::height, Rhs::width, Lhs::partialFace,
          transpose, columns, rows);
      ckernel::throttled_mop_status = 1;
    }
    llk_math_matmul<MATH_FIDELITY, MM_THROTTLE_MAX>(destination, columns, rows);
  } else {
    if (ckernel::throttled_mop_status != 0) {
      _llk_math_matmul_init_<MATH_FIDELITY, MM_THROTTLE>(
          Lhs::height, Lhs::width, Rhs::height, Rhs::width, Lhs::partialFace,
          transpose, columns, rows);
      ckernel::throttled_mop_status = 0;
    }
    llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(destination, columns, rows);
  }
#elif defined(TRISC_MATH)
  llk_math_matmul<MATH_FIDELITY, MM_THROTTLE>(destination, columns, rows);
#endif
}
} // namespace ttlang::l1::target
#endif
