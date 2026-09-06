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

template <ckernel::DataCopyType CopyType, ckernel::BroadcastType Broadcast>
inline void initializeUnaryDataCopy(uint32_t unpackFormat) {
#if defined(ARCH_BLACKHOLE)
  MATH((_llk_math_eltwise_unary_datacopy_init_<CopyType, DST_ACCUM_MODE,
                                               Broadcast>(4, unpackFormat,
                                                          false)));
#else
  MATH((_llk_math_eltwise_unary_datacopy_init_<CopyType, DST_ACCUM_MODE,
                                               Broadcast>(4, unpackFormat)));
#endif
}

template <uint32_t OutputFormat, uint32_t OutputPageWords>
inline void initializePack() {
#if defined(ARCH_BLACKHOLE)
  PACK((_llk_pack_hw_configure_<DST_ACCUM_MODE, ckernel::PackMode::Default>(
      OutputFormat, OutputFormat, OutputPageWords, 16, 32, 4, false, 0)));
  PACK((_llk_pack_init_<ckernel::PackMode::Default>(OutputFormat, 16, 32, 4, 1,
                                                    false)));
  PACK((_llk_pack_dest_init_<DST_SYNC_MODE, DST_ACCUM_MODE>()));
#else
  PACK((_llk_pack_hw_configure_<DST_ACCUM_MODE, ckernel::PackMode::Default>(
      OutputFormat, OutputFormat, OutputPageWords, 16, 4, false, false, 0)));
  PACK((_llk_pack_init_<ckernel::PackMode::Default>(OutputFormat, 16, 4, false,
                                                    false, 1)));
  PACK((_llk_pack_dest_init_<DST_SYNC_MODE, DST_ACCUM_MODE,
                             ckernel::PackMode::Default>(16, false)));
#endif
}

template <uint32_t OutputFormat, uint32_t OutputPageWords>
inline void reconfigurePack() {
#if defined(ARCH_BLACKHOLE)
  PACK((_llk_pack_reconfig_data_format_<DST_ACCUM_MODE>(
      OutputFormat, OutputFormat, OutputPageWords, 32, 4, false)));
#else
  PACK((_llk_pack_reconfig_data_format_<DST_ACCUM_MODE>(
      OutputFormat, OutputFormat, OutputPageWords, 16, 4, false, false)));
#endif
}

inline void executeMatmul(uint32_t destination, uint32_t transpose,
                          uint32_t columns, uint32_t rows) {
#if defined(ARCH_BLACKHOLE) && defined(TRISC_MATH)
  bool throttled = (*ckernel::throttle_ptr % 2) != 0;
  if (throttled) {
    if (ckernel::throttled_mop_status != 1) {
      _llk_math_matmul_init_<MATH_FIDELITY, MM_THROTTLE_MAX>(
          32, 32, 32, 32, false, transpose, columns, rows);
      ckernel::throttled_mop_status = 1;
    }
    llk_math_matmul<MATH_FIDELITY, MM_THROTTLE_MAX>(destination, columns, rows);
  } else {
    if (ckernel::throttled_mop_status != 0) {
      _llk_math_matmul_init_<MATH_FIDELITY, MM_THROTTLE>(
          32, 32, 32, 32, false, transpose, columns, rows);
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
