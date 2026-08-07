// SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROW_NORMALIZATION_H
#define TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROW_NORMALIZATION_H

#include <cstdint>

#include "api/compute/eltwise_binary.h"

namespace experimental {

template <std::uint32_t numTiles, bool hasGamma, DataFormat dataFormat>
ALWI void row_normalization_block(std::uint32_t inputDfb,
                                  std::uint32_t gammaDfb,
                                  std::uint32_t outputDfb,
                                  std::uint32_t reductionScalerBits,
                                  std::uint32_t epsilonBits) {
  static_assert(numTiles >= 1 && numTiles <= 8,
                "row normalization requires 1 to 8 tiles");
  static_assert(dataFormat == DataFormat::Float16_b,
                "row normalization supports bf16 DFBs only");

  // mul_reduce_scalar_tile applies its scaler during both reduction stages.
  // TTKernel lowering therefore passes sqrt of the semantic reduction scale.
  float reductionStageScaler;
  __builtin_memcpy(&reductionStageScaler, &reductionScalerBits,
                   sizeof(reductionStageScaler));
  add_rsqrt_init();
  multiply_full_scalar_reduction_block<numTiles>(inputDfb, inputDfb, outputDfb,
                                                 reductionStageScaler);

  add_rsqrt(0, epsilonBits);
  source_scalar_mul_init<numTiles>(inputDfb);
  source_scalar_mul<numTiles>(inputDfb, 0, 0);

  if constexpr (hasGamma) {
    ckernel::binary_dest_reuse_tiles_init<
        ckernel::EltwiseBinaryType::ELWMUL,
        ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(gammaDfb);
    for (std::uint32_t tileIndex = 0; tileIndex < numTiles; ++tileIndex) {
      ckernel::binary_dest_reuse_tiles<
          ckernel::EltwiseBinaryType::ELWMUL,
          ckernel::EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
          gammaDfb, tileIndex, tileIndex);
    }
  }
}

} // namespace experimental

#endif // TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_ROW_NORMALIZATION_H
