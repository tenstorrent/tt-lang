// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_MUL_REDUCE_H
#define TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_MUL_REDUCE_H

#include <cstdint>

#include "api/compute/experimental/mul_reduce_scalar.h"

namespace experimental {

template <std::uint32_t numTiles>
ALWI void multiply_full_scalar_reduction_block(std::uint32_t lhsDfb,
                                               std::uint32_t rhsDfb,
                                               std::uint32_t outputDfb,
                                               float reductionStageScaler) {
  static_assert(numTiles >= 1 && numTiles <= 8,
                "multiply-reduction requires 1 to 8 tiles");
  ckernel::mul_reduce_scalar_init(lhsDfb, rhsDfb);
  ckernel::mul_reduce_scalar_tile(lhsDfb, rhsDfb, outputDfb, numTiles,
                                  reductionStageScaler);
  ckernel::mul_reduce_scalar_uninit();
}

} // namespace experimental

#endif // TTLANG_TARGET_TTKERNEL_LLKS_EXPERIMENTAL_MUL_REDUCE_H
