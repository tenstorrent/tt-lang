// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPECONSTANTS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPECONSTANTS_H

#include <cstdint>

namespace mlir::tt::ttl {

/// Size of one receiver-published address-table entry.
inline constexpr int64_t kPipeAddressWordBytes = 4;

/// Alignment used for independently addressed PipeNet scratch allocations.
inline constexpr int64_t kPipeSramScratchAlignmentBytes = 32;

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPECONSTANTS_H
