//===- PipeConstants.h - Shared PipeNet constants --------------*- C++ -*-===//
//
// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
//===----------------------------------------------------------------------===//
//
// This file defines constants shared by PipeNet resource planning and
// lowering.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPECONSTANTS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPECONSTANTS_H

#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::tt::ttl {

/// Size of one receiver-published address-table entry.
inline constexpr int64_t kPipeAddressWordBytes = 4;

/// Alignment used for independently addressed PipeNet scratch allocations.
inline constexpr int64_t kPipeSramScratchAlignmentBytes = 32;

/// Return `bytes` aligned for PipeNet SRAM scratch, or no value on overflow.
inline std::optional<int64_t> alignPipeSramScratchBytes(int64_t bytes) {
  if (bytes < 0 || bytes > std::numeric_limits<int64_t>::max() -
                               (kPipeSramScratchAlignmentBytes - 1)) {
    return std::nullopt;
  }
  return llvm::alignTo(bytes, kPipeSramScratchAlignmentBytes);
}

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPECONSTANTS_H
