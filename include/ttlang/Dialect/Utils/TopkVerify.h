// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_UTILS_TOPKVERIFY_H
#define TTLANG_DIALECT_UTILS_TOPKVERIFY_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir::tt::utils {

// Constant operands are checked here. A dynamic operand is left to the metal
// runtime range checks: solving integer ranges for every operand of every
// TopK op repeats a whole-function dataflow solve.

inline LogicalResult verifyTopkMode(Operation *op, bool stableSort, bool fused,
                                    bool rankStamped, bool tieOrderUnset,
                                    BoolAttr fp32DestAccEn, uint32_t tagBits) {
  if (stableSort && tieOrderUnset) {
    return op->emitOpError("stable_sort requires an explicit tie_order");
  }
  if (fused && stableSort) {
    return op->emitOpError("fused and stable_sort are mutually exclusive");
  }
  if (rankStamped && stableSort) {
    return op->emitOpError(
        "rank_stamped and stable_sort are mutually exclusive");
  }
  if (rankStamped && fused) {
    return op->emitOpError("rank_stamped and fused are mutually exclusive");
  }
  if (fp32DestAccEn && !fp32DestAccEn.getValue() && (fused || rankStamped)) {
    return op->emitOpError(
        "fused and rank_stamped modes require fp32 destination accumulation");
  }
  if (rankStamped) {
    if (tagBits < 6 || tagBits > 16) {
      return op->emitOpError("tag_bits must be in the range [6, 16]");
    }
  } else if (tagBits != 16) {
    return op->emitOpError("tag_bits applies only to rank_stamped mode");
  }
  return success();
}

inline LogicalResult verifyTopkStep(Operation *op, Value step, StringRef name) {
  if (!step) {
    return success();
  }
  std::optional<int64_t> value = getConstantIntValue(step);
  if (!value) {
    return success();
  }
  if (*value == 0 || (*value >= 4 && *value <= 6)) {
    return success();
  }
  return op->emitOpError() << name << " must be 0 or in the range [4, 6]";
}

inline LogicalResult verifyTopkConstantInRange(Operation *op, Value value,
                                               StringRef name, int64_t lower,
                                               int64_t upper) {
  std::optional<int64_t> constant = getConstantIntValue(value);
  if (!constant) {
    return success();
  }
  if (*constant >= lower && *constant <= upper) {
    return success();
  }
  return op->emitOpError() << name << " must be in the range [" << lower << ", "
                           << upper << "]";
}

inline constexpr int64_t kTopkSupportedKValues[] = {4, 8, 16, 32, 64};

inline LogicalResult verifyTopkConstantK(Operation *op, Value value) {
  std::optional<int64_t> constant = getConstantIntValue(value);
  if (!constant || llvm::is_contained(kTopkSupportedKValues, *constant)) {
    return success();
  }
  return op->emitOpError("k must be one of {4, 8, 16, 32, 64}");
}

inline LogicalResult verifyTopkPhaseOrder(Operation *op, Value startPhase,
                                          Value endPhase, StringRef startName,
                                          StringRef endName) {
  std::optional<int64_t> start = getConstantIntValue(startPhase);
  std::optional<int64_t> end = getConstantIntValue(endPhase);
  if (start && end && *start > *end) {
    return op->emitOpError() << startName << " must not exceed " << endName;
  }
  return success();
}

inline LogicalResult verifyTopkLogkMatchesK(Operation *op, Value k,
                                            Value logk) {
  std::optional<int64_t> kValue = getConstantIntValue(k);
  std::optional<int64_t> logkValue = getConstantIntValue(logk);
  if (kValue && logkValue && (1LL << *logkValue) != *kValue) {
    return op->emitOpError("logk must equal log2(k)");
  }
  return success();
}

inline LogicalResult verifyTopkTagBits(Operation *op, uint32_t tagBits) {
  if (tagBits < 6 || tagBits > 16) {
    return op->emitOpError("tag_bits must be in the range [6, 16]");
  }
  return success();
}

inline LogicalResult verifyTopkStripFp32(Operation *op,
                                         BoolAttr fp32DestAccEn) {
  if (fp32DestAccEn && !fp32DestAccEn.getValue()) {
    return op->emitOpError(
        "strip_rank_tags requires fp32 destination accumulation");
  }
  return success();
}

} // namespace mlir::tt::utils

#endif // TTLANG_DIALECT_UTILS_TOPKVERIFY_H
