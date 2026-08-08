// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_UTILS_OPAQUECALLVERIFYUTILS_H
#define TTLANG_DIALECT_UTILS_OPAQUECALLVERIFYUTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace mlir::tt::utils {

/// Verifies the attributes shared by `ttl.opaque_call` and
/// `ttkernel.opaque_call`.
inline LogicalResult verifyOpaqueCallNames(Operation *op, StringRef callee,
                                           StringRef header) {
  if (callee.empty()) {
    return op->emitOpError("callee name must not be empty");
  }
  if (header.empty()) {
    return op->emitOpError("header path must not be empty");
  }

  return success();
}

/// Verifies operand positions that require unsigned C++ call expressions.
inline LogicalResult
verifyOpaqueCallUnsignedArgIndices(Operation *op,
                                   std::optional<ArrayRef<int32_t>> indices,
                                   ValueRange arguments) {
  if (!indices) {
    return success();
  }

  int32_t previousIndex = -1;
  for (int32_t index : *indices) {
    if (index < 0 || static_cast<size_t>(index) >= arguments.size()) {
      return op->emitOpError("unsigned function argument index ")
             << index << " is out of range for " << arguments.size()
             << " arguments";
    }
    if (index <= previousIndex) {
      return op->emitOpError(
          "unsigned function argument indices must be strictly increasing");
    }
    auto integerType = dyn_cast<IntegerType>(arguments[index].getType());
    if (!integerType || integerType.getWidth() != 32) {
      return op->emitOpError("unsigned function argument index ")
             << index << " must reference a 32-bit integer operand, got "
             << arguments[index].getType();
    }
    previousIndex = index;
  }
  return success();
}

} // namespace mlir::tt::utils

#endif // TTLANG_DIALECT_UTILS_OPAQUECALLVERIFYUTILS_H
