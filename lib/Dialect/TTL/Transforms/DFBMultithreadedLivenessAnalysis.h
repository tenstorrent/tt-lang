// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBMULTITHREADEDLIVENESSANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBMULTITHREADEDLIVENESSANALYSIS_H

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace mlir::tt::ttl {

struct DFBPhysicalIndexAssignment {
  int64_t logicalId = 0;
  int32_t physicalIndex = 0;
  Type type;
  SmallVector<BindCBOp> declarations;
  bool bounded = false;
};

/// Read-only module analysis that computes a conservative physical DFB index
/// assignment from multithreaded happens-before relations.
class DFBMultithreadedLivenessAnalysis {
public:
  explicit DFBMultithreadedLivenessAnalysis(Operation *operation);

  bool succeeded() const { return errorMessage.empty(); }
  StringRef getErrorMessage() const { return errorMessage; }
  Operation *getErrorOperation() const { return errorOperation; }

  ArrayRef<DFBPhysicalIndexAssignment> getAssignments() const {
    return assignments;
  }
  int32_t getPhysicalSlotCount() const { return physicalSlotCount; }

private:
  SmallVector<DFBPhysicalIndexAssignment> assignments;
  int32_t physicalSlotCount = 0;
  Operation *errorOperation = nullptr;
  std::string errorMessage;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBMULTITHREADEDLIVENESSANALYSIS_H
