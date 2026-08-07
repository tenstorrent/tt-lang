// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_FIXEDBLOCKCOMPUTEANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_FIXEDBLOCKCOMPUTEANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include <cstdint>
#include <string>

namespace mlir::tt::ttl {

/// Common immutable facts for a compute body whose block operation owns its
/// complete DST schedule and publishes through one output store.
struct FixedBlockComputeAnalysis {
  Operation *block = nullptr;
  SmallVector<Value> inputTensors;
  Value outputTensor;
  TileStoreOp store;
  std::uint32_t dstCapacity = 0;
};

/// Analyze a fixed block compute without modifying IR or emitting diagnostics.
/// The caller supplies the exact body operands and result defined by its block
/// operation; operation-specific shape and target constraints remain with the
/// caller.
FailureOr<FixedBlockComputeAnalysis>
analyzeFixedBlockCompute(ComputeOp compute, Operation *block,
                         ValueRange bodyInputs, Value bodyOutput,
                         Value blockResult, std::string &reason);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_FIXEDBLOCKCOMPUTEANALYSIS_H
