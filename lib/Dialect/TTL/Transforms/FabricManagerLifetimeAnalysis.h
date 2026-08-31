// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_FABRICMANAGERLIFETIMEANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_FABRICMANAGERLIFETIMEANALYSIS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttl {

/// One verified external manager ownership interval.
struct ExternalFabricManagerInterval {
  StringAttr claim;
  func::FuncOp function;
  OpaqueCallOp acquire;
  OpaqueCallOp release;
};

/// Validate external manager effects without mutating the module.
FailureOr<SmallVector<ExternalFabricManagerInterval>>
analyzeExternalFabricManagerLifetimes(ModuleOp module);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_FABRICMANAGERLIFETIMEANALYSIS_H
