// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_IR_TTLOPS_H
#define TTLANG_DIALECT_TTL_IR_TTLOPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/IndexingMapOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsEnums.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TileExecution.h"

namespace mlir::tt::ttl {

/// Canonical protocol action returned by `DFBAccessOpInterface`.
///
/// This representation lets analyses handle concrete DFB lifecycle operations
/// and external-call summaries uniformly. Concrete lifecycle operations expose
/// one action; an operation that performs several actions exposes one record
/// per action in execution order. Records describe synchronous behavior already
/// performed by the operation and do not request insertion of protocol IR.
struct DFBProtocolEffect {
  /// DFB resolved from the indexed dependency occurrence for identity queries.
  mlir::Value dfb;

  /// Lifecycle transition used to classify pointer ownership and transactions.
  DFBProtocolEffectKind kind = DFBProtocolEffectKind::Reserve;

  /// Positive tile count used to match and compare protocol transactions.
  int64_t numTiles = 0;

  /// Index into `getDFBDependencyOperands()`; aliased values remain separate.
  unsigned dependencyIndex = 0;

  /// Position among this operation's actions, including actions on other DFBs.
  unsigned sequenceIndex = 0;
};

inline bool isProducerDFBProtocolEffect(DFBProtocolEffectKind kind) {
  return kind == DFBProtocolEffectKind::Reserve ||
         kind == DFBProtocolEffectKind::Push;
}

inline bool isConsumerDFBProtocolEffect(DFBProtocolEffectKind kind) {
  return kind == DFBProtocolEffectKind::Wait ||
         kind == DFBProtocolEffectKind::Pop;
}

} // namespace mlir::tt::ttl

#include "ttlang/Dialect/TTL/IR/TTLInterfaces.h.inc"

#define GET_OP_CLASSES
#include "ttlang/Dialect/TTL/IR/TTLOps.h.inc"

#endif // TTLANG_DIALECT_TTL_IR_TTLOPS_H
