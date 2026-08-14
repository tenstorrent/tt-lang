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

/// One ordered protocol action on a DFB dependency occurrence.
struct DFBProtocolEffect {
  mlir::Value dfb;
  DFBProtocolEffectKind kind = DFBProtocolEffectKind::Reserve;
  int64_t numTiles = 0;
  unsigned dependencyIndex = 0;
  unsigned sequenceIndex = 0;
};

/// One typed non-transactional access to a DFB dependency occurrence.
struct DFBNonTransactionalAccess {
  mlir::Value dfb;
  DFBNonTransactionalAccessKind kind =
      DFBNonTransactionalAccessKind::InterfacePreserved;
  unsigned dependencyIndex = 0;
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
