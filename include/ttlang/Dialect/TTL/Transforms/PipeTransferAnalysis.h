// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Pipe Transfer Analysis
//===----------------------------------------------------------------------===//
//
// This file declares immutable associations between pipe transfer protocol
// operations. The index preserves the SSA identity of receive tokens so
// schedule verification and resource planning use the same post/wait relation.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPETRANSFERANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPETRANSFERANALYSIS_H

#include "ttlang/Analysis/ValueOriginAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <optional>

namespace mlir::tt::ttl {

/// Immutable associations between pipe protocol operations in one module.
///
/// Public receive handles associate `ttl.wait` with the exact receive
/// `ttl.copy` that produced the handle. An internal wait can receive tokens
/// from multiple posts through control flow; it is associated with every post
/// and with their shared transfer creation op. Internal posts and sends also
/// associate with their unique transfer creation op. Construction rejects
/// incompatible provenance instead of selecting one origin. The result is
/// valid only while `module` remains unchanged.
class PipeTransferIndex {
public:
  /// Builds all pipe transfer associations without modifying IR.
  static FailureOr<std::unique_ptr<PipeTransferIndex>>
  create(ModuleOp module, ValueOriginAnalysis &valueOrigins);

  /// Returns the defining pipe receive, or no value for a non-pipe wait.
  std::optional<CopyOp> getReceivePost(WaitOp waitOp) const;

  /// Returns every receive post whose token may reach `waitOp`.
  ArrayRef<Operation *>
  getPossibleReceivePosts(PipeTransferWaitOp waitOp) const;

  /// Returns the transfer creation associated with an internal protocol op,
  /// or failure when the operation has no modeled association.
  FailureOr<PipeTransferCreateOp>
  getTransferCreate(Operation *protocolOp) const;

private:
  PipeTransferIndex() = default;

  LogicalResult build(ModuleOp module, ValueOriginAnalysis &valueOrigins);

  llvm::DenseMap<Operation *, Operation *> receivePostByWait;
  llvm::DenseMap<Operation *, SmallVector<Operation *>> receivePostsByWait;
  llvm::DenseMap<Operation *, Operation *> transferCreateByProtocolOp;
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPETRANSFERANALYSIS_H
