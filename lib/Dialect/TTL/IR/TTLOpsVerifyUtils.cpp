// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "TTLOpsVerifyUtils.h"

#include "ttlang/Analysis/ValueOriginAnalysis.h"

#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::ttl::verify {
namespace {

// Return true when every possible origin has `ttl.wait` semantics.
static bool isDerivedFromTransfer(mlir::Value value) {
  mlir::tt::ValueOriginAnalysis analysis;
  llvm::SmallVector<mlir::Value> origins = analysis.getOrigins(value);
  return !origins.empty() && llvm::all_of(origins, [](mlir::Value origin) {
    return origin.getDefiningOp<mlir::tt::ttl::CopyOp>() != nullptr ||
           origin.getDefiningOp<mlir::tt::ttl::PipeTransferSendOp>() != nullptr;
  });
}

} // namespace

mlir::LogicalResult isValidWaitOperand(mlir::Operation *op,
                                       mlir::Value handle) {
  // Accept typed and untyped transfer handles. Untyped handles model async
  // pipe receive completion and are expanded before lowering.
  if (!mlir::isa<mlir::tt::ttl::TransferHandleType>(handle.getType())) {
    return op->emitOpError()
           << "expects transfer handle (!ttl.transfer_handle), got "
           << handle.getType();
  }

  if (isDerivedFromTransfer(handle)) {
    return mlir::success();
  }

  return op->emitOpError() << "expects operand to be derived from ttl.copy or "
                              "ttl.pipe_transfer.send.";
}

} // namespace mlir::tt::ttl::verify
