// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "TTLOpsVerifyUtils.h"

#include "mlir/Support/LogicalResult.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::tt::ttl::verify {
namespace {

// Return true when `v` is a transfer handle produced by `ttl.copy`, allowing
// the handle to flow through tensor containers and loop-carried values.
// `ttl.pipe_recv_post` is accepted because receive-side `ttl.copy` expands to
// that internal op before lowering while preserving the original wait contract.
static bool isDerivedFromCopy(mlir::Value value) {
  llvm::SmallPtrSet<mlir::Value, 16> seen;
  return mlir::tt::ttl::traceTransferHandleSource<bool>(
      value,
      [](mlir::Value source) {
        return source.getDefiningOp<mlir::tt::ttl::CopyOp>() != nullptr ||
               source.getDefiningOp<mlir::tt::ttl::PipeRecvPostOp>() != nullptr;
      },
      seen);
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

  if (isDerivedFromCopy(handle)) {
    return mlir::success();
  }

  return op->emitOpError() << "expects operand to be derived from ttl.copy.";
}

} // namespace mlir::tt::ttl::verify
