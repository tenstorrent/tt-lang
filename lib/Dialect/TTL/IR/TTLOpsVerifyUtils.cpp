// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "TTLOpsVerifyUtils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::tt::ttl::verify {
namespace {

static bool isDerivedFromCopy(mlir::Value v,
                              llvm::SmallPtrSetImpl<mlir::Value> &seen) {
  if (!seen.insert(v).second) {
    return false;
  }

  if (v.getDefiningOp<mlir::tt::ttl::CopyOp>() != nullptr) {
    return true;
  }

  // tensor.extract: the extracted handle is only valid if the source tensor is
  // derived from a copy.
  if (auto extractOp = v.getDefiningOp<mlir::tensor::ExtractOp>()) {
    return isDerivedFromCopy(extractOp.getTensor(), seen);
  }

  // tensor.insert: the tensor is derived from a copy if the inserted scalar is
  // derived from a copy.
  if (auto insertOp = v.getDefiningOp<mlir::tensor::InsertOp>()) {
    return isDerivedFromCopy(insertOp.getScalar(), seen);
  }

  // Loop result -> yielded value.
  if (auto res = llvm::dyn_cast<mlir::OpResult>(v)) {
    if (auto loop = llvm::dyn_cast<mlir::LoopLikeOpInterface>(res.getOwner())) {
      auto yieldedOpt = loop.getYieldedValuesMutable();
      auto resultsOpt = loop.getLoopResults();
      if (yieldedOpt && resultsOpt) {
        auto yielded = *yieldedOpt;
        auto results = *resultsOpt;
        for (unsigned idx = 0; idx < results.size(); ++idx) {
          if (results[idx] != res) {
            continue;
          }
          return isDerivedFromCopy(yielded[idx].get(), seen);
        }
      }
    }
  }

  // Loop iter_arg -> init.
  if (auto barg = llvm::dyn_cast<mlir::BlockArgument>(v)) {
    auto *parent = barg.getOwner()->getParentOp();
    if (auto loop = llvm::dyn_cast_or_null<mlir::LoopLikeOpInterface>(parent)) {
      auto iterArgs = loop.getRegionIterArgs();
      auto inits = loop.getInitsMutable();
      for (unsigned idx = 0; idx < iterArgs.size(); ++idx) {
        if (iterArgs[idx] != barg) {
          continue;
        }
        return isDerivedFromCopy(inits[idx].get(), seen);
      }
    }
  }

  return false;
}

} // namespace

mlir::LogicalResult isValidWaitOperand(mlir::Operation *op,
                                       mlir::Value handle) {
  // Accept any TransferHandleType (typed or untyped).
  // Typed handles (read/write) get corresponding barriers.
  // Untyped handles (e.g., pipe receive) are no-ops since data arrives via
  // multicast from source core.
  if (!mlir::isa<mlir::tt::ttl::TransferHandleType>(handle.getType())) {
    return op->emitOpError()
           << "expects transfer handle (!ttl.transfer_handle), got "
           << handle.getType();
  }

  llvm::SmallPtrSet<mlir::Value, 16> visited;
  if (isDerivedFromCopy(handle, visited)) {
    return mlir::success();
  }

  return op->emitOpError() << "expects operand to be the result of ttl.copy.";
}

} // namespace mlir::tt::ttl::verify
