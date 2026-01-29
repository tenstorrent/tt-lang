// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "TTLOpsVerifyUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir::tt::ttl::verify {
namespace {

static bool isTypedTransferHandleType(mlir::Type t) {
  auto xf = mlir::dyn_cast<mlir::tt::ttl::TransferHandleType>(t);
  return xf && xf.getKind().has_value();
}

static bool isTypedTransferHandleTensorType(mlir::Type t) {
  auto tensorTy = mlir::dyn_cast<mlir::TensorType>(t);
  if (!tensorTy || !tensorTy.hasRank() || tensorTy.getRank() != 1) {
    return false;
  }
  return isTypedTransferHandleType(tensorTy.getElementType());
}

static std::optional<int64_t> getConstantIndexValue(mlir::Value v) {
  if (!v) {
    return std::nullopt;
  }
  if (auto cst = v.getDefiningOp<mlir::arith::ConstantIndexOp>()) {
    return cst.value();
  }
  return std::nullopt;
}

struct IterSpace {
  int64_t lb;
  int64_t ub;
  int64_t step;
};

static std::optional<IterSpace> getConstIterSpace(mlir::scf::ForOp forOp) {
  auto lb = getConstantIndexValue(forOp.getLowerBound());
  auto ub = getConstantIndexValue(forOp.getUpperBound());
  auto step = getConstantIndexValue(forOp.getStep());
  if (!lb || !ub || !step) {
    return std::nullopt;
  }
  return IterSpace{*lb, *ub, *step};
}

static bool isDirectWaitInLoopBody(mlir::tensor::ExtractOp extractOp,
                                   mlir::scf::ForOp waitLoop) {
  // Require `ttl.wait` on the extracted handle in the same loop body.
  for (mlir::Operation *user : extractOp.getResult().getUsers()) {
    if (!llvm::isa<mlir::tt::ttl::WaitOp>(user)) {
      continue;
    }
    if (waitLoop->isProperAncestor(user)) {
      return true;
    }
  }
  return false;
}

/// Try to prove that all handle values written into a tensor container are
/// later waited on.
///
/// This is used to support the "two-phase" pattern where code batches transfer
/// handles by writing them into a tensor (via `tensor.insert`), and later reads
/// them back (via `tensor.extract`) and waits in a separate location.
///
/// This helper is intentionally value-centric; loops are just one way to prove
/// that the set of written indices is fully covered by the set of waited
/// indices. If we can match a simple for-loop based proof, we return a
/// definitive answer. Otherwise we return std::nullopt to fall back to the
/// conservative reachability walk.
///
/// Returns:
/// - true if it can prove all inserted indices are waited.
/// - false if it can prove some inserted indices are not waited.
/// - std::nullopt if it cannot match the pattern.
static std::optional<bool>
allTensorContainerWritesAreWaited(mlir::Value handle,
                                  mlir::tensor::InsertOp insertOp) {
  // Only handle the simple 1D typed-handle tensor case.
  // Note: MVP rule forbids using untyped `!ttl.transfer_handle` inside
  // containers.
  if (!isTypedTransferHandleTensorType(insertOp.getResult().getType())) {
    return std::nullopt;
  }
  if (insertOp.getIndices().size() != 1) {
    return std::nullopt;
  }

  auto writerLoop = insertOp->getParentOfType<mlir::scf::ForOp>();
  if (!writerLoop) {
    return std::nullopt;
  }
  if (insertOp.getIndices()[0] != writerLoop.getInductionVar()) {
    return std::nullopt;
  }

  // Require that this insert is yielding the updated tensor out of the loop.
  auto *terminator = writerLoop.getBody()->getTerminator();
  auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(terminator);
  if (!yieldOp) {
    return std::nullopt;
  }

  std::optional<unsigned> yieldedIdx;
  for (unsigned idx = 0; idx < yieldOp.getNumOperands(); ++idx) {
    if (yieldOp.getOperand(idx) == insertOp.getResult()) {
      yieldedIdx = idx;
      break;
    }
  }
  if (!yieldedIdx) {
    return std::nullopt;
  }

  mlir::Value handlesTensor = writerLoop.getResult(*yieldedIdx);

  // Find a reader loop that iterates over indices and waits on extracted
  // handles. Use constant iteration spaces as a simple coverage proof.
  std::optional<IterSpace> writeSpace = getConstIterSpace(writerLoop);
  if (!writeSpace) {
    return std::nullopt;
  }

  bool sawWaitLoop = false;
  bool allCovered = false;

  for (mlir::OpOperand &use : handlesTensor.getUses()) {
    auto extractOp = llvm::dyn_cast<mlir::tensor::ExtractOp>(use.getOwner());
    if (!extractOp || use.get() != extractOp.getTensor()) {
      continue;
    }
    if (extractOp.getIndices().size() != 1) {
      continue;
    }

    auto readerLoop = extractOp->getParentOfType<mlir::scf::ForOp>();
    if (!readerLoop) {
      continue;
    }
    if (extractOp.getIndices()[0] != readerLoop.getInductionVar()) {
      continue;
    }
    if (!isDirectWaitInLoopBody(extractOp, readerLoop)) {
      continue;
    }

    auto readSpace = getConstIterSpace(readerLoop);
    if (!readSpace) {
      continue;
    }

    sawWaitLoop = true;

    // Minimal, explicit check for the MVP tests: step==1 and same lower bound.
    // Require wait loop to cover the full copy loop range: [lb, ub).
    if (writeSpace->step == 1 && readSpace->step == 1 &&
        readSpace->lb == writeSpace->lb && readSpace->ub >= writeSpace->ub) {
      allCovered = true;
    }
  }

  if (!sawWaitLoop) {
    return std::nullopt;
  }
  return allCovered;
}

static bool
tryEnqueueForwardedHandle(mlir::Value v, mlir::OpOperand &use,
                          llvm::SmallVectorImpl<mlir::Value> &queue) {
  Operation *consumerOp = use.getOwner();

  if (llvm::isa<mlir::tt::ttl::WaitOp>(consumerOp)) {
    return true;
  }

  // tensor.insert: propagate from scalar -> result tensor.
  if (auto insertOp = llvm::dyn_cast<mlir::tensor::InsertOp>(consumerOp)) {
    if (use.get() == insertOp.getScalar()) {
      if (auto covered = allTensorContainerWritesAreWaited(v, insertOp)) {
        if (!*covered) {
          // We matched a two-loop pattern but the wait loop does not cover all
          // inserted indices. Do not propagate along this edge.
          return false;
        }
      }
      queue.push_back(insertOp.getResult());
    }
    return false;
  }

  // tensor.extract: propagate from tensor -> scalar result.
  if (auto extractOp = llvm::dyn_cast<mlir::tensor::ExtractOp>(consumerOp)) {
    if (use.get() == extractOp.getTensor()) {
      queue.push_back(extractOp.getResult());
    }
    return false;
  }

  // LoopLike init -> iter_arg propagation.
  if (auto loop = llvm::dyn_cast<mlir::LoopLikeOpInterface>(consumerOp)) {
    auto inits = loop.getInitsMutable();
    auto iterArgs = loop.getRegionIterArgs();
    for (unsigned idx = 0; idx < inits.size(); ++idx) {
      if (use.get() != inits[idx].get()) {
        continue;
      }
      queue.push_back(iterArgs[idx]);
      break;
    }
    return false;
  }

  // scf.yield: yielded -> loop result propagation (when parent is LoopLike).
  if (auto yieldOp = llvm::dyn_cast<mlir::scf::YieldOp>(consumerOp)) {
    auto loop =
        llvm::dyn_cast<mlir::LoopLikeOpInterface>(yieldOp->getParentOp());
    if (!loop) {
      return false;
    }
    auto yieldedOpt = loop.getYieldedValuesMutable();
    auto resultsOpt = loop.getLoopResults();
    if (!yieldedOpt || !resultsOpt) {
      return false;
    }
    auto yielded = *yieldedOpt;
    auto results = *resultsOpt;
    for (unsigned idx = 0; idx < yielded.size(); ++idx) {
      if (use.get() != yielded[idx].get()) {
        continue;
      }
      queue.push_back(results[idx]);
      break;
    }
    return false;
  }

  return false;
}

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

mlir::LogicalResult isEventuallyWaitedOn(mlir::Operation *op,
                                         mlir::Value handle) {
  // Accept any TransferHandleType (typed or untyped).
  // Typed handles get barriers, untyped handles (e.g., pipe receive) are no-ops.
  if (!mlir::isa<mlir::tt::ttl::TransferHandleType>(handle.getType())) {
    return op->emitOpError()
           << "expects transfer handle (!ttl.transfer_handle), got "
           << handle.getType();
  }

  llvm::SmallPtrSet<mlir::Value, 16> visited;
  llvm::SmallVector<mlir::Value, 16> worklist;
  worklist.push_back(handle);

  while (!worklist.empty()) {
    mlir::Value v = worklist.pop_back_val();
    if (!visited.insert(v).second) {
      continue;
    }

    for (mlir::OpOperand &use : v.getUses()) {
      if (tryEnqueueForwardedHandle(v, use, worklist)) {
        return mlir::success();
      }
    }
  }

  return op->emitOpError()
         << "expects transfer handle to be synchronized with ttl.wait.";
}

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
