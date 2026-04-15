// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

namespace mlir::tt::ttl {

//===----------------------------------------------------------------------===//
// Tile operation classification
//===----------------------------------------------------------------------===//

TileOpCategory classifyTileOp(Operation *op) {
  if (isa<CopyTileOp>(op)) {
    return TileOpCategory::CopyTile;
  }
  if (isa<CopyDstOp>(op)) {
    return TileOpCategory::CopyDst;
  }
  if (isa<TileBcastOp>(op)) {
    return TileOpCategory::Bcast;
  }
  if (isa<TileMatmulBlockOp>(op)) {
    return TileOpCategory::FPUBinary;
  }
  // TODO: add TileOpCategory::Transpose case when TTL transpose op is added.

  // FPU binary: marked by kFPUBinaryAttrName attribute.
  if (op->hasAttr(kFPUBinaryAttrName)) {
    return TileOpCategory::FPUBinary;
  }
  // SFPU unary: tile unary ops that operate in-place on DST.
  if (op->hasTrait<TTLTileUnaryOpTrait>()) {
    return TileOpCategory::SFPUUnary;
  }
  // SFPU binary: tile binary ops that read both operands from DST.
  if (op->hasTrait<TTLTileBinaryOpTrait>()) {
    return TileOpCategory::SFPUBinary;
  }
  return TileOpCategory::Unknown;
}

FusionTraceResult traceFusionToRoots(mlir::Value value) {
  FusionTraceResult result;

  // Base case: CB-attached value is a root
  if (getAttachedCB(value)) {
    result.rootInputs.insert(value);
    return result;
  }

  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    result.failureReason = TraceFailureReason::NotCBAttached;
    result.failedValue = value;
    return result;
  }

  // Special case: BcastOp can be fused when its input is CB-attached.
  if (auto bcastOp = llvm::dyn_cast<BcastOp>(defOp)) {
    mlir::Value bcastInput = bcastOp.getInput();
    if (getAttachedCB(bcastInput)) {
      result.rootInputs.insert(bcastInput);
      result.opsInOrder.insert(defOp);
      return result;
    }
    // Bcast recognized but input not CB-attached.
    result.failureReason = TraceFailureReason::NotCBAttached;
    result.failedValue = bcastInput;
    return result;
  }

  // Special case: MatmulOp with CB-attached inputs is a fusable leaf.
  // Both inputs become roots; the trace does not recurse into the matmul.
  if (auto matmulOp = llvm::dyn_cast<MatmulOp>(defOp)) {
    mlir::Value lhs = matmulOp.getLhs();
    mlir::Value rhs = matmulOp.getRhs();
    if (getAttachedCB(lhs) && getAttachedCB(rhs)) {
      result.rootInputs.insert(lhs);
      result.rootInputs.insert(rhs);
      result.opsInOrder.insert(defOp);
      return result;
    }
    // Matmul recognized but inputs not CB-attached.
    result.failureReason = TraceFailureReason::NotCBAttached;
    result.failedValue = getAttachedCB(lhs) ? rhs : lhs;
    return result;
  }

  // FillOp is a fusable leaf: it produces a value with no input operands.
  if (isa<FillOp>(defOp)) {
    result.opsInOrder.insert(defOp);
    return result;
  }

  if (!isElementwiseOp(defOp)) {
    result.failureReason = TraceFailureReason::NotFusableOp;
    result.failedValue = value;
    return result;
  }

  // Recursively trace all operands
  for (mlir::Value operand : getElementwiseOperands(defOp)) {
    auto operandTrace = traceFusionToRoots(operand);
    if (operandTrace.failureReason != TraceFailureReason::Success) {
      return operandTrace;
    }
    // Merge roots and ops (SmallSetVector handles deduplication)
    for (mlir::Value root : operandTrace.rootInputs) {
      result.rootInputs.insert(root);
    }
    for (mlir::Operation *op : operandTrace.opsInOrder) {
      result.opsInOrder.insert(op);
    }
  }

  // Add this op at the end (after all its dependencies)
  result.opsInOrder.insert(defOp);

  return result;
}

llvm::StringRef describeTraceFailure(TraceFailureReason reason) {
  switch (reason) {
  case TraceFailureReason::Success:
    return "success";
  case TraceFailureReason::NotCBAttached:
    return "value is not attached to a circular buffer";
  case TraceFailureReason::NotFusableOp:
    return "cannot trace through non-fusable op";
  }
  llvm_unreachable("unhandled TraceFailureReason");
}

//===----------------------------------------------------------------------===//
// Loop grouping for L1 accumulation and init selection
//===----------------------------------------------------------------------===//

namespace ttk = mlir::tt::ttkernel;

llvm::SmallDenseSet<Value, 2> getPackTileCBs(scf::ForOp loop) {
  llvm::SmallDenseSet<Value, 2> cbs;
  loop->walk([&](ttk::PackTileOp packOp) { cbs.insert(packOp.getOutCb()); });
  return cbs;
}

bool sharePackCB(scf::ForOp loopA, scf::ForOp loopB) {
  auto cbsA = getPackTileCBs(loopA);
  auto cbsB = getPackTileCBs(loopB);
  for (auto cb : cbsA) {
    if (cbsB.contains(cb)) {
      return true;
    }
  }
  return false;
}

SmallVector<LoopGroup> collectLoopGroups(
    ArrayRef<scf::ForOp> l1AccLoops,
    const llvm::SmallDenseMap<Operation *, Operation *> &enablePointPerLoop) {
  // Find the outermost annotated ancestor of a loop.
  auto findRoot = [](scf::ForOp loop) -> scf::ForOp {
    scf::ForOp outermost = loop;
    for (Operation *parent = loop->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (auto parentFor = dyn_cast<scf::ForOp>(parent)) {
        if (parentFor->hasAttr(kL1AccLoopAttrName) ||
            parentFor->hasAttr(kReductionLoopAttrName)) {
          outermost = parentFor;
        }
      }
    }
    return outermost;
  };

  SmallVector<LoopGroup> groups;
  llvm::SmallDenseSet<Operation *> assigned;

  for (auto loop : l1AccLoops) {
    if (!enablePointPerLoop.count(loop.getOperation())) {
      continue;
    }
    if (assigned.contains(loop.getOperation())) {
      continue;
    }

    scf::ForOp rootLoop = findRoot(loop);

    LoopGroup group;
    group.rootLoop = rootLoop;
    group.loops.push_back(loop);
    assigned.insert(loop.getOperation());

    // Collect sibling annotated loops that share a pack CB target.
    // sharePackCB walks recursively, so for nested loops (rootLoop
    // wrapping loop), it finds pack_tile ops inside the inner loop.
    for (Operation *op = rootLoop->getNextNode(); op; op = op->getNextNode()) {
      if (isa<ttk::CBPushBackOp>(op)) {
        break;
      }
      auto sibling = dyn_cast<scf::ForOp>(op);
      if (!sibling) {
        continue;
      }
      if (!sibling->hasAttr(kL1AccLoopAttrName) &&
          !sibling->hasAttr(kReductionLoopAttrName)) {
        break;
      }
      if (!sharePackCB(rootLoop, sibling)) {
        break;
      }
      group.loops.push_back(sibling);
      assigned.insert(sibling.getOperation());
    }

    // Find scope end: scan forward from rootLoop past grouped siblings,
    // init ops between them, and trailing cb_push_back ops. Only stop
    // at a non-grouped ForOp or a cb_reserve_back.
    group.scopeEnd = rootLoop;
    for (Operation *op = rootLoop->getNextNode(); op; op = op->getNextNode()) {
      if (isa<ttk::CBPushBackOp>(op)) {
        group.scopeEnd = op;
      } else if (isa<ttk::CBReserveBackOp>(op)) {
        break;
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (!assigned.contains(forOp)) {
          break;
        }
      }
    }

    groups.push_back(std::move(group));
  }

  return groups;
}

//===----------------------------------------------------------------------===//
// Compiler-allocated DFB utilities
//===----------------------------------------------------------------------===//

int32_t getNextAvailableDFBIndex(ModuleOp mod) {
  int32_t maxIndex = -1;

  mod->walk([&](BindCBOp bindOp) {
    int64_t idx = bindOp.getCbIndex().getSExtValue();
    assert(idx < kMaxCircularBuffers && "CB index exceeds hardware limit");
    if (static_cast<int32_t>(idx) > maxIndex) {
      maxIndex = static_cast<int32_t>(idx);
    }
  });

  return maxIndex + 1;
}

} // namespace mlir::tt::ttl
