// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::ttl {

std::optional<BcastType> getTileBroadcastType(ArrayRef<int64_t> dims,
                                              int64_t rank) {
  llvm::SmallDenseSet<int64_t> normalizedDims = normalizeDimsToSet(dims, rank);
  bool broadcastsInnermost = rank >= 1 && normalizedDims.contains(rank - 1);
  bool broadcastsSecondInnermost =
      rank >= 2 && normalizedDims.contains(rank - 2);
  if (broadcastsInnermost && broadcastsSecondInnermost) {
    return BcastType::Scalar;
  }
  if (broadcastsSecondInnermost) {
    return BcastType::Row;
  }
  if (broadcastsInnermost) {
    return BcastType::Col;
  }
  return std::nullopt;
}

FailureOr<ttkernel::ReduceDim> getReduceDimension(ArrayRef<int64_t> dims,
                                                  int64_t rank) {
  if (rank != 2) {
    return failure();
  }
  llvm::SmallDenseSet<int64_t> normalizedDims = normalizeDimsToSet(dims, rank);
  // TTKernel names the surviving orientation: reducing height uses a column
  // reduction, while reducing width uses a row reduction.
  bool reducesHeight = normalizedDims.contains(0);
  bool reducesWidth = normalizedDims.contains(1);
  if (reducesHeight && reducesWidth) {
    return ttkernel::ReduceDim::Scalar;
  }
  if (reducesHeight) {
    return ttkernel::ReduceDim::Col;
  }
  if (reducesWidth) {
    return ttkernel::ReduceDim::Row;
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// DST access interface defaults
//===----------------------------------------------------------------------===//

static bool isTileValue(Value value) {
  return isa<ttcore::TileType>(value.getType());
}

/// A block matmul reports one output slot before block expansion and an `M*N`
/// range after `LowerMatmulCompute` has replaced tile operands with tensors.
static int64_t getMatmulBlockOutputTileCount(TileMatmulBlockOp op) {
  auto lhsType = dyn_cast<RankedTensorType>(op.getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(op.getRhs().getType());
  if (!lhsType || !rhsType || lhsType.getRank() < 2 || rhsType.getRank() < 2 ||
      !lhsType.hasStaticShape() || !rhsType.hasStaticShape()) {
    return 1;
  }
  return lhsType.getDimSize(0) * rhsType.getDimSize(1);
}

/// Interface defaults assert for missing DST operands because callers use this
/// after DST assignment, where unresolved tile residency is invalid IR.
static void appendDstOperandFootprint(SmallVectorImpl<DstFootprint> &footprints,
                                      Value operand) {
  if (!isTileValue(operand)) {
    return;
  }
  FailureOr<DstFootprint> footprint = getDstFootprint(operand);
  assert(succeeded(footprint) && "DST operand has no DST footprint");
  footprints.push_back(*footprint);
}

/// Ordinary DST-input tile ops read tile operands from their producer slots.
/// FPU-eligible strategy-dependent binary ops read from DFBs instead.
SmallVector<DstFootprint, 2> getDefaultDstReadFootprints(Operation *op) {
  SmallVector<DstFootprint, 2> footprints;
  if (isa<CopyTileOp, DstIndexOp>(op)) {
    return footprints;
  }
  if (auto store = dyn_cast<TileStoreOp>(op)) {
    appendDstOperandFootprint(footprints, store.getTile());
    return footprints;
  }
  if (op->hasTrait<TTLDSTInputsTrait>() ||
      (op->hasTrait<TTLStrategyDependentBinaryOpTrait>() &&
       !isFPUEligibleBinaryOp(op))) {
    for (Value operand : op->getOperands()) {
      appendDstOperandFootprint(footprints, operand);
    }
  }
  return footprints;
}

/// Most tile ops write one explicit `dst_index`; block matmul is the current
/// multi-slot writer and stores only read DST for packing.
SmallVector<DstFootprint, 2> getDefaultDstWriteFootprints(Operation *op) {
  if (isa<TileStoreOp, DstIndexOp>(op)) {
    return {};
  }
  if (auto matmul = dyn_cast<TileMatmulBlockOp>(op)) {
    return {{matmul.getDstIndex(), getMatmulBlockOutputTileCount(matmul)}};
  }
  if (auto dstIndex = getTileOpDstIndex(op)) {
    return {{*dstIndex, 1}};
  }
  return {};
}

/// Result residency is separate from writes so index-like ops can name a DST
/// slot without emitting a write.
FailureOr<DstFootprint> getDefaultResultDstFootprint(Operation *op,
                                                     Value result) {
  if (!llvm::is_contained(op->getResults(), result) || !isTileValue(result)) {
    return failure();
  }
  if (auto index = dyn_cast<DstIndexOp>(op)) {
    return DstFootprint{index.getDstIndex(), 1};
  }
  if (auto matmul = dyn_cast<TileMatmulBlockOp>(op)) {
    return DstFootprint{matmul.getDstIndex(),
                        getMatmulBlockOutputTileCount(matmul)};
  }
  if (auto dstIndex = getTileOpDstIndex(op)) {
    return DstFootprint{*dstIndex, 1};
  }
  return failure();
}

/// Resolve a tile SSA value through its defining op's DST access interface.
FailureOr<DstFootprint> getDstFootprint(Value value) {
  Operation *definingOp = value.getDefiningOp();
  if (!definingOp) {
    return failure();
  }
  auto dstAccess = dyn_cast<DstAccessOpInterface>(definingOp);
  if (!dstAccess) {
    return failure();
  }
  return dstAccess.getResultDstFootprint(value);
}

/// Consumers that lower to TTKernel source operands require exactly one
/// concrete DST slot.
FailureOr<int64_t> getSingleConstantDstIndex(Value value) {
  FailureOr<DstFootprint> footprint = getDstFootprint(value);
  if (failed(footprint) || footprint->tileCount != 1) {
    return failure();
  }
  std::optional<int64_t> index = foldIndexToConstant(footprint->baseIndex);
  if (!index) {
    return failure();
  }
  return *index;
}

/// Scheduler hazards operate on concrete slots after DST assignment.
FailureOr<SmallVector<int64_t>> getConstantDstIndices(DstFootprint footprint) {
  std::optional<int64_t> base = foldIndexToConstant(footprint.baseIndex);
  if (!base || footprint.tileCount < 0) {
    return failure();
  }
  SmallVector<int64_t> indices;
  indices.reserve(footprint.tileCount);
  for (int64_t offset = 0; offset < footprint.tileCount; ++offset) {
    indices.push_back(*base + offset);
  }
  return indices;
}

static FailureOr<SmallVector<int64_t>>
getConstantDstIndices(ArrayRef<DstFootprint> footprints) {
  SmallVector<int64_t> indices;
  for (DstFootprint footprint : footprints) {
    FailureOr<SmallVector<int64_t>> expanded = getConstantDstIndices(footprint);
    if (failed(expanded)) {
      return failure();
    }
    llvm::append_range(indices, *expanded);
  }
  return indices;
}

FailureOr<SmallVector<int64_t>> getConstantDstReadIndices(Operation *op) {
  auto dstAccess = dyn_cast<DstAccessOpInterface>(op);
  if (!dstAccess) {
    return SmallVector<int64_t>{};
  }
  return getConstantDstIndices(dstAccess.getDstReadFootprints());
}

FailureOr<SmallVector<int64_t>> getConstantDstWriteIndices(Operation *op) {
  auto dstAccess = dyn_cast<DstAccessOpInterface>(op);
  if (!dstAccess) {
    return SmallVector<int64_t>{};
  }
  return getConstantDstIndices(dstAccess.getDstWriteFootprints());
}

//===----------------------------------------------------------------------===//
// Tile operation classification
//===----------------------------------------------------------------------===//

TileOpCategory classifyTileOp(Operation *op) {
  if (isa<DstIndexOp>(op)) {
    return TileOpCategory::DstIndex;
  }
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

  if (isFPUEligibleBinaryOp(op)) {
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

FusionTraceResult traceFusionToRoots(
    mlir::Value value,
    llvm::function_ref<bool(mlir::OpOperand &)> isMaterializationPlanned) {
  FusionTraceResult result;

  // A DFB-attached value is an available input to the fused computation.
  if (getAttachedCB(value)) {
    result.rootInputs.insert(value);
    result.lifetimeRootInputs.insert(value);
    return result;
  }

  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    result.failureReason = TraceFailureReason::NotCBAttached;
    result.failedValue = value;
    return result;
  }

  // BlockBroadcastOp is a fusion leaf because its input must be DFB-attached.
  if (auto bcastOp = llvm::dyn_cast<BlockBroadcastOp>(defOp)) {
    mlir::OpOperand &inputOperand = bcastOp->getOpOperand(0);
    mlir::Value bcastInput = inputOperand.get();
    bool isInputMaterialized = isMaterializationPlanned(inputOperand);
    if (isInputMaterialized || getAttachedCB(bcastInput)) {
      result.rootInputs.insert(bcastInput);
      if (!isInputMaterialized) {
        result.lifetimeRootInputs.insert(bcastInput);
      }
      result.opsInOrder.insert(defOp);
      return result;
    }
    // The broadcast cannot be formed until its input is materialized.
    result.failureReason = TraceFailureReason::NotCBAttached;
    result.failedValue = bcastInput;
    result.failedOperand = &inputOperand;
    return result;
  }

  // MatmulOp is a fusion leaf because both inputs must be DFB-attached.
  if (auto matmulOp = llvm::dyn_cast<MatmulOp>(defOp)) {
    mlir::OpOperand &lhsOperand = matmulOp->getOpOperand(0);
    mlir::OpOperand &rhsOperand = matmulOp->getOpOperand(1);
    mlir::Value lhs = lhsOperand.get();
    mlir::Value rhs = rhsOperand.get();
    bool isLhsMaterialized = isMaterializationPlanned(lhsOperand);
    bool isRhsMaterialized = isMaterializationPlanned(rhsOperand);
    bool lhsAvailable = isLhsMaterialized || getAttachedCB(lhs);
    bool rhsAvailable = isRhsMaterialized || getAttachedCB(rhs);
    if (lhsAvailable && rhsAvailable) {
      result.rootInputs.insert(lhs);
      result.rootInputs.insert(rhs);
      if (!isLhsMaterialized) {
        result.lifetimeRootInputs.insert(lhs);
      }
      if (!isRhsMaterialized) {
        result.lifetimeRootInputs.insert(rhs);
      }
      result.opsInOrder.insert(defOp);
      return result;
    }
    // The matmul cannot be formed until both inputs are materialized.
    result.failureReason = TraceFailureReason::NotCBAttached;
    result.failedValue = lhsAvailable ? rhs : lhs;
    result.failedOperand = lhsAvailable ? &rhsOperand : &lhsOperand;
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

  // Recursively trace every elementwise operand not replaced by a planned
  // materialization.
  unsigned numElementwiseOperands = getElementwiseOperands(defOp).size();
  for (unsigned operandIndex = 0; operandIndex < numElementwiseOperands;
       ++operandIndex) {
    mlir::OpOperand &operand = defOp->getOpOperand(operandIndex);
    if (isMaterializationPlanned(operand)) {
      result.rootInputs.insert(operand.get());
      continue;
    }
    auto operandTrace =
        traceFusionToRoots(operand.get(), isMaterializationPlanned);
    if (operandTrace.failureReason != TraceFailureReason::Success) {
      if (!operandTrace.failedOperand) {
        operandTrace.failedOperand = &operand;
      }
      return operandTrace;
    }
    // Merge roots and ops (SmallSetVector handles deduplication)
    for (mlir::Value root : operandTrace.rootInputs) {
      result.rootInputs.insert(root);
    }
    for (mlir::Value root : operandTrace.lifetimeRootInputs) {
      result.lifetimeRootInputs.insert(root);
    }
    for (mlir::Operation *op : operandTrace.opsInOrder) {
      result.opsInOrder.insert(op);
    }
  }

  // Add this op at the end (after all its dependencies)
  result.opsInOrder.insert(defOp);

  return result;
}

FusionTraceResult traceFusionToRoots(mlir::Value value) {
  return traceFusionToRoots(value, [](mlir::OpOperand &) { return false; });
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
    auto groupPackCBs = getPackTileCBs(rootLoop);

    // A bare non-annotated scf.for between siblings does not break the
    // group unless its body packs to one of the group's pack CBs — such
    // a pack runs with L1 acc disabled and would overwrite the shared
    // L1 slot before the next sibling accumulates onto it.
    auto bareForMutatesSharedCB = [&](scf::ForOp forOp) {
      auto innerCBs = getPackTileCBs(forOp);
      return llvm::any_of(innerCBs,
                          [&](Value cb) { return groupPackCBs.contains(cb); });
    };

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
        if (bareForMutatesSharedCB(sibling)) {
          break;
        }
        continue;
      }
      if (!sharePackCB(rootLoop, sibling)) {
        break;
      }
      group.loops.push_back(sibling);
      assigned.insert(sibling.getOperation());
    }

    // Find scope end: scan forward from rootLoop past grouped siblings,
    // init ops between them, and trailing cb_push_back ops. Stop at a
    // cb_reserve_back, any annotated scf.for that is not in this group
    // (belongs to a different scope), or a bare scf.for that packs to
    // one of the group's pack CBs.
    group.scopeEnd = rootLoop;
    for (Operation *op = rootLoop->getNextNode(); op; op = op->getNextNode()) {
      if (isa<ttk::CBPushBackOp>(op)) {
        group.scopeEnd = op;
      } else if (isa<ttk::CBReserveBackOp>(op)) {
        break;
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        if (assigned.contains(forOp)) {
          continue;
        }
        bool isAnnotated = forOp->hasAttr(kL1AccLoopAttrName) ||
                           forOp->hasAttr(kReductionLoopAttrName);
        if (isAnnotated || bareForMutatesSharedCB(forOp)) {
          break;
        }
      }
    }

    groups.push_back(std::move(group));
  }

  return groups;
}

} // namespace mlir::tt::ttl
