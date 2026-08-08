// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttl {

FailureOr<ttcore::TileType> getTileType(Type type) {
  if (auto tileType = dyn_cast<ttcore::TileType>(type)) {
    return tileType;
  }
  auto tensorType = dyn_cast<RankedTensorType>(type);
  if (!tensorType) {
    return failure();
  }
  auto tileType = dyn_cast<ttcore::TileType>(tensorType.getElementType());
  if (!tileType) {
    return failure();
  }
  return tileType;
}

LogicalResult verifyTypecastTileTypes(ttcore::TileType inputType,
                                      ttcore::TileType resultType,
                                      std::string &failureReason) {
  failureReason.clear();
  llvm::raw_string_ostream diagnostic(failureReason);
  if (inputType.getShape() != resultType.getShape()) {
    diagnostic << "input and result tile shapes must match, but got input: "
               << inputType << ", result: " << resultType;
    return failure();
  }
  if (!ttcore::isFloat(inputType.getDataType()) ||
      !ttcore::isFloat(resultType.getDataType())) {
    diagnostic
        << "only supports floating-point tile data types, but got input: "
        << inputType << ", result: " << resultType;
    return failure();
  }
  return success();
}

FailureOr<int64_t> getDFBId(Value cb) {
  auto bindOp = getDFBDeclaration(cb);
  if (!bindOp) {
    return failure();
  }
  auto dfbId = bindOp.getDfbId();
  if (!dfbId.has_value()) {
    return failure();
  }
  return dfbId->getSExtValue();
}

FailureOr<uint64_t> getDFBPagesPerBlock(CircularBufferType type) {
  uint64_t pagesPerBlock = 1;
  for (int64_t dimension : type.getShape()) {
    if (dimension <= 0) {
      return failure();
    }
    std::optional<uint64_t> product = llvm::checkedMulUnsigned(
        pagesPerBlock, static_cast<uint64_t>(dimension));
    if (!product) {
      return failure();
    }
    pagesPerBlock = *product;
  }
  return pagesPerBlock;
}

FailureOr<uint64_t> getDFBPageSizeBytes(CircularBufferType type) {
  Type elementType = type.getElementType();
  if (auto tileType = dyn_cast<ttcore::TileType>(elementType)) {
    return tileType.getSizeBytes();
  }
  if (!elementType.isIntOrFloat()) {
    return failure();
  }
  uint64_t bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0) {
    return failure();
  }
  return bitWidth / 8;
}

LogicalResult verifyDFBOperandIdentities(
    ModuleOp moduleOp, StringRef consumerPass,
    llvm::function_ref<bool(Operation *)> operationFilter,
    llvm::function_ref<FailureOr<int64_t>(Value)> identityResolver,
    StringRef operandDescription, DFBIdentityRequirement requirement) {
  WalkResult result = moduleOp.walk([&](Operation *operation) {
    if (!operationFilter(operation)) {
      return WalkResult::advance();
    }
    for (Value operand : operation->getOperands()) {
      if (!isa<CircularBufferType>(operand.getType())) {
        continue;
      }
      if (succeeded(identityResolver(operand))) {
        continue;
      }
      InFlightDiagnostic diagnostic = operation->emitOpError();
      diagnostic << "`" << consumerPass << "` requires every "
                 << operandDescription
                 << " operand to resolve to `ttl.bind_cb`";
      if (requirement == DFBIdentityRequirement::Finalized) {
        diagnostic << " with `dfb_id` after finalization";
      }
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult verifyResolvedDFBIdentities(ModuleOp moduleOp,
                                          StringRef consumerPass) {
  bool hasAllocationMetadata = moduleOp->hasAttr(kDFBAllocationsAttrName);
  bool hasDFB = false;
  WalkResult result =
      moduleOp.walk([&](Operation *nestedOperation) {
        if (!isa<BindCBOp, CBReserveOp, CBPushOp, CBWaitOp, CBPopOp>(
                nestedOperation)) {
          return WalkResult::advance();
        }
        hasDFB = true;
        if (!hasAllocationMetadata) {
          return WalkResult::interrupt();
        }
        if (auto bindOp = dyn_cast<BindCBOp>(nestedOperation);
            bindOp && !bindOp.getDfbId().has_value()) {
          bindOp.emitOpError()
              << "`" << consumerPass
              << "` requires every `ttl.bind_cb` to have `dfb_id` after "
                 "finalization";
          return WalkResult::interrupt();
        }

        return WalkResult::advance();
      });

  if (!hasDFB) {
    return success();
  }
  if (!hasAllocationMetadata) {
    moduleOp.emitOpError()
        << "`" << consumerPass
        << "` requires finalized DFB allocation metadata; run "
           "`ttl-finalize-dfb-indices` first";
    return failure();
  }
  if (result.wasInterrupted()) {
    return failure();
  }
  return verifyDFBOperandIdentities(
      moduleOp, consumerPass,
      [](Operation *operation) {
        return isa<CBReserveOp, CBPushOp, CBWaitOp, CBPopOp>(operation);
      },
      getDFBId, "DFB lifecycle", DFBIdentityRequirement::Finalized);
}

LogicalResult verifyMatmulTileTypes(ttcore::TileType lhsType,
                                    ttcore::TileType rhsType,
                                    ttcore::TileType resultType,
                                    bool transposeRhs,
                                    std::string &failureReason) {
  failureReason.clear();
  llvm::raw_string_ostream diagnostic(failureReason);
  if (lhsType.getDataType() != rhsType.getDataType()) {
    diagnostic << "element data type mismatch: lhs has " << lhsType
               << " but rhs has " << rhsType;
    return failure();
  }
  if (resultType.getDataType() != lhsType.getDataType()) {
    diagnostic << "result element data type " << resultType
               << " must match input element data type " << lhsType;
    return failure();
  }

  int64_t rhsK = transposeRhs ? rhsType.getWidth() : rhsType.getHeight();
  if (lhsType.getWidth() != rhsK) {
    diagnostic << "tile K dimension mismatch: lhs tile width "
               << lhsType.getWidth() << " does not match rhs tile "
               << (transposeRhs ? "width " : "height ") << rhsK;
    return failure();
  }

  int64_t expectedResultWidth =
      transposeRhs ? rhsType.getHeight() : rhsType.getWidth();
  if (resultType.getHeight() != lhsType.getHeight() ||
      resultType.getWidth() != expectedResultWidth) {
    diagnostic << "result tile dimensions " << resultType.getHeight() << "x"
               << resultType.getWidth() << " do not match expected "
               << lhsType.getHeight() << "x" << expectedResultWidth;
    return failure();
  }
  return success();
}

/// FPU binary execution requires both operands to address the same tile
/// coordinates.
static bool hasMatchingFPUInputIndices(Operation *operation) {
  assert(operation->getNumOperands() >= 2 &&
         "binary tile op with execution alternatives must have two data "
         "operands");
  Value lhs = operation->getOperand(0);
  Value rhs = operation->getOperand(1);

  if (auto lhsArgument = dyn_cast<BlockArgument>(lhs)) {
    auto rhsArgument = dyn_cast<BlockArgument>(rhs);
    if (!rhsArgument || lhsArgument.getOwner() != rhsArgument.getOwner()) {
      return false;
    }
    auto computeOp =
        dyn_cast_or_null<ComputeOp>(lhsArgument.getOwner()->getParentOp());
    if (!computeOp) {
      return false;
    }
    unsigned numInputs = computeOp.getNumInputs();
    if (lhsArgument.getArgNumber() >= numInputs ||
        rhsArgument.getArgNumber() >= numInputs) {
      return false;
    }
    auto indexingMaps = computeOp.getIndexingMapsArray();
    return indexingMaps[lhsArgument.getArgNumber()] ==
           indexingMaps[rhsArgument.getArgNumber()];
  }

  auto lhsExtract = lhs.getDefiningOp<tensor::ExtractOp>();
  auto rhsExtract = rhs.getDefiningOp<tensor::ExtractOp>();
  return lhsExtract && rhsExtract &&
         lhsExtract.getIndices() == rhsExtract.getIndices();
}

SmallVector<TileExecutionStrategy, 2>
getDefaultLegalTileExecutionStrategies(Operation *operation) {
  if (!operation->hasTrait<TTLStrategyDependentBinaryOpTrait>()) {
    return {};
  }

  SmallVector<TileExecutionStrategy, 2> strategies{TileExecutionStrategy::SFPU};
  auto resultType =
      dyn_cast<ttcore::TileType>(operation->getResult(0).getType());
  if (resultType && ttcore::isFloat(resultType.getDataType()) &&
      hasMatchingFPUInputIndices(operation)) {
    strategies.insert(strategies.begin(), TileExecutionStrategy::FPU);
  }
  return strategies;
}

FailureOr<TileExecutionInfo>
getDefaultTileExecutionInfo(Operation *operation,
                            std::optional<TileExecutionStrategy> strategy) {
  TileExecutionInfo info;
  info.operandRoutes.assign(operation->getNumOperands(),
                            TileOperandRoute::None);
  info.dstOperandsMaterializedByOperation.resize(operation->getNumOperands());
  info.resultInDst = operation->hasTrait<TTLDstResultOpTrait>();

  if (isa<CopyTileOp>(operation)) {
    info.primitive = TilePrimitive::Copy;
    info.operandRoutes[0] = TileOperandRoute::DataflowBuffer;
    return info;
  }
  if (isa<CopyDstOp>(operation)) {
    info.primitive = TilePrimitive::Copy;
    info.operandRoutes[0] = TileOperandRoute::Dst;
    return info;
  }
  if (isa<DstIndexOp>(operation)) {
    info.primitive = TilePrimitive::DstIndex;
    return info;
  }
  if (isa<TileStoreOp>(operation)) {
    info.primitive = TilePrimitive::Store;
    info.operandRoutes[0] = TileOperandRoute::Dst;
    return info;
  }
  if (auto broadcast = dyn_cast<TileBcastOp>(operation)) {
    switch (broadcast.getBcastType()) {
    case BcastType::Col:
      info.primitive = TilePrimitive::BroadcastColumn;
      break;
    case BcastType::Row:
      info.primitive = TilePrimitive::BroadcastRow;
      break;
    case BcastType::Scalar:
      info.primitive = TilePrimitive::BroadcastScalar;
      break;
    }
    info.operandRoutes[0] = TileOperandRoute::DataflowBuffer;
    return info;
  }
  if (auto reduce = dyn_cast<TileReduceOp>(operation)) {
    info.primitive = TilePrimitive::Reduce;
    info.operandRoutes[0] = TileOperandRoute::DataflowBuffer;
    info.operandRoutes[1] = TileOperandRoute::DataflowBuffer;
    switch (reduce.getReduceDim()) {
    case ttkernel::ReduceDim::Row:
      info.fullFp32Accumulation = FullFp32AccumulationKind::ReduceRow;
      break;
    case ttkernel::ReduceDim::Col:
      info.fullFp32Accumulation = FullFp32AccumulationKind::ReduceColumn;
      break;
    case ttkernel::ReduceDim::Scalar:
      info.fullFp32Accumulation = FullFp32AccumulationKind::ReduceScalar;
      break;
    }
    return info;
  }
  if (isa<TileTransposeOp>(operation)) {
    info.primitive = TilePrimitive::Transpose;
    info.operandRoutes[0] = TileOperandRoute::DataflowBuffer;
    return info;
  }
  if (isa<TileFillOp>(operation)) {
    info.primitive = TilePrimitive::Fill;
    return info;
  }
  if (auto matmul = dyn_cast<TileMatmulBlockOp>(operation)) {
    info.primitive = TilePrimitive::Matmul;
    info.operandRoutes[0] = TileOperandRoute::DataflowBuffer;
    info.operandRoutes[1] = TileOperandRoute::DataflowBuffer;
    if (matmul.getAccumulator()) {
      info.operandRoutes[2] = TileOperandRoute::Dst;
      info.dstOperandsMaterializedByOperation.set(2);
    }
    info.fullFp32Accumulation = FullFp32AccumulationKind::Matmul;
    info.accumulatesIntoDst = true;
    return info;
  }
  if (operation->hasTrait<TTLStrategyDependentBinaryOpTrait>()) {
    if (!strategy) {
      return failure();
    }
    info.primitive = TilePrimitive::ElementwiseBinary;
    TileOperandRoute route = *strategy == TileExecutionStrategy::FPU
                                 ? TileOperandRoute::DataflowBuffer
                                 : TileOperandRoute::Dst;
    info.operandRoutes[0] = route;
    info.operandRoutes[1] = route;
    info.accumulatesIntoDst = *strategy == TileExecutionStrategy::FPU;
    return info;
  }
  if (operation->hasTrait<TTLTileBinaryOpTrait>()) {
    info.primitive = TilePrimitive::ElementwiseBinary;
    info.operandRoutes[0] = TileOperandRoute::Dst;
    info.operandRoutes[1] = TileOperandRoute::Dst;
    return info;
  }
  if (operation->hasTrait<TTLTileUnaryOpTrait>()) {
    info.primitive = TilePrimitive::ElementwiseUnary;
    info.operandRoutes[0] = TileOperandRoute::Dst;
    return info;
  }
  return failure();
}

LogicalResult verifyTileExecutionInfo(Operation *operation,
                                      const TileExecutionInfo &info) {
  if (info.primitive == TilePrimitive::Unknown) {
    operation->emitOpError("does not define a tile execution primitive");
    return failure();
  }
  if (info.operandRoutes.size() != operation->getNumOperands()) {
    operation->emitOpError() << "defines " << info.operandRoutes.size()
                             << " tile operand routes for "
                             << operation->getNumOperands() << " operands";
    return failure();
  }
  if (info.dstOperandsMaterializedByOperation.size() !=
      operation->getNumOperands()) {
    operation->emitOpError()
        << "defines " << info.dstOperandsMaterializedByOperation.size()
        << " DST operand materialization entries for "
        << operation->getNumOperands() << " operands";
    return failure();
  }
  return success();
}

FailureOr<TileExecutionStrategy>
getSelectedTileExecutionStrategy(Operation *operation) {
  auto strategyAttr = operation->getAttrOfType<TileExecutionStrategyAttr>(
      kTileExecutionStrategyAttrName);
  if (!strategyAttr) {
    return failure();
  }
  return strategyAttr.getValue();
}

FailureOr<TileExecutionInfo>
getSelectedTileExecutionInfo(Operation *operation) {
  auto executionOp = dyn_cast<TileExecutionOpInterface>(operation);
  if (!executionOp) {
    return failure();
  }
  if (executionOp.getLegalExecutionStrategies().empty()) {
    return executionOp.getTileExecutionInfo(std::nullopt);
  }
  FailureOr<TileExecutionStrategy> strategy =
      getSelectedTileExecutionStrategy(operation);
  if (failed(strategy)) {
    return failure();
  }
  return executionOp.getTileExecutionInfo(*strategy);
}

LogicalResult
verifyTileExecutionStrategy(Operation *operation,
                            ArrayRef<TileExecutionStrategy> legalStrategies) {
  Attribute rawStrategy = operation->getAttr(kTileExecutionStrategyAttrName);
  auto strategyAttr = dyn_cast_or_null<TileExecutionStrategyAttr>(rawStrategy);
  if (rawStrategy && !strategyAttr) {
    operation->emitOpError()
        << kTileExecutionStrategyAttrName
        << " must be a #ttl.tile_execution_strategy attribute";
    return failure();
  }
  if (legalStrategies.empty() && strategyAttr) {
    operation->emitOpError()
        << kTileExecutionStrategyAttrName
        << " is only valid on tile operations with execution-strategy "
           "alternatives";
    return failure();
  }
  if (strategyAttr &&
      !llvm::is_contained(legalStrategies, strategyAttr.getValue())) {
    operation->emitOpError() << "explicit " << kTileExecutionStrategyAttrName
                             << " is not legal for its operands";
    return failure();
  }
  return success();
}

/// Return an operand route after all required strategies have been selected.
static TileOperandRoute getRequiredOperandRoute(OpOperand &operand) {
  auto executionOp = dyn_cast<TileExecutionOpInterface>(operand.getOwner());
  if (!executionOp) {
    assert(!isTileComputeOp(operand.getOwner()) &&
           "tile operation must implement TileExecutionOpInterface");
    return TileOperandRoute::None;
  }
  FailureOr<TileExecutionInfo> info =
      getSelectedTileExecutionInfo(operand.getOwner());
  assert(succeeded(info) && "tile execution strategy must be resolved");
  assert(operand.getOperandNumber() < info->operandRoutes.size() &&
         "tile execution semantics must define every operand route");
  return info->operandRoutes[operand.getOperandNumber()];
}

bool isDstInput(OpOperand &operand) {
  return getRequiredOperandRoute(operand) == TileOperandRoute::Dst;
}

bool isDstInputMaterializedByOperation(OpOperand &operand) {
  FailureOr<TileExecutionInfo> info =
      getSelectedTileExecutionInfo(operand.getOwner());
  assert(succeeded(info) && "tile execution strategy must be resolved");
  assert(operand.getOperandNumber() <
             info->dstOperandsMaterializedByOperation.size() &&
         "tile execution semantics must define every DST materialization bit");
  return info->dstOperandsMaterializedByOperation.test(
      operand.getOperandNumber());
}

LogicalResult verifyTileExecutionSemantics(Operation *root) {
  WalkResult walkResult = root->walk([&](Operation *operation) {
    auto executionOp = dyn_cast<TileExecutionOpInterface>(operation);
    if (!executionOp) {
      if (isTileComputeOp(operation)) {
        operation->emitOpError("does not implement TileExecutionOpInterface");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }
    SmallVector<TileExecutionStrategy, 2> legalStrategies =
        executionOp.getLegalExecutionStrategies();
    if (failed(verifyTileExecutionStrategy(operation, legalStrategies))) {
      return WalkResult::interrupt();
    }
    FailureOr<TileExecutionInfo> info = getSelectedTileExecutionInfo(operation);
    if (failed(info)) {
      if (!legalStrategies.empty()) {
        operation->emitOpError()
            << "requires a selected " << kTileExecutionStrategyAttrName
            << " attribute; run ttl-set-compute-kernel-config before DST "
               "assignment, scheduling, or lowering";
      } else {
        operation->emitOpError("has no tile execution semantics");
      }
      return WalkResult::interrupt();
    }
    return failed(verifyTileExecutionInfo(operation, *info))
               ? WalkResult::interrupt()
               : WalkResult::advance();
  });
  return failure(walkResult.wasInterrupted());
}

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

/// Interface defaults require resolved DST operands because callers use this
/// after DST assignment, where unresolved tile residency is invalid IR.
static LogicalResult
appendDstOperandFootprint(SmallVectorImpl<DstFootprint> &footprints,
                          Value operand) {
  if (!isTileValue(operand)) {
    return success();
  }
  FailureOr<DstFootprint> footprint = getDstFootprint(operand);
  if (failed(footprint)) {
    return failure();
  }
  footprints.push_back(*footprint);
  return success();
}

FailureOr<SmallVector<DstFootprint, 2>>
getDefaultDstReadFootprints(Operation *op) {
  SmallVector<DstFootprint, 2> footprints;
  FailureOr<TileExecutionInfo> info = getSelectedTileExecutionInfo(op);
  if (failed(info)) {
    return failure();
  }
  for (OpOperand &operand : op->getOpOperands()) {
    if (info->operandRoutes[operand.getOperandNumber()] !=
        TileOperandRoute::Dst) {
      continue;
    }
    if (failed(appendDstOperandFootprint(footprints, operand.get()))) {
      return failure();
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
  FailureOr<SmallVector<DstFootprint, 2>> footprints =
      dstAccess.getDstReadFootprints();
  if (failed(footprints)) {
    return failure();
  }
  return getConstantDstIndices(*footprints);
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
  if (isa<TileTransposeOp>(op)) {
    return TileOpCategory::Transpose;
  }

  if (op->hasTrait<TTLStrategyDependentBinaryOpTrait>()) {
    FailureOr<TileExecutionStrategy> strategy =
        getSelectedTileExecutionStrategy(op);
    assert(succeeded(strategy) && "tile execution strategy must be resolved");
    return *strategy == TileExecutionStrategy::FPU ? TileOpCategory::FPUBinary
                                                   : TileOpCategory::SFPUBinary;
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
