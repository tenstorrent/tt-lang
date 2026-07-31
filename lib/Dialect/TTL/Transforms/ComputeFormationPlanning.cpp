// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ComputeFormationPlanning.h"

#include "DFBValueLifetimeAnalysis.h"

#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#include <functional>

namespace mlir::tt::ttl {

StringRef getComputeFormationWarningMessage(ComputeFormationWarningKind kind) {
  switch (kind) {
  case ComputeFormationWarningKind::InstrumentationPreventsMatmulAccumulator:
    return "instrumentation changes code generation: matmul-accumulator "
           "folding is disabled because the combined hardware operation "
           "cannot preserve the observation point between ttl.matmul and "
           "ttl.add; the instrumented program uses separate tile operations";
  }
  llvm_unreachable("unknown compute formation warning kind");
}

namespace {

static PlanningResult<SmallVector<StoreOp>, OutputPublicationRejection>
collectStoreUsers(Operation *source) {
  if (source->getNumResults() != 1) {
    return PlanningResult<SmallVector<StoreOp>, OutputPublicationRejection>::
        rejected({source,
                  OutputPublicationRejectionKind::UnsupportedResultCount,
                  "operation must have exactly one result to publish"});
  }

  SmallVector<StoreOp> stores;
  for (OpOperand &use : source->getResult(0).getUses()) {
    if (auto store = dyn_cast<StoreOp>(use.getOwner());
        store && &store.getTensorMutable() == &use) {
      stores.push_back(store);
    }
  }
  if (stores.empty()) {
    return PlanningResult<SmallVector<StoreOp>, OutputPublicationRejection>::
        rejected({source, OutputPublicationRejectionKind::MissingStore,
                  "first result has no ttl.store users"});
  }

  Block *storeBlock = stores.front()->getBlock();
  if (llvm::any_of(stores, [&](StoreOp store) {
        return store->getBlock() != storeBlock;
      })) {
    return PlanningResult<SmallVector<StoreOp>, OutputPublicationRejection>::
        rejected({source,
                  OutputPublicationRejectionKind::StoresInDifferentBlocks,
                  "output stores are in different blocks"});
  }

  llvm::sort(stores, [](StoreOp lhs, StoreOp rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  return PlanningResult<SmallVector<StoreOp>,
                        OutputPublicationRejection>::planned(std::move(stores));
}

/// Finds the push that publishes `store` before another reserve advances the
/// same DFB producer pointer.
static std::optional<CBPushOp> findPushAfterStore(StoreOp store, Value dfb) {
  for (Operation *operation = store->getNextNode(); operation;
       operation = operation->getNextNode()) {
    if (auto push = dyn_cast<CBPushOp>(operation)) {
      if (push.getCb() == dfb) {
        return push;
      }
    }
    if (auto reserve = dyn_cast<CBReserveOp>(operation)) {
      if (reserve.getCb() == dfb) {
        return std::nullopt;
      }
    }
  }
  return std::nullopt;
}

static void addUniqueValue(SmallVectorImpl<Value> &values, Value value) {
  if (!llvm::is_contained(values, value)) {
    values.push_back(value);
  }
}

static RankedTensorType getTensorType(Value value) {
  return dyn_cast<RankedTensorType>(value.getType());
}

static AffineMap buildBroadcastInputMap(
    MLIRContext *context, int64_t rank,
    const llvm::SmallDenseSet<int64_t> &broadcastDimensions) {
  // Mapping a broadcast dimension to zero repeats one input tile instead of
  // requiring the input tensor to match the output iteration-domain extent.
  SmallVector<AffineExpr> expressions;
  expressions.reserve(rank);
  for (int64_t dimension = 0; dimension < rank; ++dimension) {
    expressions.push_back(broadcastDimensions.contains(dimension)
                              ? getAffineConstantExpr(0, context)
                              : getAffineDimExpr(dimension, context));
  }
  return AffineMap::get(rank, 0, expressions, context);
}

static AffineMap
buildBroadcastAwareInputMap(MLIRContext *context, RankedTensorType inputType,
                            RankedTensorType outputType, int64_t iterationRank,
                            ArrayRef<AffineExpr> baseExpressions) {
  // Size-one dimensions impose no iteration-domain extent. Replacing their
  // dimension expression with zero makes that constraint explicit in the IR.
  SmallVector<AffineExpr> expressions(baseExpressions);
  for (int64_t dimension = 0; dimension < inputType.getRank(); ++dimension) {
    if (inputType.getDimSize(dimension) == 1 &&
        outputType.getDimSize(dimension) != 1) {
      expressions[dimension] = getAffineConstantExpr(0, context);
    }
  }
  return AffineMap::get(iterationRank, 0, expressions, context);
}

static bool isRelocatableFusedInstrumentation(Operation *operation) {
  // The frontend gives user scopes a `ttl_` prefix. Automatic profiling uses
  // different placement rules that paired markers cannot represent reliably
  // through all compute-lowering transformations.
  if (auto signpost = dyn_cast<SignpostOp>(operation)) {
    return signpost.getName().starts_with("ttl_");
  }
  if (auto dprint = dyn_cast<DPrintOp>(operation)) {
    return dprint.getMode() == "dst" || dprint.getMode() == "tile";
  }
  return false;
}

static bool canPrecedeFusedOperations(Operation *operation) {
  if (auto signpost = dyn_cast<SignpostOp>(operation)) {
    return isRelocatableFusedInstrumentation(operation) && !signpost.getIsEnd();
  }
  return isRelocatableFusedInstrumentation(operation);
}

static bool canFollowFusedStores(Operation *operation) {
  if (auto signpost = dyn_cast<SignpostOp>(operation)) {
    return isRelocatableFusedInstrumentation(operation) && signpost.getIsEnd();
  }
  return isRelocatableFusedInstrumentation(operation);
}

static DenseMap<Operation *, Operation *> pairSignpostsInBlock(Block &block) {
  DenseMap<StringAttr, SmallVector<SignpostOp>> openSignposts;
  DenseMap<Operation *, Operation *> partners;
  for (Operation &operation : block) {
    auto signpost = dyn_cast<SignpostOp>(operation);
    if (!signpost) {
      continue;
    }
    if (!signpost.getIsEnd()) {
      openSignposts[signpost.getNameAttr()].push_back(signpost);
      continue;
    }

    auto open = openSignposts.find(signpost.getNameAttr());
    if (open == openSignposts.end() || open->second.empty()) {
      continue;
    }
    SignpostOp begin = open->second.pop_back_val();
    partners.try_emplace(begin, signpost);
    partners.try_emplace(signpost, begin);
  }
  return partners;
}

static LogicalResult preserveSignpostScopes(
    Block &block, Operation *firstFused, Operation *lastStore,
    SmallVectorImpl<FusedInstrumentationPlacement> &placements,
    std::string &failureReason) {
  DenseSet<Operation *> selected;
  for (const FusedInstrumentationPlacement &placement : placements) {
    selected.insert(placement.operation);
  }

  DenseMap<Operation *, Operation *> partners = pairSignpostsInBlock(block);
  DenseSet<Operation *> keptOutside;
  for (const FusedInstrumentationPlacement &placement : placements) {
    auto signpost = dyn_cast<SignpostOp>(placement.operation);
    if (!signpost) {
      continue;
    }
    Operation *partner = partners.lookup(signpost);
    if (!partner) {
      failureReason =
          "instrumented fusion cannot preserve an unmatched ttl.signpost";
      return failure();
    }
    if (selected.contains(partner)) {
      continue;
    }

    Operation *begin = signpost.getIsEnd() ? partner : signpost.getOperation();
    Operation *end = signpost.getIsEnd() ? signpost.getOperation() : partner;
    if (begin->isBeforeInBlock(firstFused) && lastStore->isBeforeInBlock(end)) {
      // A scope containing every absorbed operation still observes the same
      // events when the complete formed compute remains between its markers.
      keptOutside.insert(signpost);
      continue;
    }

    failureReason =
        "instrumented fusion cannot preserve a partially overlapping "
        "ttl.signpost scope";
    return failure();
  }

  llvm::erase_if(placements,
                 [&](const FusedInstrumentationPlacement &placement) {
                   return keptOutside.contains(placement.operation);
                 });
  return success();
}

static LogicalResult collectFusedInstrumentation(
    const FusionTraceResult &trace, const OutputPublicationPlan &outputs,
    Operation *sink, SmallVectorImpl<FusedInstrumentationPlacement> &placements,
    std::string &failureReason) {
  Block *block = sink->getBlock();
  if (llvm::any_of(outputs.stores, [block](StoreOp store) {
        return store->getBlock() != block;
      })) {
    failureReason = "fused source and output stores must be in one block";
    return failure();
  }

  // Cross-block recomputation uses upstream `isPure`, which requires both
  // speculation safety and absence of memory effects, rather than relying on
  // the current TTL operation whitelist:
  // https://github.com/llvm/llvm-project/blob/4279d524cc78d0bac294bb29257c62665121d9f1/mlir/include/mlir/Interfaces/SideEffectInterfaces.h
  // SSA dominance and the separate DFB availability proof then preserve the
  // producer's value. Instrumentation in another block has no placement
  // relation to the sink, so rejecting it preserves its observations.
  for (Operation *fusedOperation : trace.opsInOrder) {
    if (fusedOperation->getBlock() == block) {
      continue;
    }
    bool canRecompute = isPure(fusedOperation);
    assert(canRecompute &&
           "cross-block fusion admitted an operation that cannot be "
           "recomputed");
    if (!canRecompute) {
      failureReason =
          "fusion across blocks requires speculatable, memory-effect-free "
          "operations";
      return failure();
    }
    if (llvm::any_of(*fusedOperation->getBlock(), [](Operation &operation) {
          return isRelocatableFusedInstrumentation(&operation);
        })) {
      failureReason =
          "fusion across blocks cannot preserve instrumentation surrounding "
          "an absorbed operation";
      return failure();
    }
  }

  DenseSet<Operation *> fusedOperations(trace.opsInOrder.begin(),
                                        trace.opsInOrder.end());
  DenseSet<Operation *> outputStores;
  for (StoreOp store : outputs.stores) {
    outputStores.insert(store);
  }
  Operation *firstFused = nullptr;
  for (Operation &operation : *sink->getBlock()) {
    if (!fusedOperations.contains(&operation)) {
      continue;
    }
    if (!firstFused) {
      firstFused = &operation;
    }
  }
  if (!firstFused) {
    failureReason = "fused expression has no operation in the sink block";
    return failure();
  }

  // Recording each observation relative to an absorbed operation preserves
  // the observed sequence after those operations move into the compute body.
  SmallVector<Operation *> leading;
  for (Operation *operation = firstFused->getPrevNode(); operation;
       operation = operation->getPrevNode()) {
    if (!canPrecedeFusedOperations(operation)) {
      break;
    }
    leading.push_back(operation);
  }
  for (Operation *operation : llvm::reverse(leading)) {
    placements.push_back({operation, nullptr});
  }

  Operation *lastStore = outputs.stores.back();
  Operation *previousMovable = nullptr;
  bool hasUnsupportedOperation = false;
  for (Operation *operation = firstFused;
       operation && operation != lastStore->getNextNode();
       operation = operation->getNextNode()) {
    if (fusedOperations.contains(operation) ||
        outputStores.contains(operation)) {
      previousMovable = operation;
    } else if (isRelocatableFusedInstrumentation(operation)) {
      placements.push_back({operation, previousMovable});
    } else {
      hasUnsupportedOperation = true;
    }
  }

  for (Operation *operation = lastStore->getNextNode(); operation;
       operation = operation->getNextNode()) {
    if (canFollowFusedStores(operation)) {
      placements.push_back({operation, lastStore});
    } else {
      break;
    }
  }

  if (failed(preserveSignpostScopes(*block, firstFused, lastStore, placements,
                                    failureReason))) {
    return failure();
  }

  // Instrumentation cannot move across an operation that remains outside the
  // compute body. Rejecting only instrumented expressions leaves ordinary
  // fusion unaffected while preserving the observed operation sequence.
  if (!placements.empty() && hasUnsupportedOperation) {
    failureReason =
        "instrumented fusion contains an operation that cannot move into "
        "ttl.compute";
    return failure();
  }
  return success();
}

static FailureOr<Type> getResultTileType(Operation *source) {
  if (source->getNumResults() != 1) {
    return failure();
  }
  RankedTensorType tensorType = getTensorType(source->getResult(0));
  if (!tensorType) {
    return failure();
  }
  return ttcore::TileType::get(tensorType.getElementType());
}

static FailureOr<unsigned> findFusedRootInput(const ComputeFormationPlan &plan,
                                              Value value,
                                              FusedInputRole role) {
  for (auto [inputIndex, input] : llvm::enumerate(plan.inputs)) {
    if (input == value && plan.fusedInputRoles[inputIndex] == role) {
      return inputIndex;
    }
  }
  return failure();
}

static LogicalResult buildFusedOperationPlans(ComputeFormationPlan &plan,
                                              std::string &failureReason) {
  DenseSet<Operation *> fusedOperations(plan.trace.opsInOrder.begin(),
                                        plan.trace.opsInOrder.end());
  DenseMap<Operation *, SmallVector<MatmulOp>> foldCandidates;
  for (Operation *operation : plan.trace.opsInOrder) {
    auto matmul = dyn_cast<MatmulOp>(operation);
    if (!matmul || !matmul.getResult().hasOneUse()) {
      continue;
    }
    Operation *user = *matmul.getResult().getUsers().begin();
    Operation *interveningInstrumentation = nullptr;
    if (isa<AddOp>(user) && fusedOperations.contains(user) &&
        matmul->getBlock() == user->getBlock()) {
      for (Operation *operation = matmul->getNextNode(); operation != user;
           operation = operation->getNextNode()) {
        if (isRelocatableFusedInstrumentation(operation)) {
          interveningInstrumentation = operation;
          break;
        }
      }
    }
    if (interveningInstrumentation) {
      plan.warnings.push_back({interveningInstrumentation,
                               ComputeFormationWarningKind::
                                   InstrumentationPreventsMatmulAccumulator});
    } else if (isa<AddOp>(user) && fusedOperations.contains(user)) {
      foldCandidates[user].push_back(matmul);
    }
  }

  // Folding exactly one matmul preserves a concrete accumulator tile. If both
  // add operands are deferred matmuls, neither result exists to initialize the
  // hardware accumulator, so both matmuls remain explicit.
  DenseMap<Operation *, MatmulOp> foldedMatmulByAdd;
  DenseSet<Operation *> deferredMatmuls;
  for (auto &[add, candidates] : foldCandidates) {
    if (candidates.size() == 1) {
      foldedMatmulByAdd.try_emplace(add, candidates.front());
      deferredMatmuls.insert(candidates.front());
    }
  }

  for (Operation *operation : plan.trace.opsInOrder) {
    FusedOperationPlan operationPlan;
    operationPlan.source = operation;
    llvm::append_range(operationPlan.sourceOperands, operation->getOperands());
    auto addOperand = [&](Value value, FusedInputRole role) {
      std::optional<unsigned> rootInputIndex;
      if (plan.trace.rootInputs.contains(value)) {
        FailureOr<unsigned> inputIndex = findFusedRootInput(plan, value, role);
        if (failed(inputIndex)) {
          failureReason =
              "fused root input has no matching affine indexing role";
          return failure();
        }
        rootInputIndex = *inputIndex;
      }
      operationPlan.operands.push_back({value, rootInputIndex});
      return success();
    };
    FailureOr<Type> resultTileType = getResultTileType(operation);
    if (failed(resultTileType)) {
      failureReason = "fused operation result is not a ranked tensor";
      return failure();
    }
    operationPlan.resultTileType = *resultTileType;

    if (auto broadcast = dyn_cast<BlockBroadcastOp>(operation)) {
      RankedTensorType inputType = getTensorType(broadcast.getInput());
      if (!inputType) {
        failureReason = "fused broadcast input is not a ranked tensor";
        return failure();
      }
      if (failed(addOperand(broadcast.getInput(), FusedInputRole::Parallel))) {
        return failure();
      }
      operationPlan.tileBroadcast =
          getTileBroadcastType(broadcast.getDims(), inputType.getRank());
      operationPlan.recipe = operationPlan.tileBroadcast
                                 ? FusedOperationRecipe::TileBroadcast
                                 : FusedOperationRecipe::InterTileBroadcast;
    } else if (auto matmul = dyn_cast<MatmulOp>(operation)) {
      if (failed(addOperand(matmul.getLhs(), FusedInputRole::MatmulLeft)) ||
          failed(addOperand(matmul.getRhs(),
                            matmul.getTransposeRhs()
                                ? FusedInputRole::MatmulTransposedRight
                                : FusedInputRole::MatmulRight))) {
        return failure();
      }
      operationPlan.transposeRhs = matmul.getTransposeRhs();
      operationPlan.recipe = deferredMatmuls.contains(operation)
                                 ? FusedOperationRecipe::DeferredMatmul
                                 : FusedOperationRecipe::Matmul;
    } else if (auto foldedMatmul = foldedMatmulByAdd.lookup(operation)) {
      Value lhs = operation->getOperand(0);
      Value rhs = operation->getOperand(1);
      operationPlan.recipe = FusedOperationRecipe::MatmulAccumulator;
      operationPlan.foldedMatmul = foldedMatmul;
      operationPlan.accumulator = lhs == foldedMatmul.getResult() ? rhs : lhs;
      if (failed(
              addOperand(foldedMatmul.getLhs(), FusedInputRole::MatmulLeft)) ||
          failed(addOperand(foldedMatmul.getRhs(),
                            foldedMatmul.getTransposeRhs()
                                ? FusedInputRole::MatmulTransposedRight
                                : FusedInputRole::MatmulRight)) ||
          failed(addOperand(*operationPlan.accumulator,
                            FusedInputRole::Parallel))) {
        return failure();
      }
      operationPlan.transposeRhs = foldedMatmul.getTransposeRhs();
    } else if (isElementwiseOp(operation) || isa<FillOp>(operation)) {
      for (Value operand : getElementwiseOperands(operation)) {
        if (failed(addOperand(operand, FusedInputRole::Parallel))) {
          return failure();
        }
      }
      operationPlan.recipe = FusedOperationRecipe::TileOperation;
      if (auto multiply = dyn_cast<MulUnaryConstOp>(operation)) {
        operationPlan.constantValue = multiply.getValueAttr();
      } else if (auto fill = dyn_cast<FillOp>(operation)) {
        operationPlan.constantValue = fill.getValueAttr();
      }
    } else {
      failureReason = "fused operation has no tile-level recipe";
      return failure();
    }
    plan.fusedOperations.push_back(std::move(operationPlan));
  }
  return success();
}

static void buildIdentityIterationPlan(ComputeFormationPlan &plan) {
  MLIRContext *context = plan.source->getContext();
  AffineMap identity =
      AffineMap::getMultiDimIdentityMap(plan.resultType.getRank(), context);
  plan.iteration.inputMaps.assign(plan.inputs.size(), identity);
  plan.iteration.outputMap = identity;
  plan.iteration.iteratorTypes.assign(plan.resultType.getRank(),
                                      utils::IteratorType::parallel);
}

static LogicalResult buildFusedIterationPlan(ComputeFormationPlan &plan,
                                             std::string &failureReason) {
  MLIRContext *context = plan.source->getContext();
  DenseMap<Value, SmallVector<FusedInputRole>> inputRoles;
  auto assignInputRole = [&](Value input, FusedInputRole role) {
    SmallVector<FusedInputRole> &roles = inputRoles[input];
    if (!llvm::is_contained(roles, role)) {
      roles.push_back(role);
    }
  };

  for (Operation *operation : plan.trace.opsInOrder) {
    if (auto matmul = dyn_cast<MatmulOp>(operation)) {
      assignInputRole(matmul.getLhs(), FusedInputRole::MatmulLeft);
      assignInputRole(matmul.getRhs(),
                      matmul.getTransposeRhs()
                          ? FusedInputRole::MatmulTransposedRight
                          : FusedInputRole::MatmulRight);
      continue;
    }
    SmallVector<Value> parallelInputs;
    if (auto broadcast = dyn_cast<BlockBroadcastOp>(operation)) {
      parallelInputs.push_back(broadcast.getInput());
    } else {
      llvm::append_range(parallelInputs, getElementwiseOperands(operation));
    }
    for (Value input : parallelInputs) {
      if (plan.trace.rootInputs.contains(input)) {
        assignInputRole(input, FusedInputRole::Parallel);
      }
    }
  }

  bool hasMatmul = llvm::any_of(inputRoles, [](const auto &entry) {
    return llvm::any_of(entry.second, [](FusedInputRole role) {
      return role != FusedInputRole::Parallel;
    });
  });
  plan.inputs.clear();
  plan.fusedInputRoles.clear();
  plan.iteration.inputMaps.clear();
  if (hasMatmul) {
    // Every operation fused with a matmul executes in its [M, N, K] domain.
    // Elementwise inputs use only [M, N], while the matmul operands retain
    // their distinct contraction maps. This prevents subblocking M from also
    // slicing an operand whose first dimension is K.
    AffineExpr dimensionM = getAffineDimExpr(0, context);
    AffineExpr dimensionN = getAffineDimExpr(1, context);
    AffineExpr dimensionK = getAffineDimExpr(2, context);
    AffineMap leftMap = AffineMap::get(3, 0, {dimensionM, dimensionK}, context);
    AffineMap rightMap =
        AffineMap::get(3, 0, {dimensionK, dimensionN}, context);
    AffineMap transposedRightMap =
        AffineMap::get(3, 0, {dimensionN, dimensionK}, context);
    plan.iteration.outputMap =
        AffineMap::get(3, 0, {dimensionM, dimensionN}, context);
    for (Value input : plan.trace.rootInputs) {
      auto roles = inputRoles.find(input);
      if (roles == inputRoles.end()) {
        failureReason = "fused root input has no affine indexing role";
        return failure();
      }
      for (FusedInputRole role : roles->second) {
        plan.inputs.push_back(input);
        plan.fusedInputRoles.push_back(role);
        if (role == FusedInputRole::MatmulLeft) {
          plan.iteration.inputMaps.push_back(leftMap);
        } else if (role == FusedInputRole::MatmulRight) {
          plan.iteration.inputMaps.push_back(rightMap);
        } else if (role == FusedInputRole::MatmulTransposedRight) {
          plan.iteration.inputMaps.push_back(transposedRightMap);
        } else {
          AffineMap inputMap = plan.iteration.outputMap;
          if (RankedTensorType inputType = getTensorType(input);
              inputType && inputType.getRank() == 2) {
            inputMap =
                buildBroadcastAwareInputMap(context, inputType, plan.resultType,
                                            3, {dimensionM, dimensionN});
          }
          plan.iteration.inputMaps.push_back(inputMap);
        }
      }
    }
    plan.iteration.iteratorTypes = {utils::IteratorType::parallel,
                                    utils::IteratorType::parallel,
                                    utils::IteratorType::reduction};
  } else {
    plan.iteration.outputMap =
        AffineMap::getMultiDimIdentityMap(plan.resultType.getRank(), context);
    for (Value input : plan.trace.rootInputs) {
      RankedTensorType inputType = getTensorType(input);
      if (!inputType || inputType.getRank() != plan.resultType.getRank()) {
        failureReason =
            "fused elementwise input rank differs from the result rank";
        return failure();
      }
      SmallVector<AffineExpr> expressions;
      for (int64_t dimension = 0; dimension < plan.resultType.getRank();
           ++dimension) {
        expressions.push_back(getAffineDimExpr(dimension, context));
      }
      plan.inputs.push_back(input);
      plan.fusedInputRoles.push_back(FusedInputRole::Parallel);
      plan.iteration.inputMaps.push_back(
          buildBroadcastAwareInputMap(context, inputType, plan.resultType,
                                      plan.resultType.getRank(), expressions));
    }
    plan.iteration.iteratorTypes.assign(plan.resultType.getRank(),
                                        utils::IteratorType::parallel);
  }

  // A shared iteration dimension has one extent. Rejecting inconsistent
  // extents here prevents a later rewrite from constructing invalid affine
  // indexing and makes the plan independent of pattern application order.
  DenseMap<unsigned, int64_t> iterationDimensionSizes;
  auto recordIterationExtents = [&](AffineMap indexingMap,
                                    RankedTensorType tensorType,
                                    StringRef valueDescription) {
    if (indexingMap.getNumResults() != tensorType.getRank()) {
      failureReason =
          (valueDescription + " indexing map rank differs from tensor rank")
              .str();
      return failure();
    }
    for (unsigned resultIndex = 0; resultIndex < indexingMap.getNumResults();
         ++resultIndex) {
      auto dimension =
          dyn_cast<AffineDimExpr>(indexingMap.getResult(resultIndex));
      if (!dimension) {
        continue;
      }
      unsigned iterationDimension = dimension.getPosition();
      int64_t tensorDimensionSize = tensorType.getDimSize(resultIndex);
      if (ShapedType::isDynamic(tensorDimensionSize)) {
        continue;
      }
      auto [sizeIterator, inserted] = iterationDimensionSizes.try_emplace(
          iterationDimension, tensorDimensionSize);
      if (!inserted && sizeIterator->second != tensorDimensionSize) {
        failureReason = "fused inputs disagree on an iteration dimension";
        return failure();
      }
    }
    return success();
  };

  for (auto [inputIndex, input] : llvm::enumerate(plan.inputs)) {
    RankedTensorType inputType = getTensorType(input);
    if (!inputType) {
      failureReason = "fused input is not a ranked tensor";
      return failure();
    }
    if (failed(recordIterationExtents(plan.iteration.inputMaps[inputIndex],
                                      inputType, "fused input"))) {
      return failure();
    }
  }
  if (failed(recordIterationExtents(plan.iteration.outputMap, plan.resultType,
                                    "fused result"))) {
    return failure();
  }
  return success();
}

static LogicalResult buildOperationSpecificPlan(ComputeFormationPlan &plan,
                                                std::string &failureReason) {
  if (plan.kind == ComputeFormationKind::Elide) {
    plan.recipe = ComputeFormationRecipe::Elide;
    return success();
  }
  if (plan.kind == ComputeFormationKind::Fused) {
    plan.recipe = ComputeFormationRecipe::Fused;
    if (failed(buildFusedIterationPlan(plan, failureReason))) {
      return failure();
    }
    return buildFusedOperationPlans(plan, failureReason);
  }

  if (isa<BlockBroadcastOp>(plan.source)) {
    plan.recipe = ComputeFormationRecipe::BlockBroadcast;
    auto broadcast = cast<BlockBroadcastOp>(plan.source);
    RankedTensorType inputType = getTensorType(broadcast.getInput());
    if (!inputType) {
      failureReason = "broadcast input is not a ranked tensor";
      return failure();
    }
    llvm::SmallDenseSet<int64_t> dimensions =
        normalizeDimsToSet(broadcast.getDims(), inputType.getRank());
    plan.tileBroadcast =
        getTileBroadcastType(broadcast.getDims(), inputType.getRank());
    plan.iteration.inputMaps = {buildBroadcastInputMap(
        plan.source->getContext(), inputType.getRank(), dimensions)};
    plan.iteration.outputMap = AffineMap::getMultiDimIdentityMap(
        plan.resultType.getRank(), plan.source->getContext());
    plan.iteration.iteratorTypes.assign(plan.resultType.getRank(),
                                        utils::IteratorType::parallel);
    return success();
  }
  if (auto matmul = dyn_cast<MatmulOp>(plan.source)) {
    plan.recipe = ComputeFormationRecipe::Matmul;
    plan.transposeRhs = matmul.getTransposeRhs();
    MLIRContext *context = plan.source->getContext();
    AffineExpr dimensionM = getAffineDimExpr(0, context);
    AffineExpr dimensionN = getAffineDimExpr(1, context);
    AffineExpr dimensionK = getAffineDimExpr(2, context);
    plan.iteration.inputMaps = {
        AffineMap::get(3, 0, {dimensionM, dimensionK}, context),
        matmul.getTransposeRhs()
            ? AffineMap::get(3, 0, {dimensionN, dimensionK}, context)
            : AffineMap::get(3, 0, {dimensionK, dimensionN}, context)};
    plan.iteration.outputMap =
        AffineMap::get(3, 0, {dimensionM, dimensionN}, context);
    plan.iteration.iteratorTypes = {utils::IteratorType::parallel,
                                    utils::IteratorType::parallel,
                                    utils::IteratorType::reduction};
    return success();
  }
  if (auto reduce = dyn_cast<ReduceOp>(plan.source)) {
    plan.recipe = ComputeFormationRecipe::Reduce;
    RankedTensorType inputType = getTensorType(reduce.getInput());
    if (!inputType || inputType.getRank() != 2) {
      failureReason = "reduce requires a rank-2 input";
      return failure();
    }
    FailureOr<ttkernel::ReduceDim> reduceDimension =
        getReduceDimension(reduce.getDims(), inputType.getRank());
    if (failed(reduceDimension)) {
      failureReason = "unsupported reduction dimensions";
      return failure();
    }
    plan.reduceDimension = *reduceDimension;
    plan.reduceType = reduce.getReduceType();
    MLIRContext *context = plan.source->getContext();
    AffineExpr dimensionM = getAffineDimExpr(0, context);
    AffineExpr dimensionN = getAffineDimExpr(1, context);
    AffineExpr constantZero = getAffineConstantExpr(0, context);
    AffineMap inputMap = AffineMap::getMultiDimIdentityMap(2, context);
    switch (*reduceDimension) {
    case ttkernel::ReduceDim::Col:
      plan.iteration.outputMap =
          AffineMap::get(2, 0, {constantZero, dimensionN}, context);
      plan.iteration.iteratorTypes = {utils::IteratorType::reduction,
                                      utils::IteratorType::parallel};
      break;
    case ttkernel::ReduceDim::Row:
      plan.iteration.outputMap =
          AffineMap::get(2, 0, {dimensionM, constantZero}, context);
      plan.iteration.iteratorTypes = {utils::IteratorType::parallel,
                                      utils::IteratorType::reduction};
      break;
    case ttkernel::ReduceDim::Scalar:
      plan.iteration.outputMap =
          AffineMap::get(2, 0, {constantZero, constantZero}, context);
      plan.iteration.iteratorTypes = {utils::IteratorType::reduction,
                                      utils::IteratorType::reduction};
      break;
    }
    plan.iteration.inputMaps = {inputMap, plan.iteration.outputMap};
    return success();
  }
  if (isa<MulUnaryConstOp>(plan.source)) {
    plan.recipe = ComputeFormationRecipe::MulUnaryConst;
    plan.constantValue = cast<MulUnaryConstOp>(plan.source).getValueAttr();
    buildIdentityIterationPlan(plan);
    return success();
  }
  if (isa<FillOp>(plan.source)) {
    plan.recipe = ComputeFormationRecipe::Fill;
    plan.constantValue = cast<FillOp>(plan.source).getValueAttr();
    buildIdentityIterationPlan(plan);
    return success();
  }
  if (isa<TypecastOp>(plan.source)) {
    plan.recipe = ComputeFormationRecipe::Typecast;
    buildIdentityIterationPlan(plan);
    return success();
  }
  if (isa<TransposeOp>(plan.source)) {
    plan.recipe = ComputeFormationRecipe::Transpose;
    MLIRContext *context = plan.source->getContext();
    AffineExpr dimensionM = getAffineDimExpr(0, context);
    AffineExpr dimensionN = getAffineDimExpr(1, context);
    plan.iteration.inputMaps = {
        AffineMap::get(2, 0, {dimensionN, dimensionM}, context)};
    plan.iteration.outputMap = AffineMap::getMultiDimIdentityMap(2, context);
    plan.iteration.iteratorTypes.assign(2, utils::IteratorType::parallel);
    return success();
  }
  if (isElementwiseOp(plan.source)) {
    plan.recipe = ComputeFormationRecipe::Elementwise;
    buildIdentityIterationPlan(plan);
    return success();
  }

  failureReason = "operation has no ttl.compute construction recipe";
  return failure();
}

static std::optional<SmallVector<unsigned>>
getDirectInputOperandIndices(Operation *source) {
  if (isElementwiseOp(source)) {
    SmallVector<unsigned> indices;
    for (unsigned operandIndex = 0;
         operandIndex < getElementwiseOperands(source).size(); ++operandIndex) {
      indices.push_back(operandIndex);
    }
    return indices;
  }
  if (isa<FillOp>(source)) {
    return SmallVector<unsigned>{};
  }
  if (auto dfbInputOp = dyn_cast<DFBInputOpInterface>(source)) {
    return dfbInputOp.getDFBInputOperandIndices();
  }
  return std::nullopt;
}

static std::optional<SmallVector<Value>> collectDirectInputs(
    Operation *source,
    llvm::function_ref<bool(OpOperand &)> isMaterializationPlanned) {
  std::optional<SmallVector<unsigned>> inputIndices =
      getDirectInputOperandIndices(source);
  if (!inputIndices) {
    return std::nullopt;
  }

  SmallVector<Value> inputs;
  for (unsigned operandIndex : *inputIndices) {
    OpOperand &operand = source->getOpOperand(operandIndex);
    if (!isMaterializationPlanned(operand) && !getAttachedCB(operand.get())) {
      return std::nullopt;
    }
    inputs.push_back(operand.get());
  }
  return inputs;
}

static PlanningResult<OutputPublicationPlan>
resolveTransactionPushes(OutputPublicationPlan plan) {
  plan.pushes.clear();
  for (OutputDFBTransaction &transaction : plan.transactions) {
    transaction.push.reset();
    bool sawStoreWithoutPush = false;
    for (StoreOp store : transaction.stores) {
      std::optional<CBPushOp> push = findPushAfterStore(store, transaction.dfb);
      if (!push) {
        sawStoreWithoutPush = true;
        continue;
      }
      if (sawStoreWithoutPush ||
          (transaction.push && *transaction.push != *push)) {
        return PlanningResult<OutputPublicationPlan>::invalidIR(
            store, "stores from one reserve do not precede the same "
                   "ttl.cb_push operation");
      }
      transaction.push = *push;
    }
    if (sawStoreWithoutPush && transaction.push) {
      return PlanningResult<OutputPublicationPlan>::invalidIR(
          transaction.stores.back(),
          "stores from one reserve do not precede the same ttl.cb_push "
          "operation");
    }
    if (transaction.push) {
      plan.pushes.push_back(*transaction.push);
    }
  }

  llvm::sort(plan.pushes, [](CBPushOp lhs, CBPushOp rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  return PlanningResult<OutputPublicationPlan>::planned(std::move(plan));
}

} // namespace

PlanningResult<OutputPublicationPlan, OutputPublicationRejection>
buildOutputPublicationPlan(Operation *source) {
  PlanningResult<SmallVector<StoreOp>, OutputPublicationRejection> stores =
      collectStoreUsers(source);
  if (stores.isRejected()) {
    return PlanningResult<OutputPublicationPlan, OutputPublicationRejection>::
        rejected(stores.getRejection());
  }
  assert(stores.isPlanned() && "store collection cannot report malformed IR");

  OutputPublicationPlan plan;
  DenseMap<Operation *, unsigned> transactionByReserve;
  DenseMap<Value, Operation *> firstReserveByDFB;

  for (StoreOp store : stores.getPlan()) {
    CBReserveOp reserve = findCBReserveForView(store.getView());
    if (!reserve) {
      return PlanningResult<OutputPublicationPlan, OutputPublicationRejection>::
          invalidIR(store,
                    "output store view does not originate from ttl.cb_reserve");
    }

    Value dfb = reserve.getCb();
    addUniqueValue(plan.dfbs, dfb);
    plan.stores.push_back(store);

    auto [transactionIterator, inserted] = transactionByReserve.try_emplace(
        reserve.getOperation(), plan.transactions.size());
    if (inserted) {
      plan.transactions.push_back({dfb, reserve, {}, std::nullopt});
      auto [firstReserveIterator, insertedFirstReserve] =
          firstReserveByDFB.try_emplace(dfb, reserve.getOperation());
      if (!insertedFirstReserve &&
          firstReserveIterator->second != reserve.getOperation()) {
        addUniqueValue(plan.multiTransactionDFBs, dfb);
      }
    }

    OutputDFBTransaction &transaction =
        plan.transactions[transactionIterator->second];
    transaction.stores.push_back(store);
  }
  plan.insertionAnchor = plan.stores.back();

  PlanningResult<OutputPublicationPlan> resolved =
      resolveTransactionPushes(std::move(plan));
  if (resolved.isInvalidIR()) {
    const PlanningDiagnostic &diagnostic = resolved.getInvalidIR();
    return PlanningResult<OutputPublicationPlan, OutputPublicationRejection>::
        invalidIR(diagnostic.operation, std::move(diagnostic.message));
  }
  assert(resolved.isPlanned() &&
         "publication resolution has no recoverable rejection");
  return PlanningResult<OutputPublicationPlan,
                        OutputPublicationRejection>::planned(std::move(resolved)
                                                                 .takePlan());
}

bool isComputeFormationUsePreserved(const OutputPublicationPlan &outputs,
                                    OpOperand &use,
                                    const DominanceInfo &dominanceInfo) {
  if (auto store = dyn_cast<StoreOp>(use.getOwner());
      store && &store.getTensorMutable() == &use &&
      llvm::is_contained(outputs.stores, store)) {
    return true;
  }
  return dominanceInfo.properlyDominates(outputs.insertionAnchor,
                                         use.getOwner());
}

bool isComputeFormationElision(Operation *source) {
  auto typecast = dyn_cast<TypecastOp>(source);
  return typecast &&
         typecast.getInput().getType() == typecast.getResult().getType();
}

bool hasStandaloneComputeFormationRecipe(Operation *source) {
  if (source->getNumResults() != 1) {
    return false;
  }
  RankedTensorType resultType = getTensorType(source->getResult(0));
  std::optional<SmallVector<unsigned>> inputIndices =
      getDirectInputOperandIndices(source);
  if (!resultType || !inputIndices) {
    return false;
  }

  ComputeFormationPlan plan;
  plan.source = source;
  plan.resultType = resultType;
  plan.kind = isComputeFormationElision(source) ? ComputeFormationKind::Elide
                                                : ComputeFormationKind::Direct;
  for (unsigned operandIndex : *inputIndices) {
    plan.inputs.push_back(source->getOperand(operandIndex));
  }
  std::string failureReason;
  return succeeded(buildOperationSpecificPlan(plan, failureReason));
}

PlanningResult<OutputPublicationPlan>
resolveOutputPublicationOperations(const OutputPublicationPlan &analyzed) {
  return resolveTransactionPushes(analyzed);
}

FailureOr<SmallVector<Value>> collectComputeFormationInputs(
    Operation *source,
    llvm::function_ref<bool(OpOperand &)> isMaterializationPlanned) {
  if (source->getNumResults() != 1) {
    return failure();
  }

  if (std::optional<SmallVector<Value>> directInputs =
          collectDirectInputs(source, isMaterializationPlanned)) {
    return std::move(*directInputs);
  }

  FusionTraceResult trace =
      traceFusionToRoots(source->getResult(0), isMaterializationPlanned);
  if (trace.failureReason == TraceFailureReason::Success &&
      !trace.opsInOrder.empty()) {
    return SmallVector<Value>(trace.rootInputs.begin(), trace.rootInputs.end());
  }
  return failure();
}

static PlanningResult<ComputeFormationPlan, ComputeFormationRejection>
rejectComputeFormation(
    Operation *source, ComputeFormationRejectionKind kind, std::string message,
    std::optional<ComputeFormationPlan> candidate = std::nullopt) {
  return PlanningResult<ComputeFormationPlan, ComputeFormationRejection>::
      rejected({source, kind, std::move(message), std::move(candidate)});
}

static PlanningResult<ComputeFormationPlan, ComputeFormationRejection>
buildComputeFormationPlan(Operation *source,
                          const DFBValueLifetimeAnalysis &lifetimes,
                          const DominanceInfo &dominanceInfo,
                          std::string &failureReason) {
  if (source->getNumResults() != 1) {
    return rejectComputeFormation(
        source, ComputeFormationRejectionKind::UnsupportedCandidate,
        "operation must have exactly one result to form compute");
  }

  ComputeFormationPlan plan;
  plan.source = source;
  plan.resultType = getTensorType(source->getResult(0));
  if (!plan.resultType) {
    return rejectComputeFormation(
        source, ComputeFormationRejectionKind::UnsupportedCandidate,
        "operation result is not a ranked tensor");
  }
  llvm::append_range(plan.applicationOperands, source->getOperands());
  for (OpOperand &use : source->getResult(0).getUses()) {
    plan.resultUses.push_back({use.getOwner(), use.getOperandNumber()});
  }

  if (isComputeFormationElision(source)) {
    plan.kind = ComputeFormationKind::Elide;
    plan.inputs = {cast<TypecastOp>(source).getInput()};
  } else if (std::optional<SmallVector<Value>> directInputs =
                 collectDirectInputs(source,
                                     [](OpOperand &) { return false; })) {
    plan.kind = ComputeFormationKind::Direct;
    plan.inputs = std::move(*directInputs);
  } else if (auto reduce = dyn_cast<ReduceOp>(source)) {
    if (!getAttachedCB(reduce.getInput())) {
      Operation *inputDefinition = reduce.getInput().getDefiningOp();
      bool isUnstoredComputeResult =
          inputDefinition &&
          (isElementwiseOp(inputDefinition) ||
           isa<MatmulOp, BlockBroadcastOp, FillOp>(inputDefinition));
      return rejectComputeFormation(
          source, ComputeFormationRejectionKind::UnmaterializedInput,
          isUnstoredComputeResult
              ? "reduce input is an unstored compute result; store the "
                "intermediate result to a dataflow buffer before passing it "
                "to reduce (see issue #474)"
              : "reduce input must be dataflow-buffer-backed");
    }
    return rejectComputeFormation(
        source, ComputeFormationRejectionKind::UnmaterializedInput,
        "reduce scaler must be dataflow-buffer-backed");
  } else {
    plan.trace = traceFusionToRoots(source->getResult(0));
    if (plan.trace.failureReason != TraceFailureReason::Success ||
        plan.trace.opsInOrder.empty()) {
      return rejectComputeFormation(
          source, ComputeFormationRejectionKind::UnsupportedCandidate,
          "operation has no defined ttl.compute input semantics");
    }
    plan.kind = ComputeFormationKind::Fused;
    llvm::append_range(plan.inputs, plan.trace.rootInputs);
  }

  if (failed(buildOperationSpecificPlan(plan, failureReason))) {
    return rejectComputeFormation(
        source, ComputeFormationRejectionKind::UnsupportedCandidate,
        std::move(failureReason));
  }

  // Identity typecasts preserve position and storage. They neither form a
  // compute nor alter output publication, so requiring a store would make
  // their legality depend on a later mutation. Record the final input expected
  // after prerequisite identity elisions and apply them independently.
  if (plan.kind == ComputeFormationKind::Elide) {
    Value input = plan.inputs.front();
    while (Operation *definition = input.getDefiningOp()) {
      if (!isComputeFormationElision(definition)) {
        break;
      }
      input = cast<TypecastOp>(definition).getInput();
    }
    plan.inputs = {input};
    plan.applicationOperands = {input};
    return PlanningResult<ComputeFormationPlan,
                          ComputeFormationRejection>::planned(std::move(plan));
  }

  PlanningResult<OutputPublicationPlan, OutputPublicationRejection> outputs =
      buildOutputPublicationPlan(source);
  if (outputs.isInvalidIR()) {
    const PlanningDiagnostic &diagnostic = outputs.getInvalidIR();
    return PlanningResult<ComputeFormationPlan, ComputeFormationRejection>::
        invalidIR(diagnostic.operation, diagnostic.message);
  }
  if (outputs.isRejected()) {
    return rejectComputeFormation(
        source, ComputeFormationRejectionKind::UnsupportedOutputPublication,
        outputs.getRejection().message);
  }
  plan.outputs = std::move(outputs).takePlan();

  if (plan.kind == ComputeFormationKind::Fused &&
      failed(collectFusedInstrumentation(plan.trace, plan.outputs, plan.source,
                                         plan.instrumentation,
                                         failureReason))) {
    plan.rejectionKind = ComputeFormationRejectionKind::UnsupportedCandidate;
    plan.rejectionReason = failureReason;
    return rejectComputeFormation(
        source, plan.rejectionKind, plan.rejectionReason,
        std::optional<ComputeFormationPlan>(std::move(plan)));
  }

  if (plan.outputs.hasMultipleTransactionsForOneDFB()) {
    plan.rejectionKind =
        ComputeFormationRejectionKind::MultipleOutputTransactions;
    plan.rejectionReason =
        "one compute cannot publish multiple reserve transactions of the "
        "same dataflow buffer";
  } else if (lifetimes.anyValueMayBeReleased(plan.inputs,
                                             plan.outputs.insertionAnchor)) {
    plan.rejectionKind = ComputeFormationRejectionKind::InputMayBeReleased;
    plan.rejectionReason =
        "moving tensor evaluation to the final output store would read a "
        "dataflow buffer value after its pop";
  } else {
    for (OpOperand &use : source->getResult(0).getUses()) {
      if (!isComputeFormationUsePreserved(plan.outputs, use, dominanceInfo)) {
        plan.preFormationRemovedUses.push_back(
            {use.getOwner(), use.getOperandNumber()});
      }
    }
    if (!plan.preFormationRemovedUses.empty()) {
      plan.rejectionKind = ComputeFormationRejectionKind::ResultUseNotDominated;
      plan.rejectionReason =
          "ttl.compute inserted at the final output store would not dominate "
          "every surviving result use";
    }
  }

  if (!plan.isLegal()) {
    return rejectComputeFormation(
        source, plan.rejectionKind, plan.rejectionReason,
        std::optional<ComputeFormationPlan>(std::move(plan)));
  }
  return PlanningResult<ComputeFormationPlan,
                        ComputeFormationRejection>::planned(std::move(plan));
}

static FailureOr<PassthroughStorePlan>
buildPassthroughStorePlan(StoreOp store,
                          const DFBValueLifetimeAnalysis &lifetimes,
                          std::string &failureReason) {
  SmallVector<Value> inputChain = {store.getTensor()};
  while (Operation *definition = inputChain.back().getDefiningOp()) {
    if (!isComputeFormationElision(definition)) {
      break;
    }
    inputChain.push_back(cast<TypecastOp>(definition).getInput());
  }
  Value input = inputChain.back();
  if (!getAttachedCB(input)) {
    failureReason = "store input is not dataflow-buffer-backed";
    return failure();
  }
  CBReserveOp reserve = findCBReserveForView(store.getView());
  if (!reserve) {
    failureReason = "store view does not originate from ttl.cb_reserve";
    return failure();
  }
  RankedTensorType tensorType = getTensorType(input);
  if (!tensorType) {
    failureReason = "store input is not a ranked tensor";
    return failure();
  }
  if (lifetimes.getAvailability(input, store) ==
      DFBValueAvailability::MayBeReleased) {
    failureReason =
        "passthrough store would read a dataflow buffer value after release";
    return failure();
  }

  PassthroughStorePlan plan;
  plan.store = store;
  plan.originalInput = store.getTensor();
  plan.input = input;
  plan.reserve = reserve;
  plan.outputView = store.getView();
  plan.outputDFB = reserve.getCb();
  plan.tensorType = tensorType;
  AffineMap identity = AffineMap::getMultiDimIdentityMap(tensorType.getRank(),
                                                         store->getContext());
  plan.iteration.inputMaps = {identity};
  plan.iteration.outputMap = identity;
  plan.iteration.iteratorTypes.assign(tensorType.getRank(),
                                      utils::IteratorType::parallel);
  // These associations represent the passthrough result before a result SSA
  // value exists. Recording them avoids searching a mutated use-list later.
  for (Value chainValue : inputChain) {
    for (OpOperand &use : chainValue.getUses()) {
      auto association = dyn_cast<AttachCBOp>(use.getOwner());
      if (association && association.getCb() == plan.outputDFB &&
          !llvm::is_contained(plan.outputAssociations, association)) {
        plan.outputAssociations.push_back(association);
      }
    }
  }
  return plan;
}

/// Returns the source operations erased when one fused plan is applied.
///
/// The sink is replaced unconditionally. An earlier operation is erased only
/// when every result user has already been erased, which is the same reverse
/// use-empty rule used by `buildFusedCompute`. An operation with an independent
/// store or other external use therefore remains available to a later plan.
static DenseSet<Operation *>
collectErasedFusedOperations(const ComputeFormationPlan &formation) {
  DenseSet<Operation *> erased;
  if (formation.kind != ComputeFormationKind::Fused) {
    return erased;
  }

  erased.insert(formation.source);
  for (Operation *operation : llvm::reverse(formation.trace.opsInOrder)) {
    if (operation == formation.source) {
      continue;
    }
    bool allUsersErased =
        llvm::all_of(operation->getUsers(),
                     [&](Operation *user) { return erased.contains(user); });
    if (allUsersErased) {
      erased.insert(operation);
    }
  }
  return erased;
}

static bool
requiresIntermediateDFBResolution(ComputeFormationRejectionKind rejectionKind) {
  switch (rejectionKind) {
  case ComputeFormationRejectionKind::UnmaterializedInput:
  case ComputeFormationRejectionKind::MultipleOutputTransactions:
  case ComputeFormationRejectionKind::InputMayBeReleased:
  case ComputeFormationRejectionKind::ResultUseNotDominated:
    return true;
  case ComputeFormationRejectionKind::None:
  case ComputeFormationRejectionKind::UnsupportedCandidate:
  case ComputeFormationRejectionKind::UnsupportedOutputPublication:
  case ComputeFormationRejectionKind::DeferredDependency:
    return false;
  }
  llvm_unreachable("unknown compute formation rejection kind");
}

PlanningResult<KernelComputeFormationPlan>
ComputeFormationPlanner::build() const {
  KernelComputeFormationPlan kernelPlan;
  DominanceInfo dominanceInfo(kernel);
  SmallVector<Operation *> candidates;
  std::optional<PlanningDiagnostic> invalidIR;
  kernel->walk([&](Operation *source) {
    if (invalidIR) {
      return;
    }
    if (source->getNumResults() != 1) {
      return;
    }

    std::string failureReason;
    PlanningResult<ComputeFormationPlan, ComputeFormationRejection> formation =
        buildComputeFormationPlan(source, lifetimes, dominanceInfo,
                                  failureReason);
    if (formation.isInvalidIR()) {
      invalidIR = formation.getInvalidIR();
      return;
    }
    if (formation.isRejected()) {
      ComputeFormationRejection rejection =
          std::move(formation).takeRejection();
      if (!rejection.candidate) {
        kernelPlan.rejectionKinds.try_emplace(source, rejection.kind);
        kernelPlan.rejectionReasons.try_emplace(source,
                                                std::move(rejection.message));
        return;
      }
      kernelPlan.formations.try_emplace(source,
                                        std::move(*rejection.candidate));
      candidates.push_back(source);
      return;
    }
    kernelPlan.formations.try_emplace(source, std::move(formation).takePlan());
    candidates.push_back(source);
  });
  if (invalidIR) {
    return PlanningResult<KernelComputeFormationPlan>::invalidIR(
        invalidIR->operation, std::move(invalidIR->message));
  }
  kernelPlan.analyzedSources = candidates;

  // A fused outer formation may erase every use that precedes an inner
  // candidate's insertion anchor. Accept the inner plan only when each such
  // consumer has its own already accepted fused plan containing the inner
  // source. The outer source's independent store keeps it present until that
  // plan runs, and the inner store keeps the inner source present afterward.
  bool acceptedDependentFormation;
  do {
    acceptedDependentFormation = false;
    for (Operation *source : candidates) {
      ComputeFormationPlan &formation = kernelPlan.formations.at(source);
      if (formation.isLegal() ||
          formation.rejectionKind !=
              ComputeFormationRejectionKind::ResultUseNotDominated) {
        continue;
      }
      bool allUsesRemoved = llvm::all_of(
          formation.preFormationRemovedUses,
          [&](const ComputeFormationUse &use) {
            auto remover = kernelPlan.formations.find(use.owner);
            return remover != kernelPlan.formations.end() &&
                   remover->second.isLegal() &&
                   remover->second.kind == ComputeFormationKind::Fused &&
                   remover->second.trace.opsInOrder.contains(source);
          });
      if (!allUsesRemoved) {
        continue;
      }
      formation.rejectionKind = ComputeFormationRejectionKind::None;
      formation.rejectionReason.clear();
      acceptedDependentFormation = true;
    }
  } while (acceptedDependentFormation);

  DenseSet<Operation *> deferredDependencies;
  for (Operation *source : candidates) {
    const ComputeFormationPlan &formation = kernelPlan.formations.at(source);
    if (formation.isLegal() ||
        !requiresIntermediateDFBResolution(formation.rejectionKind)) {
      continue;
    }
    for (Operation *absorbed : formation.trace.opsInOrder) {
      if (absorbed != source && kernelPlan.formations.contains(absorbed)) {
        deferredDependencies.insert(absorbed);
      }
    }
  }
  // A formation rejected for an unresolved DFB prerequisite still owns its
  // expression until intermediate DFB insertion rewrites the required
  // operands. Forming an absorbed producer first would erase that expression.
  // Rejections without an intermediate DFB requirement create no dependency.
  for (Operation *source : deferredDependencies) {
    ComputeFormationPlan &formation = kernelPlan.formations.at(source);
    formation.rejectionKind = ComputeFormationRejectionKind::DeferredDependency;
    formation.rejectionReason =
        "a dependent expression requires dataflow buffer materialization "
        "before this operation can be lowered";
  }

  DenseSet<Operation *> erasedCandidates;
  DenseMap<Operation *, SmallVector<Operation *>> precedingFormations;
  for (Operation *source : candidates) {
    const ComputeFormationPlan &formation = kernelPlan.formations.at(source);
    if (!formation.isLegal()) {
      continue;
    }
    if (formation.kind == ComputeFormationKind::Elide) {
      Operation *inputDefinition = source->getOperand(0).getDefiningOp();
      if (inputDefinition && kernelPlan.formations.contains(inputDefinition) &&
          kernelPlan.formations.at(inputDefinition).isLegal() &&
          kernelPlan.formations.at(inputDefinition).kind ==
              ComputeFormationKind::Elide) {
        precedingFormations[source].push_back(inputDefinition);
      }
    }
    DenseSet<Operation *> erasedOperations =
        collectErasedFusedOperations(formation);
    for (Operation *absorbed : formation.trace.opsInOrder) {
      if (absorbed != source && kernelPlan.formations.contains(absorbed) &&
          kernelPlan.formations.at(absorbed).isLegal()) {
        if (erasedOperations.contains(absorbed)) {
          erasedCandidates.insert(absorbed);
          continue;
        }
        SmallVector<Operation *> &preceding = precedingFormations[absorbed];
        if (!llvm::is_contained(preceding, source)) {
          preceding.push_back(source);
        }
      }
    }
  }

  // Apply every absorbing consumer before forming its source separately; each
  // independent use keeps the source available until all absorbers run.
  // Identity elisions instead follow SSA definition order because each outer
  // plan records the result that its inner identity operation replaces.
  DenseSet<Operation *> scheduled;
  DenseSet<Operation *> visiting;
  std::function<void(Operation *)> schedule = [&](Operation *source) {
    if (erasedCandidates.contains(source) || scheduled.contains(source)) {
      return;
    }
    bool inserted = visiting.insert(source).second;
    assert(inserted && "compute formation dependencies must be acyclic");
    for (Operation *preceding : precedingFormations.lookup(source)) {
      schedule(preceding);
    }
    visiting.erase(source);
    scheduled.insert(source);
    kernelPlan.formationOrder.push_back(source);
  };

  for (Operation *candidate : llvm::reverse(candidates)) {
    if (kernelPlan.formations.at(candidate).isLegal()) {
      schedule(candidate);
    }
  }

  for (auto [laterIndex, laterSource] :
       llvm::enumerate(kernelPlan.formationOrder)) {
    DenseSet<Operation *> erasedBeforeLater;
    bool absorbedByEarlierFormation = false;
    for (Operation *earlierSource :
         ArrayRef<Operation *>(kernelPlan.formationOrder)
             .take_front(laterIndex)) {
      const ComputeFormationPlan &earlier =
          kernelPlan.formations.at(earlierSource);
      if (!earlier.trace.opsInOrder.contains(laterSource)) {
        continue;
      }
      absorbedByEarlierFormation = true;
      DenseSet<Operation *> erasedOperations =
          collectErasedFusedOperations(earlier);
      assert(!erasedOperations.contains(laterSource) &&
             "an operation erased by an earlier formation must not be "
             "scheduled separately");
      erasedBeforeLater.insert(erasedOperations.begin(),
                               erasedOperations.end());
    }
    if (!absorbedByEarlierFormation) {
      continue;
    }

    // One use outside every preceding absorbed subgraph proves that the
    // source remains present until its own formation is applied.
    ComputeFormationPlan &later = kernelPlan.formations.at(laterSource);
    auto preservingUse =
        llvm::find_if(later.resultUses, [&](const ComputeFormationUse &use) {
          return !erasedBeforeLater.contains(use.owner);
        });
    assert(preservingUse != later.resultUses.end() &&
           "a surviving formation candidate must retain a result use");
    later.preservingUses.push_back(*preservingUse);
  }

  kernel->walk([&](StoreOp store) {
    std::string failureReason;
    FailureOr<PassthroughStorePlan> plan =
        buildPassthroughStorePlan(store, lifetimes, failureReason);
    if (succeeded(plan)) {
      kernelPlan.passthroughStores.try_emplace(store, std::move(*plan));
    } else {
      kernelPlan.rejectionReasons.try_emplace(store, std::move(failureReason));
    }
  });

  DenseSet<Operation *> plannedStores;
  for (Operation *source : kernelPlan.formationOrder) {
    const ComputeFormationPlan &formation = kernelPlan.formations.at(source);
    for (StoreOp store : formation.outputs.stores) {
      plannedStores.insert(store.getOperation());
    }
  }
  for (const auto &entry : kernelPlan.passthroughStores) {
    plannedStores.insert(entry.first);
  }
  kernel->walk([&](StoreOp store) {
    if (!plannedStores.contains(store.getOperation())) {
      kernelPlan.unassignedStores.push_back(store);
    }
  });
  return PlanningResult<KernelComputeFormationPlan>::planned(
      std::move(kernelPlan));
}

FailureOr<const ComputeFormationPlan *>
KernelComputeFormationPlan::get(Operation *source) const {
  auto formation = formations.find(source);
  if (formation == formations.end() || !formation->second.isLegal()) {
    return failure();
  }
  return &formation->second;
}

const ComputeFormationPlan &
KernelComputeFormationPlan::getAnalyzedFormation(Operation *source) const {
  auto formation = formations.find(source);
  assert(formation != formations.end() &&
         "source must be returned by getAnalyzedSources");
  return formation->second;
}

StringRef
KernelComputeFormationPlan::getRejectionReason(Operation *source) const {
  auto formation = formations.find(source);
  if (formation != formations.end() && !formation->second.isLegal()) {
    return formation->second.rejectionReason;
  }
  auto reason = rejectionReasons.find(source);
  if (reason == rejectionReasons.end()) {
    return "operation is not a supported ttl.compute source";
  }
  return reason->second;
}

std::optional<ComputeFormationRejectionKind>
KernelComputeFormationPlan::getRejectionKind(Operation *source) const {
  auto formation = formations.find(source);
  if (formation != formations.end() && !formation->second.isLegal()) {
    return formation->second.rejectionKind;
  }
  auto rejection = rejectionKinds.find(source);
  if (rejection == rejectionKinds.end()) {
    return std::nullopt;
  }
  return rejection->second;
}

PlanningDiagnostic
KernelComputeFormationPlan::getUnassignedStoreDiagnostic(StoreOp store) const {
  PlanningDiagnostic diagnostic{store, getRejectionReason(store).str()};
  std::optional<Operation *> formationSource =
      getUnassignedStoreFormationSource(store);
  if (!formationSource) {
    return diagnostic;
  }

  Operation *source = *formationSource;
  std::optional<ComputeFormationRejectionKind> sourceRejection =
      getRejectionKind(source);
  assert(sourceRejection && "reported formation source requires a rejection");

  diagnostic.message = getRejectionReason(source);
  if (*sourceRejection == ComputeFormationRejectionKind::UnmaterializedInput) {
    diagnostic.operation = source;
  }
  return diagnostic;
}

std::optional<Operation *>
KernelComputeFormationPlan::getUnassignedStoreFormationSource(
    StoreOp store) const {
  Operation *source = store.getTensor().getDefiningOp();
  if (!source) {
    return std::nullopt;
  }

  std::optional<ComputeFormationRejectionKind> sourceRejection =
      getRejectionKind(source);
  if (!sourceRejection ||
      *sourceRejection == ComputeFormationRejectionKind::UnsupportedCandidate) {
    return std::nullopt;
  }
  return source;
}

FailureOr<const PassthroughStorePlan *>
KernelComputeFormationPlan::get(StoreOp store) const {
  auto plan = passthroughStores.find(store);
  if (plan == passthroughStores.end()) {
    return failure();
  }
  return &plan->second;
}

} // namespace mlir::tt::ttl
