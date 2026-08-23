// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ComputeOpCreationPlanning.h"
#include "DFBValueLifetimeAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/ComputeTarget.h"

#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsEnums.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "ttl-convert-ttl-to-compute"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLPRODUCERCOMPUTECREATION
#define GEN_PASS_DEF_TTLCONVERTTTLTOCOMPUTE
#include "ttlang/Dialect/TTL/Passes.h.inc"

/// Controls whether unresolved DFB-only consumers are deferred or diagnosed.
enum class TTLToComputeMode {
  ProducerCreation,
  FinalConversion,
};

static bool containsOperation(func::FuncOp kernel, Operation *candidate) {
  bool found = false;
  kernel.walk([&](Operation *operation) { found |= operation == candidate; });
  return found;
}

static LogicalResult
getCreationPlan(Operation *source, PatternRewriter &rewriter,
                const KernelComputeOpCreationPlan &kernelPlan,
                const ComputeOpCreationPlan *&creation) {
  FailureOr<const ComputeOpCreationPlan *> planned = kernelPlan.get(source);
  if (failed(planned)) {
    return rewriter.notifyMatchFailure(source,
                                       kernelPlan.getRejectionReason(source));
  }
  creation = *planned;
  assert(creation->source == source &&
         "kernel plan must be indexed by its source operation");
  if (!llvm::equal(source->getOperands(), creation->applicationOperands)) {
    return rewriter.notifyMatchFailure(
        source, "candidate operands changed after ComputeOp creation analysis");
  }
  for (OpOperand &use : source->getResult(0).getUses()) {
    if (llvm::none_of(creation->resultUses,
                      [&](const ComputeOpCreationUse &recordedUse) {
                        return recordedUse.matches(use);
                      })) {
      return rewriter.notifyMatchFailure(
          source,
          "candidate acquired a new use after ComputeOp creation analysis");
    }
  }
  if (!creation->preservingUses.empty() &&
      llvm::none_of(source->getResult(0).getUses(), [&](OpOperand &use) {
        return llvm::any_of(creation->preservingUses,
                            [&](const ComputeOpCreationUse &recordedUse) {
                              return recordedUse.matches(use);
                            });
      })) {
    return rewriter.notifyMatchFailure(
        source, "use preserving an overlapping creation was removed");
  }
  for (const ComputeOpCreationUse &removedUse :
       creation->preCreationRemovedUses) {
    if (llvm::any_of(source->getResult(0).getUses(),
                     [&](OpOperand &use) { return removedUse.matches(use); })) {
      return rewriter.notifyMatchFailure(
          source, "prerequisite creation did not remove a result use");
    }
  }
  auto kernel = source->getParentOfType<func::FuncOp>();
  for (const ComputeInstrumentationPlacement &placement :
       creation->instrumentation) {
    if (!containsOperation(kernel, placement.operation) ||
        (placement.after && !containsOperation(kernel, placement.after))) {
      return rewriter.notifyMatchFailure(
          source, "instrumentation changed after ComputeOp creation analysis");
    }
  }
  return success();
}

static LogicalResult
resolveCurrentOutputs(Operation *source, PatternRewriter &rewriter,
                      const ComputeOpCreationPlan &creation,
                      OutputPublicationPlan &outputs) {
  PlanningResult<OutputPublicationPlan> resolved =
      resolveOutputPublicationOperations(creation.outputs);
  if (resolved.isInvalidIR()) {
    return rewriter.notifyMatchFailure(source, resolved.getInvalidIR().message);
  }
  assert(resolved.isPlanned() &&
         "output resolution has no recoverable rejection");
  outputs = std::move(resolved).takePlan();
  return success();
}

static LogicalResult
getCreationPlan(Operation *source, ComputeOpCreationKind expectedKind,
                ComputeOpCreationRecipe expectedRecipe, ValueRange inputs,
                PatternRewriter &rewriter,
                const KernelComputeOpCreationPlan &kernelPlan,
                const ComputeOpCreationPlan *&creation) {
  if (failed(getCreationPlan(source, rewriter, kernelPlan, creation))) {
    return failure();
  }
  if (creation->kind != expectedKind) {
    return rewriter.notifyMatchFailure(
        source, "lowering strategy differs from the analyzed creation");
  }
  if (creation->recipe != expectedRecipe) {
    return rewriter.notifyMatchFailure(
        source, "tile recipe differs from the analyzed creation plan");
  }
  if (!llvm::equal(inputs, creation->inputs)) {
    return rewriter.notifyMatchFailure(
        source, "lowering inputs differ from the analyzed creation inputs");
  }
  return success();
}

static RankedTensorType getTensorType(Value value) {
  return dyn_cast<RankedTensorType>(value.getType());
}

/// Converts the typed planning representation to `ttl.compute`'s current
/// string-based iterator syntax. Keeping strings out of the plan prevents
/// misspelled values and makes iterator semantics explicit to C++ callers.
static SmallVector<Attribute>
buildIteratorTypeAttributes(OpBuilder &builder,
                            ArrayRef<utils::IteratorType> iteratorTypes) {
  SmallVector<Attribute> attributes;
  llvm::transform(iteratorTypes, std::back_inserter(attributes),
                  [&](utils::IteratorType type) -> Attribute {
                    return builder.getStringAttr(
                        utils::stringifyIteratorType(type));
                  });
  return attributes;
}

static Value buildInitTensor(OpBuilder &b, Location loc, RankedTensorType type,
                             Value exemplar) {
  SmallVector<Value> dynDims;
  for (auto dim : llvm::enumerate(type.getShape())) {
    if (dim.value() == ShapedType::kDynamic) {
      dynDims.push_back(tensor::DimOp::create(b, loc, exemplar, dim.index()));
    }
  }
  return tensor::EmptyOp::create(b, loc, type.getShape(), type.getElementType(),
                                 dynDims);
}

/// Selects the insertion position proven by output-publication planning.
static void insertAtCreationAnchor(PatternRewriter &rewriter,
                                   const OutputPublicationPlan &outputs) {
  rewriter.setInsertionPoint(outputs.insertionAnchor);
}

/// Creates one tile store using the output transaction selected by planning.
static void emitTileStore(PatternRewriter &rewriter, Location loc,
                          Value tileResult, ComputeOp computeOp, StoreOp store,
                          const ComputeOpCreationPlan &creation) {
  SmallVector<Value> iterIndices = getOrCreateIterIndices(rewriter, computeOp);
  auto indexingMaps = computeOp.getIndexingMapsArray();
  size_t numInputs = computeOp.getNumInputs();

  FailureOr<unsigned> outputIndex =
      computeOp.getOutputIndexForView(store.getView());
  assert(succeeded(outputIndex) &&
         "planned store must map to one formal compute output");
  AffineMap outputMap = indexingMaps[numInputs + *outputIndex];
  SmallVector<Value> indices =
      applyIndexingMap(rewriter, loc, outputMap, iterIndices);

  TileStoreOp tileStore = createTileOpWithPlaceholderDstIndex<TileStoreOp>(
      rewriter, loc, tileResult, store.getView(), indices);
  const WaitedDFBMutationPlan *waitedMutation = nullptr;
  for (const WaitedDFBMutationPlan &mutation : creation.waitedMutations) {
    if (mutation.store != store) {
      continue;
    }
    assert(!waitedMutation && "one store cannot replace two waited DFBs");
    waitedMutation = &mutation;
  }
  bool isWaitedMutation = waitedMutation != nullptr;
  assert(isWaitedMutation == isa<CBWaitOp>(findCBAcquireOp(store.getView())) &&
         "wait-backed tile store must consume a proved mutation plan");
  if (isWaitedMutation) {
    CBWaitOp waitedAcquire = waitedMutation->wait;
    CBPopOp waitedRelease = waitedMutation->release;
    assert(waitedAcquire == findCBAcquireOp(store.getView()) &&
           waitedAcquire.getCb() == waitedMutation->dfb &&
           waitedRelease.getCb() == waitedMutation->dfb &&
           waitedMutation->transactionTiles ==
               getDFBLifecycleTileCount(waitedAcquire) &&
           waitedMutation->transactionTiles ==
               getDFBLifecycleTileCount(waitedRelease) &&
           waitedMutation->capacityTiles ==
               cast<CircularBufferType>(waitedMutation->dfb.getType())
                   .getTotalElements() &&
           "waited DFB mutation changed after planning");
    tileStore.setStoreKind(DFBTileStoreKind::ConsumerReplacement);
  }
}

static void
replaceOutputPushesBeforeCompute(PatternRewriter &rewriter, ComputeOp computeOp,
                                 const OutputPublicationPlan &outputs,
                                 SmallVectorImpl<CBPushOp> &replacedPushes) {
  OpBuilder::InsertionGuard guard(rewriter);
  Operation *insertAfter = computeOp;
  for (CBPushOp push : outputs.pushes) {
    assert(push->getBlock() == computeOp->getBlock() &&
           "pushes absorbed into a compute must be siblings of that compute");
    // Publications already after the new compute preserve their ordering.
    if (!push->isBeforeInBlock(computeOp)) {
      continue;
    }
    rewriter.setInsertionPointAfter(insertAfter);
    auto replacement = cast<CBPushOp>(rewriter.clone(*push));
    insertAfter = replacement;
    replacedPushes.push_back(push);
  }
}

static void eraseAbsorbedOutputOps(PatternRewriter &rewriter,
                                   const OutputPublicationPlan &outputs,
                                   ComputeOp computeOp,
                                   ArrayRef<CBPushOp> replacedPushes) {
  for (StoreOp store : outputs.stores) {
    assert(store->getBlock() == computeOp->getBlock() &&
           "stores absorbed into a compute must be siblings of that compute");
    rewriter.eraseOp(store);
  }
  for (CBPushOp push : replacedPushes) {
    assert(push->getBlock() == computeOp->getBlock() &&
           "pushes absorbed into a compute must be siblings of that compute");
    rewriter.eraseOp(push);
  }
}

//===----------------------------------------------------------------------===//
// Tile op emission for fusion
//===----------------------------------------------------------------------===//

static Value emitExpTileOp(OpBuilder &builder, Location loc, Type tileType,
                           Value input, const ExpFlagsPlan &flags) {
  FloatAttr scale = flags.scale;
  if (scale && !flags.approx.getValue()) {
    // The current accurate BF16 LLK path passes this runtime value to
    // immediate-only SFPMULI. Keep accurate semantics by materializing the
    // scale as a supported tile multiply before invoking unscaled exp.
    input = createTileOpWithPlaceholderDstIndex<TileMulUnaryConstOp>(
        builder, loc, tileType, input, scale);
    scale = nullptr;
  }

  Value dstIndex = createPlaceholderDstIndex(builder, loc);
  auto tileOp =
      ExpTileOp::create(builder, loc, tileType, input, dstIndex, flags.approx,
                        scale, flags.inputClamping, flags.iterations);
  addPlaceholderDstIndexAttr(tileOp.getOperation());
  return tileOp.getResult();
}

/// Creates the generic tile operation selected by a verified fused plan.
static Value emitTileOpFor(OpBuilder &builder, Location loc,
                           const FusedOperationPlan &operationPlan,
                           ValueRange tileOperands) {
  Operation *sourceOp = operationPlan.source;
  Type tileType = operationPlan.resultTileType;

#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  if (isa<TTL_OP##Op>(sourceOp))                                               \
    return createTileOpWithPlaceholderDstIndex<TILE_OP>(                       \
        builder, loc, tileType, tileOperands[0]);
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)             \
  if (isa<TTL_OP##Op>(sourceOp))                                               \
    return createTileOpWithPlaceholderDstIndex<TILE_OP>(                       \
        builder, loc, tileType, tileOperands[0], tileOperands[1]);
#define TTL_BINARY_TILE_OP_MINMAX(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)      \
  TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // ExpOp is excluded from TTLElementwiseOps.def because it carries hardware
  // flags that must be forwarded to the tile operation.
  if (isa<ExpOp>(sourceOp)) {
    assert(tileOperands.size() == 1 && "exp fusion requires one tile operand");
    assert(operationPlan.expFlags &&
           "exp recipe must record its hardware flags");
    return emitExpTileOp(builder, loc, tileType, tileOperands[0],
                         *operationPlan.expFlags);
  }

  if (isa<MulUnaryConstOp>(sourceOp)) {
    assert(tileOperands.size() == 1 &&
           "mul_unary_const fusion requires one tile operand");
    assert(operationPlan.constantValue &&
           "multiply-constant recipe must record its scalar value");
    return createTileOpWithPlaceholderDstIndex<TileMulUnaryConstOp>(
        builder, loc, tileType, tileOperands[0], *operationPlan.constantValue);
  }

  // The recorded result tile type preserves a typecast's destination type
  // without consulting tensor IR after mutation begins.
  if (isa<TypecastOp>(sourceOp)) {
    return createTileOpWithPlaceholderDstIndex<TileTypecastOp>(
        builder, loc, tileType, tileOperands[0]);
  }

  if (isa<FillOp>(sourceOp)) {
    assert(operationPlan.constantValue &&
           "fill recipe must record its scalar value");
    return createTileOpWithPlaceholderDstIndex<TileFillOp>(
        builder, loc, tileType, *operationPlan.constantValue);
  }

  llvm_unreachable("planner must reject unsupported fused operations");
}

/// Emits the instrumentation placement selected before IR mutation.
///
/// Both direct and fused creation use this class so instrumentation ownership
/// and ordering cannot differ between the two creation strategies. Anchors are
/// source operations from the immutable plan; callers invoke `emitAfter` when
/// the corresponding tile recipe or tile store has been created.
class ComputeInstrumentationEmitter {
public:
  ComputeInstrumentationEmitter(
      PatternRewriter &rewriter,
      ArrayRef<ComputeInstrumentationPlacement> placements)
      : rewriter(rewriter), expectedCount(placements.size()) {
    for (const ComputeInstrumentationPlacement &placement : placements) {
      if (placement.after) {
        instrumentationAfter[placement.after].push_back(placement.operation);
      } else {
        leadingInstrumentation.push_back(placement.operation);
      }
    }
  }

  void emitLeading() {
    for (Operation *operation : leadingInstrumentation) {
      emit(operation);
    }
  }

  void emitAfter(Operation *anchor) {
    for (Operation *operation : instrumentationAfter.lookup(anchor)) {
      emit(operation);
    }
  }

  bool hasAfter(Operation *anchor) const {
    return !instrumentationAfter.lookup(anchor).empty();
  }

  bool emittedAll() const { return emittedCount == expectedCount; }

private:
  void emit(Operation *operation) {
    if (auto signpost = dyn_cast<SignpostOp>(operation)) {
      SignpostOp::create(rewriter, signpost.getLoc(), signpost.getNameAttr(),
                         signpost.getIsEndAttr());
    } else {
      assert(isa<DPrintOp>(operation) &&
             "planner recorded unsupported compute instrumentation");
      rewriter.clone(*operation);
    }
    ++emittedCount;
  }

  PatternRewriter &rewriter;
  DenseMap<Operation *, SmallVector<Operation *>> instrumentationAfter;
  SmallVector<Operation *> leadingInstrumentation;
  unsigned expectedCount = 0;
  unsigned emittedCount = 0;
};

//===----------------------------------------------------------------------===//
// Fused compute building
//===----------------------------------------------------------------------===//

/// Applies a precomputed fused creation. Fusion tracing, indexing, lifetime
/// legality, publication, and instrumentation placement are immutable plan
/// data; this function only constructs the selected tile-level operations.
static LogicalResult buildFusedCompute(Operation *sinkOp,
                                       PatternRewriter &rewriter,
                                       const ComputeOpCreationPlan &creation,
                                       const OutputPublicationPlan &outputs) {
  assert(creation.recipe == ComputeOpCreationRecipe::Fused &&
         "fused builder requires a fused creation recipe");
  const FusionTraceResult &trace = creation.trace;
  RankedTensorType type = creation.resultType;

  // Verify every recorded dependency before creating IR. This prevents plan
  // invalidation by an earlier rewrite from leaving a partially built compute.
  DenseSet<Value> availableValues;
  for (const FusedOperationPlan &operationPlan : creation.fusedOperations) {
    if (!llvm::equal(operationPlan.source->getOperands(),
                     operationPlan.sourceOperands)) {
      return rewriter.notifyMatchFailure(
          sinkOp, "fused operation operands changed after creation analysis");
    }
    if (llvm::any_of(operationPlan.operands,
                     [&](const FusedOperationOperand &operand) {
                       if (operand.rootInputIndex) {
                         unsigned inputIndex = *operand.rootInputIndex;
                         return inputIndex >= creation.inputs.size() ||
                                creation.inputs[inputIndex] != operand.value;
                       }
                       return !availableValues.contains(operand.value);
                     })) {
      return rewriter.notifyMatchFailure(
          sinkOp, "fused operation dependency is unavailable");
    }
    if (operationPlan.recipe == FusedOperationRecipe::DeferredMatmul ||
        operationPlan.recipe == FusedOperationRecipe::DeferredExpScale) {
      continue;
    }
    availableValues.insert(operationPlan.source->getResult(0));
  }
  if (!availableValues.contains(sinkOp->getResult(0))) {
    return rewriter.notifyMatchFailure(
        sinkOp, "fused plan does not produce the selected result");
  }

  Location loc = sinkOp->getLoc();
  SmallVector<Attribute> maps;
  for (AffineMap inputMap : creation.iteration.inputMaps) {
    maps.push_back(AffineMapAttr::get(inputMap));
  }
  for (size_t outputIndex = 0; outputIndex < outputs.dfbs.size();
       ++outputIndex) {
    maps.push_back(AffineMapAttr::get(creation.iteration.outputMap));
  }
  SmallVector<Attribute> iteratorTypes =
      buildIteratorTypeAttributes(rewriter, creation.iteration.iteratorTypes);

  // Position compute after all reserves by inserting before the last store.
  insertAtCreationAnchor(rewriter, outputs);

  // Create init tensors and attach to output CBs.
  // Use the first root input as exemplar for dynamic dims. For fill-only
  // chains with no root inputs, use tensor.empty directly (static shapes).
  SmallVector<Value> allInitAttached;
  SmallVector<Type> resultTypes;
  for (Value outputDFB : outputs.dfbs) {
    Value init = creation.inputs.empty()
                     ? tensor::EmptyOp::create(rewriter, loc, type.getShape(),
                                               type.getElementType())
                           .getResult()
                     : buildInitTensor(rewriter, loc, type, creation.inputs[0]);
    Value initAttached =
        AttachCBOp::create(rewriter, loc, init.getType(), init, outputDFB);
    allInitAttached.push_back(initAttached);
    resultTypes.push_back(type);
  }

  // Create ttl.compute op
  auto computeOp = ComputeOp::create(
      rewriter, loc, TypeRange(resultTypes), ValueRange(creation.inputs),
      ValueRange(allInitAttached), rewriter.getArrayAttr(maps),
      rewriter.getArrayAttr(iteratorTypes));

  // Build the body region
  Block *body = rewriter.createBlock(&computeOp.getBody());

  for (ttcore::TileType inputTileType : creation.inputTileTypes) {
    body->addArgument(inputTileType, loc);
  }
  for (size_t i = 0; i < outputs.dfbs.size(); ++i) {
    body->addArgument(creation.resultTileType, loc);
  }

  rewriter.setInsertionPointToStart(body);

  // Internal tensor results have one tile value. External roots use their
  // recorded block-argument indices because one root may have several maps.
  DenseMap<Value, Value> tensorToTile;

  assert(!trace.opsInOrder.empty() &&
         "buildFusedCompute requires non-empty opsInOrder");
  ComputeInstrumentationEmitter instrumentationEmitter(
      rewriter, creation.instrumentation);
  instrumentationEmitter.emitLeading();

  // Execute the recorded recipes in dependency order. Every non-deferred
  // result becomes available to subsequent recipes through `tensorToTile`.
  Value finalResult;
  for (const FusedOperationPlan &operationPlan : creation.fusedOperations) {
    Operation *op = operationPlan.source;

    SmallVector<Value> tileOperands;
    for (const FusedOperationOperand &operand : operationPlan.operands) {
      Value tileOperand = operand.rootInputIndex
                              ? body->getArgument(*operand.rootInputIndex)
                              : tensorToTile.lookup(operand.value);
      assert(tileOperand && "verified fused operand must have a tile value");
      tileOperands.push_back(tileOperand);
    }

    Value tileResult;
    switch (operationPlan.recipe) {
    case FusedOperationRecipe::TileOperation:
      tileResult = emitTileOpFor(rewriter, loc, operationPlan, tileOperands);
      break;
    case FusedOperationRecipe::InterTileBroadcast:
      tileResult = tileOperands.front();
      break;
    case FusedOperationRecipe::TileBroadcast:
      assert(operationPlan.tileBroadcast &&
             "tile-broadcast recipe must record its hardware kind");
      tileResult = createTileOpWithPlaceholderDstIndex<TileBcastOp>(
          rewriter, loc, operationPlan.resultTileType, tileOperands.front(),
          body->getArguments().back(), *operationPlan.tileBroadcast);
      break;
    case FusedOperationRecipe::Matmul: {
      auto matmul = createTileOpWithPlaceholderDstIndex<TileMatmulBlockOp>(
          rewriter, loc, operationPlan.resultTileType, tileOperands[0],
          tileOperands[1], Value());
      matmul.setTransposeRhs(operationPlan.transposeRhs);
      tileResult = matmul;
      break;
    }
    case FusedOperationRecipe::DeferredMatmul:
      assert(!instrumentationEmitter.hasAfter(op) &&
             "instrumented matmul must not be folded into its user");
      continue;
    case FusedOperationRecipe::DeferredExpScale:
      instrumentationEmitter.emitAfter(op);
      continue;
    case FusedOperationRecipe::MatmulAccumulator: {
      assert(operationPlan.foldedMatmul && operationPlan.accumulator &&
             "accumulating matmul recipe must record the folded expression");
      auto matmul = createTileOpWithPlaceholderDstIndex<TileMatmulBlockOp>(
          rewriter, loc, operationPlan.resultTileType, tileOperands[0],
          tileOperands[1], tileOperands[2]);
      matmul.setTransposeRhs(operationPlan.transposeRhs);
      tileResult = matmul;
      break;
    }
    }
    tensorToTile[op->getResult(0)] = tileResult;
    finalResult = tileResult;
    instrumentationEmitter.emitAfter(op);
  }

  // Output stores and their instrumentation retain their complete source
  // order, including distinct signpost scopes around multiple stores.
  for (StoreOp store : outputs.stores) {
    emitTileStore(rewriter, loc, finalResult, computeOp, store, creation);
    instrumentationEmitter.emitAfter(store);
  }
  assert(instrumentationEmitter.emittedAll() &&
         "every planned instrumentation operation must have a supported "
         "source anchor");

  YieldOp::create(rewriter, loc);
  SmallVector<CBPushOp> replacedPushes;
  replaceOutputPushesBeforeCompute(rewriter, computeOp, outputs,
                                   replacedPushes);
  eraseAbsorbedOutputOps(rewriter, outputs, computeOp, replacedPushes);
  rewriter.replaceOp(sinkOp, computeOp.getResult(0));

  // Erase the fused ops in reverse topological order (sink to roots).
  // This ensures each op's users are erased before the op itself.
  for (auto it = trace.opsInOrder.rbegin(); it != trace.opsInOrder.rend();
       ++it) {
    Operation *op = *it;
    if (op != sinkOp && op->use_empty()) {
      rewriter.eraseOp(op);
    }
  }

  // Erase the original instrumentation after cloning it into the compute body.
  for (const ComputeInstrumentationPlacement &placement :
       creation.instrumentation) {
    rewriter.eraseOp(placement.operation);
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Lowering to ttl.compute with tile ops
//===----------------------------------------------------------------------===//

/// Applies one direct creation plan. The callback selects the tile operation;
/// input selection, iteration semantics, output transactions, and lifetime
/// legality have already been decided from immutable IR.
static LogicalResult buildComputeFromInputs(
    Operation *op, PatternRewriter &rewriter,
    ComputeOpCreationRecipe expectedRecipe,
    const KernelComputeOpCreationPlan &kernelPlan,
    llvm::function_ref<Value(OpBuilder &, Location, Type, Block *,
                             const ComputeOpCreationPlan &)>
        emitTileOp) {
  const ComputeOpCreationPlan *creation = nullptr;
  if (failed(getCreationPlan(op, rewriter, kernelPlan, creation))) {
    return failure();
  }
  if (creation->kind != ComputeOpCreationKind::Direct ||
      creation->recipe != expectedRecipe) {
    return rewriter.notifyMatchFailure(
        op, "direct creation differs from the analyzed creation plan");
  }
  OutputPublicationPlan outputs;
  if (failed(resolveCurrentOutputs(op, rewriter, *creation, outputs))) {
    return failure();
  }

  Location loc = op->getLoc();
  ValueRange inputs(creation->inputs);
  RankedTensorType outputType = creation->resultType;

  SmallVector<Attribute> maps;
  for (AffineMap inputMap : creation->iteration.inputMaps) {
    maps.push_back(AffineMapAttr::get(inputMap));
  }
  for (size_t outputIndex = 0; outputIndex < outputs.dfbs.size();
       ++outputIndex) {
    maps.push_back(AffineMapAttr::get(creation->iteration.outputMap));
  }
  SmallVector<Attribute> iteratorTypes =
      buildIteratorTypeAttributes(rewriter, creation->iteration.iteratorTypes);

  insertAtCreationAnchor(rewriter, outputs);

  SmallVector<Value> allInitAttached;
  SmallVector<Type> resultTypes;
  for (Value outputDFB : outputs.dfbs) {
    Value init =
        inputs.empty()
            ? tensor::EmptyOp::create(rewriter, loc, outputType.getShape(),
                                      outputType.getElementType())
                  .getResult()
            : buildInitTensor(rewriter, loc, outputType, inputs[0]);
    Value initAttached =
        AttachCBOp::create(rewriter, loc, init.getType(), init, outputDFB);
    allInitAttached.push_back(initAttached);
    resultTypes.push_back(outputType);
  }

  auto computeOp = ComputeOp::create(rewriter, loc, TypeRange(resultTypes),
                                     inputs, ValueRange(allInitAttached),
                                     rewriter.getArrayAttr(maps),
                                     rewriter.getArrayAttr(iteratorTypes));

  Block *body = rewriter.createBlock(&computeOp.getBody());
  for (ttcore::TileType inputTileType : creation->inputTileTypes) {
    body->addArgument(inputTileType, loc);
  }
  for (size_t i = 0; i < outputs.dfbs.size(); ++i) {
    body->addArgument(creation->resultTileType, loc);
  }

  rewriter.setInsertionPointToStart(body);
  ComputeInstrumentationEmitter instrumentationEmitter(
      rewriter, creation->instrumentation);
  instrumentationEmitter.emitLeading();
  Value result =
      emitTileOp(rewriter, loc, creation->resultTileType, body, *creation);
  instrumentationEmitter.emitAfter(op);
  for (StoreOp store : outputs.stores) {
    emitTileStore(rewriter, loc, result, computeOp, store, *creation);
    instrumentationEmitter.emitAfter(store);
  }
  assert(instrumentationEmitter.emittedAll() &&
         "every planned instrumentation operation must have a supported "
         "source anchor");
  YieldOp::create(rewriter, loc);
  SmallVector<CBPushOp> replacedPushes;
  replaceOutputPushesBeforeCompute(rewriter, computeOp, outputs,
                                   replacedPushes);
  eraseAbsorbedOutputOps(rewriter, outputs, computeOp, replacedPushes);
  rewriter.replaceOp(op, computeOp.getResult(0));
  for (const ComputeInstrumentationPlacement &placement :
       creation->instrumentation) {
    rewriter.eraseOp(placement.operation);
  }
  return success();
}

/// Try fusion for an op whose inputs are not all CB-attached.
/// Returns success if fusion was performed, failure otherwise.
static LogicalResult tryFusion(Operation *op, PatternRewriter &rewriter,
                               const KernelComputeOpCreationPlan &kernelPlan) {
  const ComputeOpCreationPlan *creation = nullptr;
  if (failed(getCreationPlan(op, rewriter, kernelPlan, creation))) {
    return failure();
  }
  if (creation->kind == ComputeOpCreationKind::Fused) {
    OutputPublicationPlan outputs;
    if (failed(resolveCurrentOutputs(op, rewriter, *creation, outputs))) {
      return failure();
    }
    return buildFusedCompute(op, rewriter, *creation, outputs);
  }
  return rewriter.notifyMatchFailure(op, "operation has no fusable expression");
}

/// Build a ttl.compute op with a single binary tile operation in the body.
/// Inputs must already be attached to CBs via ttl.attach_cb.
/// Output CBs are the reserved CBs to which the op's result is stored.
template <typename TileOp>
static LogicalResult
buildBinaryCompute(Operation *op, PatternRewriter &rewriter, Value lhs,
                   Value rhs, const KernelComputeOpCreationPlan &kernelPlan) {
  if (!getAttachedCB(lhs) || !getAttachedCB(rhs)) {
    return tryFusion(op, rewriter, kernelPlan);
  }

  return buildComputeFromInputs(
      op, rewriter, ComputeOpCreationRecipe::Elementwise, kernelPlan,
      [](OpBuilder &builder, Location location, Type tileType, Block *body,
         const ComputeOpCreationPlan &) {
        return createTileOpWithPlaceholderDstIndex<TileOp>(
            builder, location, tileType, body->getArgument(0),
            body->getArgument(1));
      });
}

/// Build a ttl.compute op with a single unary tile operation in the body.
/// Input must already be attached to a CB via ttl.attach_cb.
/// Output CBs are the reserved CBs to which the op's result is stored.
template <typename TileOp>
static LogicalResult
buildUnaryCompute(Operation *op, PatternRewriter &rewriter, Value input,
                  const KernelComputeOpCreationPlan &kernelPlan) {
  if (!getAttachedCB(input)) {
    return tryFusion(op, rewriter, kernelPlan);
  }

  return buildComputeFromInputs(
      op, rewriter, ComputeOpCreationRecipe::Elementwise, kernelPlan,
      [](OpBuilder &builder, Location location, Type tileType, Block *body,
         const ComputeOpCreationPlan &) {
        return createTileOpWithPlaceholderDstIndex<TileOp>(
            builder, location, tileType, body->getArgument(0));
      });
}

namespace {
//===----------------------------------------------------------------------===//
// Templated Elementwise Lowering Patterns
//===----------------------------------------------------------------------===//

/// Base for rewrites whose common creation legality is precomputed.
template <typename SourceOp>
struct PlannedComputeRewritePattern : OpRewritePattern<SourceOp> {
  PlannedComputeRewritePattern(MLIRContext *context,
                               const KernelComputeOpCreationPlan &kernelPlan)
      : OpRewritePattern<SourceOp>(context), kernelPlan(kernelPlan) {}

protected:
  const KernelComputeOpCreationPlan &kernelPlan;
};

/// Pattern for binary elementwise ops: TTL tensor op -> ttl.compute with tile
/// op.
template <typename TTLOp, typename TileOp>
struct LowerBinaryToCompute : PlannedComputeRewritePattern<TTLOp> {
  using PlannedComputeRewritePattern<TTLOp>::PlannedComputeRewritePattern;

  LogicalResult matchAndRewrite(TTLOp op,
                                PatternRewriter &rewriter) const override {
    return buildBinaryCompute<TileOp>(op.getOperation(), rewriter, op.getLhs(),
                                      op.getRhs(), this->kernelPlan);
  }
};

/// Pattern for unary elementwise ops: TTL tensor op -> ttl.compute with tile
/// op.
template <typename TTLOp, typename TileOp>
struct LowerUnaryToCompute : PlannedComputeRewritePattern<TTLOp> {
  using PlannedComputeRewritePattern<TTLOp>::PlannedComputeRewritePattern;

  LogicalResult matchAndRewrite(TTLOp op,
                                PatternRewriter &rewriter) const override {
    return buildUnaryCompute<TileOp>(op.getOperation(), rewriter, op.getInput(),
                                     this->kernelPlan);
  }
};

/// Pattern for ttl.exp: TTL tensor op -> ttl.compute with ttl.tile_exp,
/// forwarding the exp hardware flags. Dedicated (not the generic unary
/// template) because exp carries extra attributes.
struct LowerExpToCompute : PlannedComputeRewritePattern<ExpOp> {
  using PlannedComputeRewritePattern<ExpOp>::PlannedComputeRewritePattern;

  LogicalResult matchAndRewrite(ExpOp op,
                                PatternRewriter &rewriter) const override {
    if (!getAttachedCB(op.getInput())) {
      return tryFusion(op.getOperation(), rewriter, this->kernelPlan);
    }

    return buildComputeFromInputs(
        op, rewriter, ComputeOpCreationRecipe::Elementwise, this->kernelPlan,
        [](OpBuilder &builder, Location location, Type tileType, Block *body,
           const ComputeOpCreationPlan &creation) {
          assert(creation.expFlags &&
                 "exp recipe must record its hardware flags");
          return emitExpTileOp(builder, location, tileType,
                               body->getArgument(0), *creation.expFlags);
        });
  }
};

//===----------------------------------------------------------------------===//
// Block Broadcast Lowering Pattern
//===----------------------------------------------------------------------===//

/// Trace whether the value feeding a broadcast came from a reduction through
/// a CB push/wait cycle, and if so return which reduce dimension was used.
///
/// Follows the chain:
///   bcast input -> attach_cb -> cb_wait [CB] <- cb_push <- store <- reduce
/// and returns the ReduceDim of the producing reduce.
///
/// The correct hardware BcastType depends on the tile data layout left by
/// the producing reduce (tt-metal llk_unpack_AB.h L72-114):
///   REDUCE_SCALAR -> valid data at element [0,0]
///   REDUCE_COL    -> valid data in row 0
///   REDUCE_ROW    -> valid data in column 0
/// The derived BcastType is used to select the correct hardware unpack type;
/// a mismatch with the producing reduce replicates garbage (#444).
///
/// TODO(#449): replace this tracing with a structured approach (e.g.,
/// propagate reduce dim as an attribute during lowering).
///
/// Returns std::nullopt when no unique reduce can be traced (no CB, ambiguous
/// stores, non-reduce producer, etc.). Returns a ReduceDim when a unique
/// reduce was successfully traced.
static std::optional<ttkernel::ReduceDim> getInputReduceDim(Value bcastInput) {
  Value cb = getAttachedCB(bcastInput);
  if (!cb) {
    return std::nullopt;
  }

  // Find the unique store to this CB in the enclosing function.  Walking the
  // function rather than just the immediate block handles cases where the
  // store is inside a nested region (e.g., nested with-stmt scopes).
  StoreOp foundStore;
  bool ambiguous = false;
  auto *defOp = bcastInput.getDefiningOp();
  if (!defOp) {
    return std::nullopt;
  }
  auto enclosingFunc = defOp->getParentOfType<func::FuncOp>();
  if (!enclosingFunc) {
    return std::nullopt;
  }
  enclosingFunc.walk([&](StoreOp storeOp) {
    if (ambiguous || getAttachedCB(storeOp.getView()) != cb) {
      return;
    }
    if (foundStore) {
      ambiguous = true;
      return;
    }
    foundStore = storeOp;
  });
  if (!foundStore || ambiguous) {
    return std::nullopt;
  }

  auto reduceOp = foundStore.getTensor().getDefiningOp<ReduceOp>();
  if (!reduceOp) {
    return std::nullopt;
  }

  auto inputType = getTensorType(reduceOp.getInput());
  if (!inputType) {
    return std::nullopt;
  }
  auto reduceDim = getReduceDimension(reduceOp.getDims(), inputType.getRank());
  if (failed(reduceDim)) {
    return std::nullopt;
  }
  return *reduceDim;
}

/// Validate a single BlockBroadcastOp. Called from runOnOperation() before
/// patterns run, so emitOpError is safe (not inside a pattern rewriter).
static LogicalResult validateBlockBroadcastOp(BlockBroadcastOp op,
                                              TTLToComputeMode mode) {
  auto outputType = getTensorType(op.getResult());
  auto inputType = getTensorType(op.getInput());
  if (!outputType || !inputType) {
    return success(); // pattern will handle gracefully
  }

  if (!getAttachedCB(op.getInput())) {
    if (mode == TTLToComputeMode::ProducerCreation) {
      return success();
    }
    return op.emitOpError(
        "broadcast input must come directly from a circular buffer, not from "
        "an elementwise result; move the broadcast to its own compute block "
        "or make it the first operation in a fused sequence");
  }

  int64_t rank = inputType.getRank();

  // Validate broadcast dims vs. producing reduce (#444). The derived
  // BcastType determines the hardware unpack type and must agree with the
  // tile layout left by any directly-producing reduce.
  if (auto reduceDim = getInputReduceDim(op.getInput())) {
    auto tileBcastType = getTileBroadcastType(op.getDims(), rank);
    if (!tileBcastType) {
      return op.emitOpError(
          "broadcast feeds an inter-tile-only pattern from a reduce result; "
          "the innermost dims must participate in the broadcast");
    }
    BcastType requiredBcastType;
    StringRef requiredKind, requiredDims;
    switch (*reduceDim) {
    case ttkernel::ReduceDim::Scalar:
      requiredBcastType = BcastType::Scalar;
      requiredKind = "scalar";
      requiredDims = "[-2, -1]";
      break;
    case ttkernel::ReduceDim::Col:
      requiredBcastType = BcastType::Row;
      requiredKind = "row";
      requiredDims = "[-2]";
      break;
    case ttkernel::ReduceDim::Row:
      requiredBcastType = BcastType::Col;
      requiredKind = "column";
      requiredDims = "[-1]";
      break;
    }
    if (*tileBcastType != requiredBcastType) {
      return op.emitOpError("broadcast dims are incompatible with the "
                            "producing reduce; need ")
             << requiredKind << " broadcast (dims=" << requiredDims << ")";
    }
  }

  return success();
}

/// Pattern for block broadcast: TTL tensor op -> ttl.compute body. Supports
/// arbitrary rank; inter-tile broadcast is handled by constant-0 entries in
/// the input affine map, and intra-tile broadcast on the innermost two dims
/// is handled by ttl.tile_bcast in the body.
struct LowerBlockBroadcastToCompute
    : PlannedComputeRewritePattern<BlockBroadcastOp> {
  using PlannedComputeRewritePattern<
      BlockBroadcastOp>::PlannedComputeRewritePattern;

  LogicalResult matchAndRewrite(BlockBroadcastOp op,
                                PatternRewriter &rewriter) const override {
    return buildComputeFromInputs(
        op, rewriter, ComputeOpCreationRecipe::BlockBroadcast, this->kernelPlan,
        [](OpBuilder &builder, Location location, Type tileType, Block *body,
           const ComputeOpCreationPlan &creation) -> Value {
          if (!creation.tileBroadcast) {
            return body->getArgument(0);
          }
          // The output block argument supplies the DFB required by hardware
          // pack reconfiguration; the input alone cannot identify it.
          return createTileOpWithPlaceholderDstIndex<TileBcastOp>(
              builder, location, tileType, body->getArgument(0),
              body->getArgument(1), *creation.tileBroadcast);
        });
  }
};

//===----------------------------------------------------------------------===//
// Matmul Lowering
//===----------------------------------------------------------------------===//

/// Lowers ttl.matmul to ttl.compute with ttl.tile_matmul_block in the body.
/// When the matmul feeds into an elementwise op, defers to let
/// buildFusedCompute handle the full chain (including matmul+add fusion
/// into 3-operand tile_matmul_block via the deferred-matmul fold).
/// Standalone matmul (result stored directly) is lowered here with a 3D
/// [M, N, K] iteration space.
struct LowerMatmulToCompute : PlannedComputeRewritePattern<MatmulOp> {
  using PlannedComputeRewritePattern<MatmulOp>::PlannedComputeRewritePattern;

  LogicalResult matchAndRewrite(MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    if (!getAttachedCB(lhs) || !getAttachedCB(rhs)) {
      return rewriter.notifyMatchFailure(op,
                                         "matmul inputs must be CB-attached");
    }

    // Defer when the matmul feeds into an elementwise op (e.g., add, relu,
    // sub). The downstream op's fusion (buildFusedCompute) handles the full
    // chain with matmul-aware 3D indexing maps and the deferred-matmul fold.
    if (op.getResult().hasOneUse() &&
        isElementwiseOp(*op.getResult().getUsers().begin())) {
      return rewriter.notifyMatchFailure(op, "deferring matmul to fusion");
    }

    return buildComputeFromInputs(
        op, rewriter, ComputeOpCreationRecipe::Matmul, this->kernelPlan,
        [](OpBuilder &builder, Location location, Type tileType, Block *body,
           const ComputeOpCreationPlan &creation) {
          auto tileOp = createTileOpWithPlaceholderDstIndex<TileMatmulBlockOp>(
              builder, location, tileType, body->getArgument(0),
              body->getArgument(1), Value());
          tileOp.setTransposeRhs(creation.transposeRhs);
          return tileOp;
        });
  }
};

//===----------------------------------------------------------------------===//
// Store Lowering
//===----------------------------------------------------------------------===//

/// Lowers passthrough ttl.store (CB-attached input) by creating a compute
/// with tile_store. Stores whose input comes from an elementwise op are
/// already erased when their producers are incorporated into a ComputeOp.
struct LowerStoreToCompute : OpRewritePattern<StoreOp> {
  LowerStoreToCompute(MLIRContext *context,
                      const KernelComputeOpCreationPlan &kernelPlan)
      : OpRewritePattern<StoreOp>(context), kernelPlan(kernelPlan) {}

  LogicalResult matchAndRewrite(StoreOp op,
                                PatternRewriter &rewriter) const override {
    FailureOr<const PassthroughStorePlan *> planned = kernelPlan.get(op);
    if (failed(planned)) {
      return rewriter.notifyMatchFailure(op, kernelPlan.getRejectionReason(op));
    }
    const PassthroughStorePlan &plan = **planned;
    if (op.getTensor() != plan.input || op.getView() != plan.outputView) {
      return rewriter.notifyMatchFailure(
          op, "store operands changed after ComputeOp creation analysis");
    }
    Value input = plan.input;
    RankedTensorType inputType = plan.tensorType;

    Location loc = op.getLoc();
    SmallVector<Attribute> maps = {
        AffineMapAttr::get(plan.iteration.inputMaps.front()),
        AffineMapAttr::get(plan.iteration.outputMap)};
    SmallVector<Attribute> iteratorTypes =
        buildIteratorTypeAttributes(rewriter, plan.iteration.iteratorTypes);

    Value init = buildInitTensor(rewriter, loc, inputType, input);
    Value initAttached =
        AttachCBOp::create(rewriter, loc, init.getType(), init, plan.outputDFB);

    auto computeOp = ComputeOp::create(
        rewriter, loc, TypeRange{inputType}, ValueRange{input},
        ValueRange{initAttached}, rewriter.getArrayAttr(maps),
        rewriter.getArrayAttr(iteratorTypes));

    Block *body = rewriter.createBlock(&computeOp.getBody());
    body->addArgument(plan.tileType, loc);
    body->addArgument(plan.tileType, loc);

    rewriter.setInsertionPointToEnd(body);
    SmallVector<Value> iterIndices =
        getOrCreateIterIndices(rewriter, computeOp);
    SmallVector<Value> storeIndices =
        applyIndexingMap(rewriter, loc, plan.iteration.outputMap, iterIndices);
    createTileOpWithPlaceholderDstIndex<TileStoreOp>(
        rewriter, loc, body->getArgument(0), plan.outputView, storeIndices);
    YieldOp::create(rewriter, loc);

    for (AttachCBOp association : plan.outputAssociations) {
      rewriter.replaceOp(association, computeOp.getResult(0));
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  const KernelComputeOpCreationPlan &kernelPlan;
};

//===----------------------------------------------------------------------===//
// Pattern Type Aliases - Generated from TTLElementwiseOps.def (tile-based)
//===----------------------------------------------------------------------===//
// Reduce Lowering
//===----------------------------------------------------------------------===//

/// Lowers ttl.reduce to ttl.compute with ttl.tile_reduce in the body.
/// The iteration domain covers the full input shape with reduction iterators
/// on the reduced dimensions.
struct LowerReduceToCompute : PlannedComputeRewritePattern<ReduceOp> {
  using PlannedComputeRewritePattern<ReduceOp>::PlannedComputeRewritePattern;

  LogicalResult matchAndRewrite(ReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = getTensorType(op.getInput());
    if (!inputType || !getTensorType(op.getResult())) {
      return failure();
    }

    if (!getAttachedCB(op.getScaler())) {
      return rewriter.notifyMatchFailure(op,
                                         "reduce scaler must be CB-attached");
    }

    if (!getAttachedCB(op.getInput())) {
      return rewriter.notifyMatchFailure(op,
                                         "reduce input must be DFB-attached");
    }

    return buildComputeFromInputs(
        op, rewriter, ComputeOpCreationRecipe::Reduce, this->kernelPlan,
        [](OpBuilder &builder, Location location, Type tileType, Block *body,
           const ComputeOpCreationPlan &creation) {
          assert(creation.reduceType && creation.reduceDimension &&
                 "reduce recipe must record function and dimension");
          return createTileOpWithPlaceholderDstIndex<TileReduceOp>(
              builder, location, tileType, body->getArgument(0),
              body->getArgument(1), body->getArgument(2), *creation.reduceType,
              *creation.reduceDimension);
        });
  }
};

/// Lower ttl.mul_unary_const to ttl.compute with a single tile operation, or
/// fuse it with its producer when the input is not attached to a dataflow
/// buffer.
struct LowerMulUnaryConstToCompute
    : PlannedComputeRewritePattern<MulUnaryConstOp> {
  using PlannedComputeRewritePattern<
      MulUnaryConstOp>::PlannedComputeRewritePattern;

  LogicalResult matchAndRewrite(MulUnaryConstOp op,
                                PatternRewriter &rewriter) const override {
    if (!getAttachedCB(op.getInput())) {
      return tryFusion(op.getOperation(), rewriter, this->kernelPlan);
    }

    return buildComputeFromInputs(
        op, rewriter, ComputeOpCreationRecipe::MulUnaryConst, this->kernelPlan,
        [](OpBuilder &builder, Location location, Type tileType, Block *body,
           const ComputeOpCreationPlan &creation) {
          assert(creation.constantValue &&
                 "multiply-constant recipe must record its scalar value");
          return createTileOpWithPlaceholderDstIndex<TileMulUnaryConstOp>(
              builder, location, tileType, body->getArgument(0),
              *creation.constantValue);
        });
  }
};

//===----------------------------------------------------------------------===//
// Fill Lowering
//===----------------------------------------------------------------------===//

struct LowerFillToCompute : PlannedComputeRewritePattern<FillOp> {
  using PlannedComputeRewritePattern<FillOp>::PlannedComputeRewritePattern;

  LogicalResult matchAndRewrite(FillOp op,
                                PatternRewriter &rewriter) const override {
    return buildComputeFromInputs(
        op, rewriter, ComputeOpCreationRecipe::Fill, this->kernelPlan,
        [](OpBuilder &builder, Location location, Type tileType, Block *,
           const ComputeOpCreationPlan &creation) {
          assert(creation.constantValue &&
                 "fill recipe must record its scalar value");
          return createTileOpWithPlaceholderDstIndex<TileFillOp>(
              builder, location, tileType, *creation.constantValue);
        });
  }
};

//===----------------------------------------------------------------------===//
// Typecast Lowering
//===----------------------------------------------------------------------===//

/// Lowers ttl.typecast to ttl.compute with ttl.tile_typecast in the body.
struct LowerTypecastToCompute : PlannedComputeRewritePattern<TypecastOp> {
  using PlannedComputeRewritePattern<TypecastOp>::PlannedComputeRewritePattern;

  LogicalResult matchAndRewrite(TypecastOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = getTensorType(op.getResult());
    if (!resultType) {
      return failure();
    }

    if (op.getInput().getType() == op.getResult().getType()) {
      const ComputeOpCreationPlan *creation = nullptr;
      if (failed(getCreationPlan(op, ComputeOpCreationKind::Elide,
                                 ComputeOpCreationRecipe::Elide,
                                 ValueRange{op.getInput()}, rewriter,
                                 this->kernelPlan, creation))) {
        return failure();
      }
      rewriter.replaceOp(op, op.getInput());
      return success();
    }

    if (!getAttachedCB(op.getInput())) {
      return tryFusion(op, rewriter, this->kernelPlan);
    }

    return buildComputeFromInputs(
        op, rewriter, ComputeOpCreationRecipe::Typecast, this->kernelPlan,
        [](OpBuilder &builder, Location location, Type outputTileType,
           Block *body, const ComputeOpCreationPlan &) {
          return createTileOpWithPlaceholderDstIndex<TileTypecastOp>(
              builder, location, outputTileType, body->getArgument(0));
        });
  }
};

//===----------------------------------------------------------------------===//
// Transpose Lowering
//===----------------------------------------------------------------------===//

/// Lowers ttl.transpose to ttl.compute with ttl.tile_transpose in the body.
/// Input indexing uses swapped dimensions: (d0, d1) -> (d1, d0).
struct LowerTransposeToCompute : PlannedComputeRewritePattern<TransposeOp> {
  using PlannedComputeRewritePattern<TransposeOp>::PlannedComputeRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter &rewriter) const override {
    auto inputType = getTensorType(op.getInput());
    auto resultType = getTensorType(op.getResult());
    if (!inputType || !resultType) {
      return failure();
    }

    if (!getAttachedCB(op.getInput())) {
      return tryFusion(op, rewriter, this->kernelPlan);
    }

    return buildComputeFromInputs(
        op, rewriter, ComputeOpCreationRecipe::Transpose, this->kernelPlan,
        [](OpBuilder &builder, Location location, Type tileType, Block *body,
           const ComputeOpCreationPlan &) {
          return createTileOpWithPlaceholderDstIndex<TileTransposeOp>(
              builder, location, tileType, body->getArgument(0),
              body->getArgument(1));
        });
  }
};

//===----------------------------------------------------------------------===//

// Generate type aliases for binary operations using tile ops
// (TTK_INIT and TTK_COMPUTE are unused here, only needed for TTKernel lowering)
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)             \
  using Lower##TTL_OP = LowerBinaryToCompute<TTL_OP##Op, TILE_OP>;
#define TTL_BINARY_TILE_OP_MINMAX(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)      \
  using Lower##TTL_OP = LowerBinaryToCompute<TTL_OP##Op, TILE_OP>;
// Generate type aliases for unary operations using tile ops
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  using Lower##TTL_OP = LowerUnaryToCompute<TTL_OP##Op, TILE_OP>;
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

//===----------------------------------------------------------------------===//
// Pass Implementations
//===----------------------------------------------------------------------===//

static void populateTTLToComputePatternsForMode(
    RewritePatternSet &patterns, TTLToComputeMode mode,
    const KernelComputeOpCreationPlan &kernelPlan);

static LogicalResult
validateExistingComputeOps(func::FuncOp kernel,
                           const ComputeTargetEnvironment &target) {
  bool hasErrors = false;
  kernel.walk([&](ComputeOp compute) {
    bool requiresComputeShape = false;
    compute.walk([&](Operation *operation) {
      std::optional<ComputePrimitive> primitive =
          getComputePrimitive(operation);
      if (!primitive) {
        return;
      }
      requiresComputeShape |= *primitive != ComputePrimitive::Passthrough;
      std::string failureReason;
      if (failed(target.validateOperation(operation, failureReason))) {
        operation->emitOpError(failureReason);
        hasErrors = true;
      }
    });
    if (requiresComputeShape) {
      for (BlockArgument argument : compute.getBody().front().getArguments()) {
        auto tileType = dyn_cast<ttcore::TileType>(argument.getType());
        if (!tileType) {
          continue;
        }
        std::string failureReason;
        if (failed(target.validateKernelTileType(tileType, failureReason))) {
          compute.emitOpError() << "block argument " << argument.getArgNumber()
                                << " " << failureReason;
          hasErrors = true;
        }
      }
    }
  });
  return failure(hasErrors);
}

static LogicalResult runTTLToCompute(func::FuncOp kernel,
                                     TTLToComputeMode mode) {
  if (kernel.isExternal()) {
    return success();
  }

  std::string targetFailureReason;
  FailureOr<std::unique_ptr<ComputeTargetEnvironment>> target =
      ComputeTargetEnvironment::get(kernel, targetFailureReason);
  if (failed(target)) {
    kernel.emitOpError(targetFailureReason);
    return failure();
  }
  if (failed(validateExistingComputeOps(kernel, **target))) {
    return failure();
  }

  // Validate bcast ops before running patterns. Emitting errors here (outside
  // a pattern rewriter) is safe for the Python bindings.
  bool hasErrors = false;
  kernel.walk([&](BlockBroadcastOp op) {
    if (failed(validateBlockBroadcastOp(op, mode))) {
      hasErrors = true;
    }
  });
  if (hasErrors) {
    return failure();
  }

  PlanningResult<std::unique_ptr<DFBValueLifetimeAnalysis>> plannedLifetimes =
      DFBValueLifetimeAnalysis::create(kernel);
  if (plannedLifetimes.isInvalidIR()) {
    const PlanningDiagnostic &diagnostic = plannedLifetimes.getInvalidIR();
    diagnostic.operation->emitOpError(diagnostic.message);
    return failure();
  }
  assert(plannedLifetimes.isPlanned() &&
         "lifetime analysis has no recoverable rejection");
  std::unique_ptr<DFBValueLifetimeAnalysis> lifetimes =
      std::move(plannedLifetimes).takePlan();
  ComputeOpCreationPlanner creationPlanner(kernel, *lifetimes, **target);
  PlanningResult<KernelComputeOpCreationPlan> plannedKernel =
      creationPlanner.build();
  if (plannedKernel.isInvalidIR()) {
    const PlanningDiagnostic &planningDiagnostic = plannedKernel.getInvalidIR();
    planningDiagnostic.operation->emitOpError(planningDiagnostic.message);
    return failure();
  }
  assert(plannedKernel.isPlanned() &&
         "kernel ComputeOp creation planning has no recoverable rejection");
  KernelComputeOpCreationPlan kernelPlan = std::move(plannedKernel).takePlan();

  if (mode == TTLToComputeMode::FinalConversion &&
      !kernelPlan.getUnassignedStores().empty()) {
    StoreOp store = kernelPlan.getUnassignedStores().front();
    PlanningDiagnostic diagnostic =
        kernelPlan.getUnassignedStoreDiagnostic(store);
    diagnostic.operation->emitOpError(
        "cannot lower tensor store to ttl.compute: ")
        << diagnostic.message;
    return failure();
  }

  SmallVector<ComputeOpCreationWarning> emittedWarnings;
  for (Operation *source : kernelPlan.getCreationOrder()) {
    const ComputeOpCreationPlan &creation =
        kernelPlan.getAnalyzedCreation(source);
    for (const ComputeOpCreationWarning &warning : creation.warnings) {
      bool alreadyEmitted = llvm::any_of(
          emittedWarnings, [&](const ComputeOpCreationWarning &emitted) {
            return emitted.operation == warning.operation &&
                   emitted.kind == warning.kind;
          });
      if (!alreadyEmitted) {
        warning.operation->emitWarning(
            getComputeOpCreationWarningMessage(warning.kind));
        emittedWarnings.push_back(warning);
      }
    }
  }

  RewritePatternSet patterns(kernel.getContext());
  populateTTLToComputePatternsForMode(patterns, mode, kernelPlan);
  FrozenRewritePatternSet frozenPatterns(std::move(patterns));

  GreedyRewriteConfig candidateConfig;
  candidateConfig.setStrictness(GreedyRewriteStrictness::ExistingOps);
  candidateConfig.enableFolding(false);
  for (Operation *source : kernelPlan.getCreationOrder()) {
    if (failed(applyOpPatternsGreedily({source}, frozenPatterns,
                                       candidateConfig))) {
      return failure();
    }
    if (containsOperation(kernel, source)) {
      source->emitError("failed to apply verified ComputeOp creation plan");
      return failure();
    }
  }

  GreedyRewriteConfig remainingConfig;
  remainingConfig.enableFolding(false);
  return applyPatternsGreedily(kernel, frozenPatterns, remainingConfig);
}

struct TTLProducerComputeCreationPass
    : public tt::ttl::impl::TTLProducerComputeCreationBase<
          TTLProducerComputeCreationPass> {
  using tt::ttl::impl::TTLProducerComputeCreationBase<
      TTLProducerComputeCreationPass>::TTLProducerComputeCreationBase;

  void runOnOperation() override {
    if (failed(runTTLToCompute(getOperation(),
                               TTLToComputeMode::ProducerCreation))) {
      return signalPassFailure();
    }
  }
};

struct TTLConvertTTLToComputePass
    : public tt::ttl::impl::TTLConvertTTLToComputeBase<
          TTLConvertTTLToComputePass> {
  using tt::ttl::impl::TTLConvertTTLToComputeBase<
      TTLConvertTTLToComputePass>::TTLConvertTTLToComputeBase;

  void runOnOperation() override {
    if (failed(runTTLToCompute(getOperation(),
                               TTLToComputeMode::FinalConversion))) {
      return signalPassFailure();
    }
  }
};

static void populateTTLToComputePatternsForMode(
    RewritePatternSet &patterns, TTLToComputeMode mode,
    const KernelComputeOpCreationPlan &kernelPlan) {
  MLIRContext *ctx = patterns.getContext();

  // Register patterns for lowering to ttl.compute with tile ops.
  // These are generated from TTLElementwiseOps.def using tile-based mappings.
  // (TTK_INIT and TTK_COMPUTE are unused here, only needed for TTKernel
  // lowering)
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)             \
  patterns.add<Lower##TTL_OP>(ctx, kernelPlan);
#define TTL_BINARY_TILE_OP_MINMAX(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)      \
  patterns.add<Lower##TTL_OP>(ctx, kernelPlan);
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  patterns.add<Lower##TTL_OP>(ctx, kernelPlan);
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  patterns.add<LowerExpToCompute>(ctx, kernelPlan);
  patterns.add<LowerBlockBroadcastToCompute>(ctx, kernelPlan);
  patterns.add<LowerMatmulToCompute>(ctx, kernelPlan);
  patterns.add<LowerReduceToCompute>(ctx, kernelPlan);
  patterns.add<LowerMulUnaryConstToCompute>(ctx, kernelPlan);
  patterns.add<LowerTransposeToCompute>(ctx, kernelPlan);
  patterns.add<LowerTypecastToCompute>(ctx, kernelPlan);
  patterns.add<LowerFillToCompute>(ctx, kernelPlan);
  patterns.add<LowerStoreToCompute>(ctx, kernelPlan);
}

} // namespace

} // namespace mlir::tt::ttl
