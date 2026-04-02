// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Materialize Block Expressions Pass
//===----------------------------------------------------------------------===//
//
// Materializes ttl.store ops into ttl.compute ops.  For stores consuming
// block_expr DAGs, traces backward to find root inputs and emits tile ops.
// For passthrough stores (DFB-attached input), emits a compute with just
// a tile_store.
//
// Signpost scopes are handled via attributes (signpost_scopes) attached
// by TTLAttachSignpostScopes and reconstructed by TTLEmitSignpostScopes.
// This pass forwards the attribute from source block_expr/store ops to
// the emitted tile ops.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttl-materialize-block-exprs"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLMATERIALIZEBLOCKEXPRS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static bool isBlockExprOp(Operation *op) {
  return isa<AddBlockExprOp, SubBlockExprOp, MulBlockExprOp, DivBlockExprOp,
             MaxBlockExprOp, MinBlockExprOp, ExpBlockExprOp, LogBlockExprOp,
             SqrtBlockExprOp, RsqrtBlockExprOp, TanhBlockExprOp,
             SigmoidBlockExprOp, NegBlockExprOp, AbsBlockExprOp,
             ReluBlockExprOp, FloorBlockExprOp, RecipBlockExprOp,
             SinBlockExprOp, CosBlockExprOp, TanBlockExprOp, BlockExprMatmulOp,
             BlockExprBcastOp, BlockExprFillOp>(op);
}

static bool isBlockExprUnary(Operation *op) {
  return isa<ExpBlockExprOp, LogBlockExprOp, SqrtBlockExprOp, RsqrtBlockExprOp,
             TanhBlockExprOp, SigmoidBlockExprOp, NegBlockExprOp,
             AbsBlockExprOp, ReluBlockExprOp, FloorBlockExprOp,
             RecipBlockExprOp, SinBlockExprOp, CosBlockExprOp, TanBlockExprOp>(
      op);
}

static SmallVector<Value, 2> getBlockExprOperands(Operation *op) {
  if (isa<BlockExprFillOp>(op))
    return {};
  if (isBlockExprUnary(op))
    return {op->getOperand(0)};
  if (isa<BlockExprBcastOp>(op))
    return {op->getOperand(0)};
  return {op->getOperand(0), op->getOperand(1)};
}

struct BlockExprTrace {
  llvm::SmallSetVector<Value, 4> rootInputs;
  llvm::SmallSetVector<Operation *, 8> opsInOrder;
};

static BlockExprTrace traceBlockExprToRoots(Value value) {
  BlockExprTrace result;
  std::function<bool(Value)> trace = [&](Value v) -> bool {
    Operation *defOp = v.getDefiningOp();
    if (!defOp || !isBlockExprOp(defOp)) {
      result.rootInputs.insert(v);
      return true;
    }
    for (Value operand : getBlockExprOperands(defOp))
      if (!trace(operand))
        return false;
    result.opsInOrder.insert(defOp);
    return true;
  };
  trace(value);
  return result;
}

static BlockExprTrace
mergeTraces(const DenseMap<Operation *, BlockExprTrace> &storeTraces,
            SmallVectorImpl<StoreOp> &stores) {
  BlockExprTrace merged;
  DenseSet<Operation *> allOps;
  for (StoreOp store : stores) {
    auto it = storeTraces.find(store.getOperation());
    assert(it != storeTraces.end());
    const BlockExprTrace &trace = it->second;
    for (Value root : trace.rootInputs)
      merged.rootInputs.insert(root);
    for (Operation *op : trace.opsInOrder)
      allOps.insert(op);
  }
  Block *block = stores.front()->getBlock();
  for (auto &op : *block)
    if (allOps.contains(&op))
      merged.opsInOrder.insert(&op);
  return merged;
}

static Value emitTileOpForBlockExpr(OpBuilder &b, Location loc,
                                    Operation *blockExprOp,
                                    ValueRange tileOperands, Type tileType) {
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  if (isa<TTL_OP##BlockExprOp>(blockExprOp))                                   \
    return TILE_OP::create(b, loc, tileType, tileOperands[0]);
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)             \
  if (isa<TTL_OP##BlockExprOp>(blockExprOp))                                   \
    return TILE_OP::create(b, loc, tileType, tileOperands[0], tileOperands[1]);
#define TTL_BINARY_TILE_OP_MINMAX(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)      \
  TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"
  return nullptr;
}

static RankedTensorType getTensorType(Value v) {
  return dyn_cast<RankedTensorType>(v.getType());
}

/// Collect tile/dst-mode dprint ops between block_expr ops and stores.
/// These need to be relocated into the compute body.
static SmallVector<DPrintOp>
collectDPrintsForCompute(const BlockExprTrace &trace,
                         SmallVectorImpl<StoreOp> &stores) {
  if (trace.opsInOrder.empty())
    return {};

  DenseSet<Operation *> fusedSet(trace.opsInOrder.begin(),
                                 trace.opsInOrder.end());
  DenseSet<Operation *> storeSet;
  for (StoreOp store : stores)
    storeSet.insert(store.getOperation());

  // Find span: first fused op to last store.
  Operation *firstFused = nullptr;
  Operation *lastStore = nullptr;
  for (auto &op : *stores.front()->getBlock()) {
    if (fusedSet.contains(&op) && !firstFused)
      firstFused = &op;
    if (storeSet.contains(&op))
      lastStore = &op;
  }
  if (!firstFused || !lastStore)
    return {};

  SmallVector<DPrintOp> result;
  for (auto *op = firstFused; op != lastStore->getNextNode();
       op = op->getNextNode()) {
    if (auto dp = dyn_cast<DPrintOp>(op)) {
      StringRef mode = dp.getMode();
      if (mode == "dst" || mode == "tile")
        result.push_back(dp);
    }
  }
  return result;
}

static LogicalResult materializeStoreGroup(SmallVectorImpl<StoreOp> &stores,
                                           const BlockExprTrace &trace,
                                           OpBuilder &builder) {
  assert(!stores.empty());
  // Strip SignpostScopeAttr from the location used for infrastructure ops
  // (compute, init, tile_store, yield).  Only tile ops carry scope metadata
  // via opLoc.  This prevents downstream passes from inheriting scope info
  // on compiler-inserted ops (sync, inits).
  Location storeLoc = stores.back().getLoc();
  Location loc = storeLoc;
  if (auto fusedLoc = dyn_cast<FusedLoc>(storeLoc))
    if (isa_and_nonnull<SignpostScopeAttr>(fusedLoc.getMetadata()))
      loc = fusedLoc.getLocations().empty() ? storeLoc
                                            : fusedLoc.getLocations().front();

  auto outputType = getTensorType(stores.front().getView());
  if (!outputType)
    return stores.front().emitError("store view must have ranked tensor type");
  for (size_t i = 1; i < stores.size(); ++i) {
    StoreOp store = stores[i];
    if (getTensorType(store.getView()) != outputType)
      return store.emitError(
          "multi-store group: output type mismatch across stores");
  }

  if (trace.opsInOrder.empty())
    return success();

  // Validate bcast inputs.
  for (Operation *op : trace.opsInOrder) {
    if (auto bcastOp = dyn_cast<BlockExprBcastOp>(op)) {
      Value input = bcastOp.getInput();
      if (input.getDefiningOp() && isBlockExprOp(input.getDefiningOp()))
        return bcastOp.emitError(
            "broadcast input must come directly from a circular buffer "
            "(DFB-attached value), not from an intermediate expression");
    }
  }

  // Collect tile/dst dprints before materialization erases their context.
  auto dprintsForBody = collectDPrintsForCompute(trace, stores);

  MLIRContext *ctx = builder.getContext();

  SmallVector<Value> outCbs;
  for (StoreOp store : stores) {
    auto reserve = store.getView().getDefiningOp<CBReserveOp>();
    if (!reserve)
      return store.emitError("store view not from cb_reserve");
    outCbs.push_back(reserve.getCb());
  }

  // Build indexing maps.
  SmallVector<Attribute> maps;
  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(outputType.getRank(), ctx);
  for (Value root : trace.rootInputs) {
    auto inputType = getTensorType(root);
    if (inputType && inputType.getRank() == outputType.getRank()) {
      SmallVector<AffineExpr> exprs;
      bool hasBroadcast = false;
      for (int64_t d = 0; d < outputType.getRank(); ++d) {
        if (inputType.getDimSize(d) == 1 && outputType.getDimSize(d) != 1) {
          exprs.push_back(getAffineConstantExpr(0, ctx));
          hasBroadcast = true;
        } else {
          exprs.push_back(getAffineDimExpr(d, ctx));
        }
      }
      if (hasBroadcast)
        maps.push_back(AffineMapAttr::get(
            AffineMap::get(outputType.getRank(), 0, exprs, ctx)));
      else
        maps.push_back(AffineMapAttr::get(identityMap));
    } else {
      maps.push_back(AffineMapAttr::get(identityMap));
    }
  }
  for (size_t i = 0; i < outCbs.size(); ++i)
    maps.push_back(AffineMapAttr::get(identityMap));

  SmallVector<Attribute> iterTypes(outputType.getRank(),
                                   builder.getStringAttr("parallel"));

  builder.setInsertionPoint(stores.back());

  SmallVector<Value> allInitAttached;
  SmallVector<Type> resultTypes;
  for (Value outCb : outCbs) {
    Value init = tensor::EmptyOp::create(builder, loc, outputType.getShape(),
                                         outputType.getElementType());
    Value initAttached =
        AttachCBOp::create(builder, loc, init.getType(), init, outCb);
    allInitAttached.push_back(initAttached);
    resultTypes.push_back(outputType);
  }

  auto computeOp = ComputeOp::create(
      builder, loc, TypeRange(resultTypes), trace.rootInputs.getArrayRef(),
      ValueRange(allInitAttached), builder.getArrayAttr(maps),
      builder.getArrayAttr(iterTypes));

  Block *body = builder.createBlock(&computeOp.getBody());
  Type tileType = ttcore::TileType::get(outputType.getElementType());
  for (size_t i = 0; i < trace.rootInputs.size(); ++i)
    body->addArgument(tileType, loc);
  for (size_t i = 0; i < outCbs.size(); ++i)
    body->addArgument(tileType, loc);

  builder.setInsertionPointToStart(body);

  DenseMap<Value, Value> tensorToTile;
  for (size_t i = 0; i < trace.rootInputs.size(); ++i)
    tensorToTile[trace.rootInputs[i]] = body->getArgument(i);

  // Matmul+add deferred emission.
  DenseMap<Value, std::pair<Value, Value>> deferredMatmul;

  Value finalResult;
  for (Operation *op : trace.opsInOrder) {
    // Use the source block_expr op's location (carries FusedLoc scope metadata).
    Location opLoc = op->getLoc();
    Value tileResult;

    if (auto fillOp = dyn_cast<BlockExprFillOp>(op)) {
      tileResult =
          TileFillOp::create(builder, opLoc, tileType, fillOp.getValueAttr());
    } else if (auto bcastOp = dyn_cast<BlockExprBcastOp>(op)) {
      Value inputTile = tensorToTile[bcastOp.getInput()];
      Value outputTile = body->getArguments().back();
      tileResult = TileBcastOp::create(builder, opLoc, tileType, inputTile,
                                       outputTile, bcastOp.getBcastTypeAttr());
    } else if (auto matmulOp = dyn_cast<BlockExprMatmulOp>(op)) {
      Value lhsTile = tensorToTile[matmulOp.getLhs()];
      Value rhsTile = tensorToTile[matmulOp.getRhs()];
      bool deferred = false;
      if (matmulOp.getResult().hasOneUse()) {
        Operation *user = *matmulOp.getResult().getUsers().begin();
        if (isa<AddBlockExprOp>(user) && trace.opsInOrder.contains(user)) {
          deferredMatmul[matmulOp.getResult()] = {lhsTile, rhsTile};
          deferred = true;
        }
      }
      if (!deferred)
        tileResult = TileMatmulBlockOp::create(builder, opLoc, tileType,
                                               lhsTile, rhsTile, Value());
    } else {
      if (isa<AddBlockExprOp>(op)) {
        Value lhs = op->getOperand(0), rhs = op->getOperand(1);
        auto tryFold = [&](Value tensorA, Value tensorB) -> Value {
          auto dfIt = deferredMatmul.find(tensorA);
          if (dfIt == deferredMatmul.end())
            return nullptr;
          auto [mmLhs, mmRhs] = dfIt->second;
          Value accTile = tensorToTile.lookup(tensorB);
          if (!accTile)
            return nullptr;
          deferredMatmul.erase(dfIt);
          return TileMatmulBlockOp::create(builder, opLoc, tileType, mmLhs,
                                           mmRhs, accTile);
        };
        if (Value folded = tryFold(lhs, rhs))
          tileResult = folded;
        else if (Value folded = tryFold(rhs, lhs))
          tileResult = folded;
      }

      if (!tileResult) {
        for (Value operand : getBlockExprOperands(op)) {
          auto dfIt = deferredMatmul.find(operand);
          if (dfIt != deferredMatmul.end()) {
            auto [mmLhs, mmRhs] = dfIt->second;
            tensorToTile[operand] = TileMatmulBlockOp::create(
                builder, opLoc, tileType, mmLhs, mmRhs, Value());
            deferredMatmul.erase(dfIt);
          }
        }
      }

      if (!tileResult) {
        SmallVector<Value, 2> tileOperands;
        for (Value operand : getBlockExprOperands(op)) {
          auto it = tensorToTile.find(operand);
          if (it == tensorToTile.end())
            return op->emitError("block_expr materialization: operand not "
                                 "mapped to tile value");
          tileOperands.push_back(it->second);
        }
        tileResult =
            emitTileOpForBlockExpr(builder, opLoc, op, tileOperands, tileType);
        if (!tileResult)
          return op->emitError("block_expr materialization: unsupported op");
      }
    }

    if (tileResult) {
      tensorToTile[op->getResult(0)] = tileResult;
      finalResult = tileResult;
    }
  }

  // Emit tile_stores.
  SmallVector<Value> iterIndices = getOrCreateIterIndices(builder, computeOp);
  auto indexingMaps = computeOp.getIndexingMapsArray();
  size_t numInputs = trace.rootInputs.size();

  DenseMap<Value, size_t> cbToOutputIdx;
  if (outCbs.size() > 1)
    for (auto [idx, outCb] : llvm::enumerate(outCbs))
      cbToOutputIdx[outCb] = idx;

  for (int si = stores.size() - 1; si >= 0; --si) {
    StoreOp store = stores[si];
    Value tileVal = tensorToTile.lookup(store.getTensor());
    if (!tileVal)
      tileVal = finalResult;

    size_t outputIdx = 0;
    if (outCbs.size() > 1) {
      Value viewCB = getAttachedCB(store.getView());
      if (viewCB) {
        auto it = cbToOutputIdx.find(viewCB);
        if (it != cbToOutputIdx.end())
          outputIdx = it->second;
      }
    }
    AffineMap outputMap = indexingMaps[numInputs + outputIdx];
    SmallVector<Value> indices =
        applyIndexingMapToIterIndices(builder, loc, outputMap, iterIndices);

    // Use the store's scoped location so tile_store carries the scope.
    TileStoreOp::create(builder, store.getLoc(), tileVal, store.getView(),
                        indices);

  }

  // Emit collected dprints inside the compute body (after tile ops,
  // before yield). Their signpost_scopes attributes are preserved.
  for (DPrintOp dp : dprintsForBody)
    builder.clone(*dp.getOperation());

  YieldOp::create(builder, loc);

  // Erase stores, block_expr ops, and collected dprints.
  for (StoreOp store : stores)
    store.erase();
  for (auto it = trace.opsInOrder.rbegin(); it != trace.opsInOrder.rend();
       ++it)
    if ((*it)->use_empty())
      (*it)->erase();
  for (DPrintOp dp : dprintsForBody)
    dp.erase();

  return success();
}

/// Materialize a passthrough store into a compute with just a tile_store.
static LogicalResult materializePassthrough(StoreOp storeOp,
                                            OpBuilder &builder) {
  Value input = storeOp.getTensor();
  Value reserveView = storeOp.getView();
  auto reserve = reserveView.getDefiningOp<CBReserveOp>();
  if (!reserve)
    return storeOp.emitError("store view not from cb_reserve");
  Value outputCb = reserve.getCb();

  auto inputType = getTensorType(input);
  if (!inputType)
    return storeOp.emitError("passthrough store input must have tensor type");

  Location storeLoc = storeOp.getLoc();
  Location loc = storeLoc;
  if (auto fusedLoc = dyn_cast<FusedLoc>(storeLoc))
    if (isa_and_nonnull<SignpostScopeAttr>(fusedLoc.getMetadata()))
      loc = fusedLoc.getLocations().empty() ? storeLoc
                                            : fusedLoc.getLocations().front();
  MLIRContext *ctx = builder.getContext();

  AffineMap identityMap =
      AffineMap::getMultiDimIdentityMap(inputType.getRank(), ctx);
  SmallVector<Attribute> maps = {AffineMapAttr::get(identityMap),
                                 AffineMapAttr::get(identityMap)};
  SmallVector<Attribute> iterTypes(inputType.getRank(),
                                   builder.getStringAttr("parallel"));

  builder.setInsertionPoint(storeOp);
  Value init = tensor::EmptyOp::create(builder, loc, inputType.getShape(),
                                       inputType.getElementType());
  Value initAttached =
      AttachCBOp::create(builder, loc, init.getType(), init, outputCb);

  auto computeOp = ComputeOp::create(
      builder, loc, TypeRange{inputType}, ValueRange{input},
      ValueRange{initAttached}, builder.getArrayAttr(maps),
      builder.getArrayAttr(iterTypes));

  Block *body = builder.createBlock(&computeOp.getBody());
  Type tileType = ttcore::TileType::get(inputType.getElementType());
  body->addArgument(tileType, loc);
  body->addArgument(tileType, loc);

  builder.setInsertionPointToEnd(body);
  SmallVector<Value> iterIndices = getOrCreateIterIndices(builder, computeOp);
  SmallVector<Value> storeIndices =
      applyIndexingMapToIterIndices(builder, loc, identityMap, iterIndices);

  TileStoreOp::create(builder, loc, body->getArgument(0), reserveView,
                       storeIndices);

  YieldOp::create(builder, loc);

  for (OpOperand &use : llvm::make_early_inc_range(input.getUses()))
    if (auto attachOp = dyn_cast<AttachCBOp>(use.getOwner()))
      if (attachOp.getCb() == outputCb)
        attachOp.replaceAllUsesWith(computeOp.getResult(0));

  storeOp.erase();
  return success();
}

struct TTLMaterializeBlockExprsPass
    : public impl::TTLMaterializeBlockExprsBase<TTLMaterializeBlockExprsPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    SmallVector<StoreOp> blockExprStores;
    SmallVector<StoreOp> passthroughStores;
    DenseMap<Operation *, BlockExprTrace> storeTraces;

    funcOp.walk([&](StoreOp storeOp) {
      Value tensor = storeOp.getTensor();
      if (tensor.getDefiningOp() && isBlockExprOp(tensor.getDefiningOp())) {
        blockExprStores.push_back(storeOp);
        storeTraces[storeOp.getOperation()] = traceBlockExprToRoots(tensor);
      } else {
        passthroughStores.push_back(storeOp);
      }
    });

    OpBuilder builder(&getContext());

    if (!blockExprStores.empty()) {
      DenseMap<Value, SmallVector<StoreOp>> groupMap;
      for (StoreOp store : blockExprStores)
        groupMap[store.getTensor()].push_back(store);

      for (auto &[tensorVal, groupStores] : groupMap) {
        DenseMap<Operation *, unsigned> blockPos;
        unsigned pos = 0;
        for (auto &op : *groupStores.front()->getBlock())
          blockPos[&op] = pos++;
        llvm::sort(groupStores, [&](StoreOp lhs, StoreOp rhs) {
          return blockPos[lhs.getOperation()] < blockPos[rhs.getOperation()];
        });

        BlockExprTrace merged =
            (groupStores.size() == 1)
                ? storeTraces[groupStores.front().getOperation()]
                : mergeTraces(storeTraces, groupStores);

        if (failed(materializeStoreGroup(groupStores, merged, builder)))
          return signalPassFailure();
      }
    }

    for (StoreOp store : passthroughStores)
      if (failed(materializePassthrough(store, builder)))
        return signalPassFailure();
  }
};

} // namespace
} // namespace mlir::tt::ttl
