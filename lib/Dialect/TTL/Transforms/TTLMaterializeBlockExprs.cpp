// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Materialize Block Expressions Pass
//===----------------------------------------------------------------------===//
//
// Replaces convert-ttl-to-compute. For each ttl.store, traces the store's
// tensor operand backward through block_expr ops to find DFB-attached root
// inputs, then creates a ttl.compute at the store position with tile ops
// matching the expression DAG.
//
// Fusion is structural: the block expression DAG rooted at the store IS
// the fusion graph.
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

/// Check if an operation is a block_expr op (lazy expression node).
static bool isBlockExprOp(Operation *op) {
  return isa<AddBlockExprOp, SubBlockExprOp, MulBlockExprOp, DivBlockExprOp,
             MaxBlockExprOp, MinBlockExprOp, ExpBlockExprOp, LogBlockExprOp,
             SqrtBlockExprOp, RsqrtBlockExprOp, TanhBlockExprOp,
             SigmoidBlockExprOp, NegBlockExprOp, AbsBlockExprOp,
             ReluBlockExprOp, FloorBlockExprOp, RecipBlockExprOp,
             SinBlockExprOp, CosBlockExprOp, TanBlockExprOp, BlockExprMatmulOp,
             BlockExprBcastOp>(op);
}

/// Check if a block_expr op is unary (one operand).
static bool isBlockExprUnary(Operation *op) {
  return isa<ExpBlockExprOp, LogBlockExprOp, SqrtBlockExprOp, RsqrtBlockExprOp,
             TanhBlockExprOp, SigmoidBlockExprOp, NegBlockExprOp,
             AbsBlockExprOp, ReluBlockExprOp, FloorBlockExprOp,
             RecipBlockExprOp, SinBlockExprOp, CosBlockExprOp, TanBlockExprOp>(
      op);
}

/// Get the operands of a block_expr op that should be traced backward.
static SmallVector<Value, 2> getBlockExprOperands(Operation *op) {
  if (isBlockExprUnary(op)) {
    return {op->getOperand(0)};
  }
  if (isa<BlockExprBcastOp>(op)) {
    // Bcast: only trace the input, not the output (which provides shape).
    return {op->getOperand(0)};
  }
  // Binary ops (add, sub, mul, div, max, min, matmul): trace both operands.
  return {op->getOperand(0), op->getOperand(1)};
}

/// Trace backward from a value through block_expr ops to DFB-attached roots.
/// Collects all block_expr ops in topological order (roots first).
struct BlockExprTrace {
  llvm::SmallSetVector<Value, 4> rootInputs;
  llvm::SmallSetVector<Operation *, 8> opsInOrder;
};

static BlockExprTrace traceBlockExprToRoots(Value value) {
  BlockExprTrace result;

  std::function<bool(Value)> trace = [&](Value v) -> bool {
    Operation *defOp = v.getDefiningOp();
    if (!defOp || !isBlockExprOp(defOp)) {
      // Not a block_expr op: this is a root input (DFB-attached value).
      result.rootInputs.insert(v);
      return true;
    }

    // Trace operands recursively.
    for (Value operand : getBlockExprOperands(defOp)) {
      if (!trace(operand)) {
        return false;
      }
    }

    // Add this op after its dependencies (topological order).
    result.opsInOrder.insert(defOp);
    return true;
  };

  trace(value);
  return result;
}

/// Emit the tile-level op corresponding to a block_expr op.
/// Returns the result Value, or null on failure.
static Value emitTileOpForBlockExpr(OpBuilder &b, Location loc,
                                    Operation *blockExprOp,
                                    ValueRange tileOperands, Type tileType) {
  // Map block_expr ops to their corresponding tile ops. The generated
  // class names are <Name>BlockExprOp (e.g., AddBlockExprOp).
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

/// Materialize a single store: trace its expression, build ttl.compute.
static LogicalResult materializeStore(StoreOp storeOp, OpBuilder &builder) {
  Value expr = storeOp.getTensor();
  Value view = storeOp.getView();
  Location loc = storeOp.getLoc();

  auto outputType = getTensorType(view);
  if (!outputType) {
    return storeOp.emitError("store view must have ranked tensor type");
  }

  // Trace the expression backward through block_expr ops.
  BlockExprTrace trace = traceBlockExprToRoots(expr);

  // If the expression is not a block_expr (passthrough store), leave it
  // for ConvertTTLToCompute or handle directly.
  if (trace.opsInOrder.empty()) {
    // Passthrough: the store input is directly DFB-attached.
    // This is handled by the existing ConvertTTLToCompute passthrough
    // logic. Leave the store as-is for now.
    return success();
  }

  MLIRContext *ctx = builder.getContext();

  // Find the output CB from the store's view.
  auto reserve = view.getDefiningOp<CBReserveOp>();
  if (!reserve) {
    return storeOp.emitError("store view not from cb_reserve");
  }
  Value outCb = reserve.getCb();

  // Build indexing maps: broadcast-aware for inputs, identity for output.
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
      if (hasBroadcast) {
        maps.push_back(AffineMapAttr::get(
            AffineMap::get(outputType.getRank(), 0, exprs, ctx)));
      } else {
        maps.push_back(AffineMapAttr::get(identityMap));
      }
    } else {
      maps.push_back(AffineMapAttr::get(identityMap));
    }
  }
  // Output map.
  maps.push_back(AffineMapAttr::get(identityMap));

  // Iterator types: all parallel.
  SmallVector<Attribute> iterTypes(outputType.getRank(),
                                   builder.getStringAttr("parallel"));

  // Position compute before the store.
  builder.setInsertionPoint(storeOp);

  // Create init tensor and attach to output CB.
  Value init = tensor::EmptyOp::create(builder, loc, outputType.getShape(),
                                       outputType.getElementType());
  Value initAttached =
      AttachCBOp::create(builder, loc, init.getType(), init, outCb);

  // Create ttl.compute.
  auto computeOp = ComputeOp::create(
      builder, loc, TypeRange{outputType}, trace.rootInputs.getArrayRef(),
      ValueRange{initAttached}, builder.getArrayAttr(maps),
      builder.getArrayAttr(iterTypes));

  // Build the body region.
  Block *body = builder.createBlock(&computeOp.getBody());
  Type tileType = ttcore::TileType::get(outputType.getElementType());

  for (size_t i = 0; i < trace.rootInputs.size(); ++i) {
    body->addArgument(tileType, loc);
  }
  body->addArgument(tileType, loc); // output block arg

  builder.setInsertionPointToStart(body);

  // Map tensor values to tile values.
  DenseMap<Value, Value> tensorToTile;
  for (size_t i = 0; i < trace.rootInputs.size(); ++i) {
    tensorToTile[trace.rootInputs[i]] = body->getArgument(i);
  }

  // Matmul+add deferred emission map.
  DenseMap<Value, std::pair<Value, Value>> deferredMatmul;

  // Emit tile ops in topological order.
  Value finalResult;
  for (Operation *op : trace.opsInOrder) {
    Value tileResult;

    if (auto bcastOp = dyn_cast<BlockExprBcastOp>(op)) {
      Value inputTile = tensorToTile[bcastOp.getInput()];
      Value outputTile = body->getArguments().back();
      tileResult = TileBcastOp::create(builder, loc, tileType, inputTile,
                                       outputTile, bcastOp.getBcastTypeAttr());
    } else if (auto matmulOp = dyn_cast<BlockExprMatmulOp>(op)) {
      Value lhsTile = tensorToTile[matmulOp.getLhs()];
      Value rhsTile = tensorToTile[matmulOp.getRhs()];

      // Defer if sole user is a block_expr.add in this trace.
      bool deferred = false;
      if (matmulOp.getResult().hasOneUse()) {
        Operation *user = *matmulOp.getResult().getUsers().begin();
        if (isa<AddBlockExprOp>(user) && trace.opsInOrder.contains(user)) {
          deferredMatmul[matmulOp.getResult()] = {lhsTile, rhsTile};
          deferred = true;
        }
      }
      if (!deferred) {
        tileResult = TileMatmulBlockOp::create(builder, loc, tileType, lhsTile,
                                               rhsTile, Value());
      }
    } else {
      // Check for matmul+add fold.
      if (isa<AddBlockExprOp>(op)) {
        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);
        auto tryFold = [&](Value tensorA, Value tensorB) -> Value {
          auto dfIt = deferredMatmul.find(tensorA);
          if (dfIt == deferredMatmul.end()) {
            return nullptr;
          }
          auto [mmLhs, mmRhs] = dfIt->second;
          Value accTile = tensorToTile.lookup(tensorB);
          if (!accTile) {
            return nullptr;
          }
          deferredMatmul.erase(dfIt);
          return TileMatmulBlockOp::create(builder, loc, tileType, mmLhs, mmRhs,
                                           accTile);
        };
        Value folded = tryFold(lhs, rhs);
        if (!folded) {
          folded = tryFold(rhs, lhs);
        }
        if (folded) {
          tileResult = folded;
        }
      }

      // Emit deferred matmuls if the fold didn't apply.
      if (!tileResult) {
        for (Value operand : getBlockExprOperands(op)) {
          auto dfIt = deferredMatmul.find(operand);
          if (dfIt != deferredMatmul.end()) {
            auto [mmLhs, mmRhs] = dfIt->second;
            Value mmTile = TileMatmulBlockOp::create(builder, loc, tileType,
                                                     mmLhs, mmRhs, Value());
            tensorToTile[operand] = mmTile;
            deferredMatmul.erase(dfIt);
          }
        }
      }

      // Standard elementwise emission.
      if (!tileResult) {
        SmallVector<Value, 2> tileOperands;
        for (Value operand : getBlockExprOperands(op)) {
          auto it = tensorToTile.find(operand);
          if (it == tensorToTile.end()) {
            return op->emitError("block_expr materialization: operand not "
                                 "mapped to tile value");
          }
          tileOperands.push_back(it->second);
        }

        tileResult =
            emitTileOpForBlockExpr(builder, loc, op, tileOperands, tileType);
        if (!tileResult) {
          return op->emitError("block_expr materialization: unsupported op");
        }
      }
    }

    if (tileResult) {
      tensorToTile[op->getResult(0)] = tileResult;
      finalResult = tileResult;
    }
  }

  // Emit tile_store and yield.
  SmallVector<Value> iterIndices = getOrCreateIterIndices(builder, computeOp);
  auto indexingMaps = computeOp.getIndexingMapsArray();
  AffineMap outputMap = indexingMaps.back();
  SmallVector<Value> indices =
      applyIndexingMapToIterIndices(builder, loc, outputMap, iterIndices);

  TileStoreOp::create(builder, loc, finalResult, view, indices);
  YieldOp::create(builder, loc);

  // Replace the store's result usage and erase it.
  storeOp.erase();

  // Erase block_expr ops in reverse topological order.
  for (auto it = trace.opsInOrder.rbegin(); it != trace.opsInOrder.rend();
       ++it) {
    Operation *op = *it;
    if (op->use_empty()) {
      op->erase();
    }
  }

  return success();
}

struct TTLMaterializeBlockExprsPass
    : public impl::TTLMaterializeBlockExprsBase<TTLMaterializeBlockExprsPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Collect all stores first (we'll modify the function body).
    SmallVector<StoreOp> stores;
    funcOp.walk([&](StoreOp storeOp) {
      // Only process stores whose input is a block_expr op.
      Value tensor = storeOp.getTensor();
      if (tensor.getDefiningOp() && isBlockExprOp(tensor.getDefiningOp())) {
        stores.push_back(storeOp);
      }
    });

    OpBuilder builder(&getContext());
    for (StoreOp storeOp : stores) {
      if (failed(materializeStore(storeOp, builder))) {
        return signalPassFailure();
      }
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
