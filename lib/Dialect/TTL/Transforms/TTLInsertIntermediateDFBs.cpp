// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTL Insert Intermediate DFBs
//===----------------------------------------------------------------------===//
//
// Inserts compiler-allocated intermediate dataflow buffers at fusion split
// points. Tensor-level ops whose tile-level lowerings require DFB inputs
// may receive operands from fused expression chains that are not
// DFB-attached. This pass materializes those intermediates to L1 via DFBs
// so that convert-ttl-to-compute sees all required operands as CB-attached.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "ttl-insert-intermediate-dfbs"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLINSERTINTERMEDIATEDFBS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Materialize a value to a compiler-allocated DFB. Inserts bind_cb,
/// cb_reserve, store, cb_wait, attach_cb. Returns the CB-attached result,
/// or failure if the maximum CB count would be exceeded.
FailureOr<Value> materializeToDFB(Value intermediate, ModuleOp moduleOp,
                                  OpBuilder &builder) {
  auto tensorType = mlir::cast<RankedTensorType>(intermediate.getType());
  Location loc = intermediate.getLoc();
  MLIRContext *ctx = builder.getContext();

  // Intra-thread push/wait requires double-buffering so the packer and
  // unpacker can operate on different buffer halves simultaneously.
  SmallVector<int64_t> shape(tensorType.getShape());
  Type elementType = tensorType.getElementType();
  int64_t blockCount = 2;
  auto cbType = CircularBufferType::get(ctx, shape, elementType, blockCount);

  int32_t dfbIndex = getNextAvailableDFBIndex(moduleOp);

  Operation *defOp = intermediate.getDefiningOp();
  assert(defOp && "intermediate must have a defining op");

  // Hoist BindCBOp to the function body entry: its cb_index is function-
  // scoped and TTLFinalizeDFBIndices requires every compiler-allocated
  // BindCBOp to live there. Only BindCBOp hoists; reserve/store/wait/attach
  // stay at the def site to preserve per-invocation accounting inside
  // loops and conditional branches.
  auto funcOp = defOp->getParentOfType<func::FuncOp>();
  assert(funcOp && "intermediate must be inside a func::FuncOp");
  Block &body = funcOp.getBody().front();

  // Place after the last leading BindCBOp so ordering is deterministic.
  Operation *insertAfter = nullptr;
  for (Operation &op : body) {
    if (isa<BindCBOp>(&op)) {
      insertAfter = &op;
    } else if (insertAfter) {
      break;
    }
  }
  if (insertAfter) {
    builder.setInsertionPointAfter(insertAfter);
  } else {
    builder.setInsertionPointToStart(&body);
  }

  auto indexAttr = builder.getIndexAttr(dfbIndex);
  auto blockCountAttr = builder.getI64IntegerAttr(blockCount);
  auto bindCB =
      BindCBOp::create(builder, loc, cbType, indexAttr, blockCountAttr);
  bindCB->setAttr(kCompilerAllocatedAttrName, builder.getUnitAttr());

  // Remaining ops bind to the intermediate's def site.
  builder.setInsertionPointAfter(defOp);

  auto reserve =
      CBReserveOp::create(builder, loc, tensorType, bindCB.getResult());

  StoreOp::create(builder, loc, intermediate, reserve.getResult(),
                  /*accumulate=*/nullptr);

  // cb_push is inserted by ttl-insert-cb-sync which runs after this pass.

  auto wait = CBWaitOp::create(builder, loc, tensorType, bindCB.getResult());

  auto attachWait = AttachCBOp::create(builder, loc, tensorType,
                                       wait.getResult(), bindCB.getResult());

  return attachWait.getResult();
}

/// Returns true if `dimsAttr` reduces every axis of a rank-`inputRank`
/// input (REDUCE_SCALAR mode in the LLK). Negative dims are normalized.
static bool reducesAllDims(ArrayRef<int64_t> dimsAttr, int64_t inputRank) {
  llvm::SmallSet<int64_t, 2> normDims;
  for (int64_t dim : dimsAttr) {
    normDims.insert((dim % inputRank + inputRank) % inputRank);
  }
  return static_cast<int64_t>(normDims.size()) == inputRank;
}

/// Always-feed-fill(1.0) rewrite for `ttl.reduce`. The LLK's REDUCE_SCALAR
/// (dims=[0,1]) double-applies the scaler tile internally — documented in
/// tt-metal's reduce_op.cpp note about sqrt-compensation — so any non-1.0
/// scaler tile (compile-time fill or runtime value) produces wrong math.
/// `1.0 × 1.0 = 1.0` makes the double-apply harmless when the reduce's
/// scaler operand is always `fill(1.0)`; the actual scaler is applied
/// once via a separate post-reduce multiply where the math is unambiguous.
///
/// Two post-multiply paths share the same structure:
///   - Statically-known fill(c) (the numeric-scalar API): wrap the reduce
///     result in `ttl.mul_unary_const(result, c)`. Fast path: lowers to
///     LLK `mul_unary_tile` with an i32 bit pattern of `c`. No extra DFB.
///   - Any other scaler tile (user tile-form, runtime-computed from a
///     prior op, etc.) AND `dims=[0,1]`: wrap the reduce result in
///     `ttl.mul(result, original_scaler)`. Element-wise tile multiply;
///     correct at the only read position [0, 0] of a REDUCE_SCALAR output
///     because `result[0, 0] = intermediate[0, 0] * scaler[0, 0]`.
///
/// Non-fill scalers on single-dim reduces (REDUCE_COL / REDUCE_ROW) are
/// left alone — the LLK handles those correctly already (no double-apply).
///
/// One neutral fill is cached per source FillOp so multiple reduces
/// sharing the same compile-time scaler converge on the same SSA value
/// and the standard DFB-materialization cache deduplicates the DFB.
LogicalResult rewriteReduceScalersToPostMul(func::FuncOp funcOp,
                                            OpBuilder &builder) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();

  SmallVector<ReduceOp> reducesToRewrite;
  funcOp.walk([&](ReduceOp reduceOp) {
    Value scaler = reduceOp.getScaler();
    if (auto fillOp = scaler.getDefiningOp<FillOp>()) {
      float c = fillOp.getValueAttr().getValue().convertToFloat();
      if (c != 1.0f) {
        reducesToRewrite.push_back(reduceOp);
      }
      return;
    }
    auto inputType = cast<RankedTensorType>(reduceOp.getInput().getType());
    if (reducesAllDims(reduceOp.getDims(), inputType.getRank())) {
      reducesToRewrite.push_back(reduceOp);
    }
  });

  llvm::DenseMap<FillOp, Value> neutralFillFor;
  for (ReduceOp reduceOp : reducesToRewrite) {
    Location loc = reduceOp.getLoc();
    Value scaler = reduceOp.getScaler();
    auto fillOp = scaler.getDefiningOp<FillOp>();

    if (fillOp) {
      // Compile-time fill(c). MulUnaryConstOp has DFBInputOpInterface, so
      // the standard inserter loop will materialize the reduce result into
      // a compiler-allocated DFB later in this pass.
      float c = fillOp.getValueAttr().getValue().convertToFloat();
      auto [iter, inserted] = neutralFillFor.try_emplace(fillOp, Value{});
      if (inserted) {
        builder.setInsertionPointAfter(fillOp);
        auto neutral = FillOp::create(builder, fillOp.getLoc(),
                                      fillOp.getType(),
                                      builder.getF32FloatAttr(1.0));
        iter->second = neutral.getResult();
      }
      reduceOp.getScalerMutable().assign(iter->second);
      builder.setInsertionPointAfter(reduceOp);
      auto mulOp = MulUnaryConstOp::create(
          builder, loc, reduceOp.getResult().getType(),
          reduceOp.getResult(), builder.getF32FloatAttr(c));
      reduceOp.getResult().replaceAllUsesExcept(mulOp.getResult(), mulOp);
    } else {
      // Runtime tile scaler. MulOp does not declare DFBInputOpInterface
      // (it's the generic elementwise multiply used throughout the DSL),
      // so we materialize the reduce result through a compiler-allocated
      // DFB explicitly here. Then MulOp sees a CB-attached lhs.
      //
      // Snapshot the reduce-result's existing uses before adding new
      // ones; materializeToDFB creates a fresh use (the store of the
      // reduce result into the intermediate DFB) that we must not
      // redirect to the mul.
      SmallVector<OpOperand *> existingUses;
      for (OpOperand &use : reduceOp.getResult().getUses()) {
        existingUses.push_back(&use);
      }
      builder.setInsertionPoint(reduceOp);
      auto neutral = FillOp::create(builder, loc,
                                    cast<RankedTensorType>(scaler.getType()),
                                    builder.getF32FloatAttr(1.0));
      reduceOp.getScalerMutable().assign(neutral.getResult());
      builder.setInsertionPointAfter(reduceOp);
      auto materialized =
          materializeToDFB(reduceOp.getResult(), moduleOp, builder);
      if (failed(materialized)) {
        return failure();
      }
      auto mulOp = MulOp::create(builder, loc, reduceOp.getResult().getType(),
                                 *materialized, scaler);
      for (OpOperand *use : existingUses) {
        use->set(mulOp.getResult());
      }
    }
  }
  return success();
}

struct TTLInsertIntermediateDFBsPass
    : public impl::TTLInsertIntermediateDFBsBase<
          TTLInsertIntermediateDFBsPass> {
  using TTLInsertIntermediateDFBsBase::TTLInsertIntermediateDFBsBase;

  void runOnOperation() override {
    auto funcOp = getOperation();
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return;
    }

    SmallVector<DFBInputOpInterface> candidates;
    funcOp.walk([&](DFBInputOpInterface op) { candidates.push_back(op); });

    // When compiler DFBs are disabled, verify that no operations require
    // them and emit an actionable error if any do.
    if (!enable) {
      for (DFBInputOpInterface dfbInputOp : candidates) {
        Operation *op = dfbInputOp.getOperation();
        auto requiredIndices = dfbInputOp.getDFBInputOperandIndices();

        for (unsigned idx : requiredIndices) {
          Value operand = op->getOperand(idx);
          if (getAttachedCB(operand)) {
            continue;
          }

          // Targeted diagnostic for the scalar-constant reduce path. The DSL
          // for `ttl.math.reduce_{sum,max}(x, <number>, ...)` synthesizes a
          // FillOp scaler that depends on compiler-allocated DFBs.
          if (auto reduceOp = dyn_cast<ReduceOp>(op);
              reduceOp && idx == 1 && operand.getDefiningOp<FillOp>()) {
            reduceOp->emitOpError(
                "numeric scalar reduce scaler requires compiler-allocated "
                "DFBs; pass a 1x1 user-DFB-attached scaler tile instead, or "
                "enable compiler DFBs (drop --no-ttl-compiler-dfbs)");
            signalPassFailure();
            return;
          }

          op->emitOpError("operand #")
              << idx
              << " requires a DFB-attached value but compiler-allocated DFBs "
                 "are disabled (--no-ttl-compiler-dfbs); either enable "
                 "compiler DFBs or store the intermediate to a user-declared "
                 "DFB before this operation";
          signalPassFailure();
          return;
        }
      }
      return;
    }

    OpBuilder builder(funcOp.getContext());

    // Structural rewrite: every `ttl.reduce(x, scaler)` whose scaler is
    // not `fill(1.0)` becomes the chain
    //   intermediate = ttl.reduce(x, fill(1.0))
    //   result       = post_mul(intermediate, scaler)
    // where post_mul is `ttl.mul_unary_const` for compile-time fill(c)
    // scalers and `ttl.mul` for any runtime tile scaler. Re-collect
    // candidates afterward so the new ops go through normal DFB
    // materialization below.
    if (failed(rewriteReduceScalersToPostMul(funcOp, builder))) {
      signalPassFailure();
      return;
    }
    candidates.clear();
    funcOp.walk([&](DFBInputOpInterface op) { candidates.push_back(op); });

    llvm::DenseMap<Value, Value> materialized;

    for (DFBInputOpInterface dfbInputOp : candidates) {
      Operation *op = dfbInputOp.getOperation();
      auto requiredIndices = dfbInputOp.getDFBInputOperandIndices();

      for (unsigned idx : requiredIndices) {
        Value operand = op->getOperand(idx);

        if (getAttachedCB(operand)) {
          continue;
        }

        if (auto iter = materialized.find(operand); iter != materialized.end()) {
          op->setOperand(idx, iter->second);
          continue;
        }

        auto replacement = materializeToDFB(operand, moduleOp, builder);
        if (failed(replacement)) {
          signalPassFailure();
          return;
        }

        // Replace only this specific operand. Elementwise consumers of
        // the same value retain the original SSA value and fuse with
        // the producer in a single compute block.
        op->setOperand(idx, *replacement);

        materialized[operand] = *replacement;
      }
    }

    // Sweep FillOps left dead by scaler rewrites. Non-reduce consumers of
    // a rewritten fill (e.g., a user-declared DFB store) keep it alive.
    funcOp.walk([](FillOp fillOp) {
      if (fillOp->use_empty()) {
        fillOp.erase();
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
