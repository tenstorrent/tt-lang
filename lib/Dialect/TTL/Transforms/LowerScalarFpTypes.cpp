// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h" // IWYU pragma: keep

#include "ttlang/Dialect/Utils/ConversionUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

namespace mlir::tt::ttl {
#define GEN_PASS_DEF_TTLLOWERSCALARFPTYPES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

namespace ttk = mlir::tt::ttkernel;

/// Try to unwrap an unrealized_conversion_cast(iN -> fN). Returns the
/// integer source if the cast matches, or nullptr.
static Value unwrapIntToFloatCast(Value floatVal) {
  auto cast = floatVal.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast || cast.getInputs().size() != 1)
    return nullptr;
  Value src = cast.getInputs()[0];
  if (!src.getType().isSignlessInteger())
    return nullptr;
  if (src.getType().getIntOrFloatBitWidth() !=
      floatVal.getType().getIntOrFloatBitWidth())
    return nullptr;
  return src;
}

/// Propagate integer types through scf.if results.
///
/// When convert-ttl-to-ttkernel lowers raw_element_read, it produces:
///   %i = load_from_l1 -> i32
///   %f = unrealized_conversion_cast %i : i32 to f32
///
/// If these float values flow through scf.if, the cast chain breaks:
///   scf.if -> (f32) {
///     scf.yield %f1 : f32    // f1 = cast(i32 -> f32)
///   } else {
///     scf.yield %f2 : f32    // f2 = cast(i32 -> f32)
///   }
///   %result_i = cast %result : f32 to i32   // from materializeIntBits
///
/// This function rewrites the scf.if to yield i32 directly, eliminating
/// the intermediate float casts.
static bool propagateIntTypeThroughScfIf(ModuleOp mod) {
  // Collect candidates first to avoid modifying IR during walk.
  SmallVector<scf::IfOp> candidates;
  mod.walk([&](scf::IfOp ifOp) {
    if (ifOp.getNumResults() == 0)
      return;
    if (!ifOp.elseYield())
      return;
    candidates.push_back(ifOp);
  });

  bool changed = false;
  for (auto ifOp : candidates) {
    auto thenYield = ifOp.thenYield();
    auto elseYield = ifOp.elseYield();

    SmallVector<unsigned> convertIndices;
    for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
      Value result = ifOp.getResult(i);
      if (!mlir::isa<FloatType>(result.getType()))
        continue;

      Value thenInt = unwrapIntToFloatCast(thenYield.getOperand(i));
      Value elseInt = unwrapIntToFloatCast(elseYield.getOperand(i));
      if (!thenInt || !elseInt)
        continue;

      // Require all uses to be unrealized_conversion_cast(fN -> iN).
      bool allUsesAreCastToInt = true;
      for (OpOperand &use : result.getUses()) {
        auto userCast =
            dyn_cast<UnrealizedConversionCastOp>(use.getOwner());
        if (!userCast || userCast.getResults().size() != 1 ||
            !userCast.getResults()[0].getType().isSignlessInteger()) {
          allUsesAreCastToInt = false;
          break;
        }
      }
      if (!allUsesAreCastToInt)
        continue;

      convertIndices.push_back(i);
    }

    if (convertIndices.empty())
      continue;

    // Build new result types: float -> int for converted indices.
    SmallVector<Type> newResultTypes;
    for (unsigned i = 0; i < ifOp.getNumResults(); ++i)
      newResultTypes.push_back(ifOp.getResult(i).getType());

    for (unsigned idx : convertIndices) {
      Value thenInt = unwrapIntToFloatCast(thenYield.getOperand(idx));
      newResultTypes[idx] = thenInt.getType();
    }

    // Create a new scf.if with updated result types and move bodies.
    OpBuilder builder(ifOp);
    auto newIf = scf::IfOp::create(builder, ifOp.getLoc(), newResultTypes,
                                   ifOp.getCondition(),
                                   /*withElseRegion=*/true);
    newIf.getThenRegion().takeBody(ifOp.getThenRegion());
    newIf.getElseRegion().takeBody(ifOp.getElseRegion());

    // Patch yield operands to yield integer values directly.
    auto newThenYield = newIf.thenYield();
    auto newElseYield = newIf.elseYield();
    for (unsigned idx : convertIndices) {
      Value thenInt = unwrapIntToFloatCast(newThenYield.getOperand(idx));
      Value elseInt = unwrapIntToFloatCast(newElseYield.getOperand(idx));
      newThenYield.setOperand(idx, thenInt);
      newElseYield.setOperand(idx, elseInt);
    }

    // Replace uses of old results with new results.
    for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
      Value oldResult = ifOp.getResult(i);
      Value newResult = newIf.getResult(i);

      if (llvm::is_contained(convertIndices, i)) {
        // Result changed fN -> iN. Bypass the consumer cast ops.
        SmallVector<UnrealizedConversionCastOp> castsToErase;
        for (OpOperand &use : oldResult.getUses()) {
          auto userCast =
              dyn_cast<UnrealizedConversionCastOp>(use.getOwner());
          if (userCast && userCast.getResults().size() == 1 &&
              userCast.getResults()[0].getType() == newResult.getType()) {
            userCast.getResults()[0].replaceAllUsesWith(newResult);
            castsToErase.push_back(userCast);
          }
        }
        for (auto cast : castsToErase)
          cast.erase();

        // Safety: if any uses remain, insert a back-cast.
        if (!oldResult.use_empty()) {
          OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointAfter(newIf);
          auto backCast = UnrealizedConversionCastOp::create(
              builder, ifOp.getLoc(), oldResult.getType(), newResult);
          oldResult.replaceAllUsesWith(backCast.getResult(0));
        }
      } else {
        oldResult.replaceAllUsesWith(newResult);
      }
    }

    ifOp.erase();
    changed = true;
  }

  return changed;
}

/// Try to resolve a float value to its underlying integer source,
/// looking through casts and the iter_arg of a scf.for.
/// Returns the integer value if found, nullptr otherwise.
static Value resolveToInt(Value floatVal, BlockArgument iterArg) {
  // Direct cast: unrealized_conversion_cast(iN -> fN).
  if (Value intSrc = unwrapIntToFloatCast(floatVal))
    return intSrc;
  // The iter_arg itself -- will become integer after the type change.
  if (floatVal == iterArg)
    return iterArg;
  return nullptr;
}

/// Propagate integer types through scf.for iter_args/results.
///
/// Handles the common pattern where the loop body contains an scf.if
/// that conditionally reassigns the iter_arg:
///   scf.for iter_args(%arg = cast(%init_i)) -> f32 {
///     %new = cast(%loaded_i : i32 -> f32)
///     %pick = scf.if %cond -> f32 {
///       scf.yield %new : f32        // cast(iN -> fN)
///     } else {
///       scf.yield %arg : f32        // iter_arg passthrough
///     }
///     scf.yield %pick : f32
///   }
///   %result_i = cast(%result : f32 -> i32)
///
/// Rewritten so the for, its inner scf.if, and yields all carry i32.
static bool propagateIntTypeThroughScfFor(ModuleOp mod) {
  SmallVector<scf::ForOp> candidates;
  mod.walk([&](scf::ForOp forOp) {
    if (forOp.getNumResults() == 0)
      return;
    candidates.push_back(forOp);
  });

  bool changed = false;
  for (auto forOp : candidates) {
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    unsigned numIterArgs = forOp.getNumResults();

    SmallVector<unsigned> convertIndices;
    for (unsigned i = 0; i < numIterArgs; ++i) {
      Value result = forOp.getResult(i);
      if (!mlir::isa<FloatType>(result.getType()))
        continue;

      // Init must be unrealized_conversion_cast(iN -> fN).
      Value initInt = unwrapIntToFloatCast(forOp.getInitArgs()[i]);
      if (!initInt)
        continue;

      BlockArgument iterArg = forOp.getRegionIterArg(i);
      Value forYieldVal = yieldOp.getOperand(i);

      // The for yield must resolve to integer: either a direct cast,
      // the iter_arg itself, or an scf.if whose branches all resolve.
      bool yieldResolvable = false;
      if (resolveToInt(forYieldVal, iterArg)) {
        yieldResolvable = true;
      } else if (auto innerIf =
                     forYieldVal.getDefiningOp<scf::IfOp>()) {
        // Check which result index of the scf.if this is.
        unsigned ifResultIdx = 0;
        for (unsigned r = 0; r < innerIf.getNumResults(); ++r) {
          if (innerIf.getResult(r) == forYieldVal) {
            ifResultIdx = r;
            break;
          }
        }
        auto thenYield = innerIf.thenYield();
        auto elseYield = innerIf.elseYield();
        if (elseYield &&
            resolveToInt(thenYield.getOperand(ifResultIdx), iterArg) &&
            resolveToInt(elseYield.getOperand(ifResultIdx), iterArg)) {
          yieldResolvable = true;
        }
      }
      if (!yieldResolvable)
        continue;

      // All external uses must be unrealized_conversion_cast(fN -> iN).
      bool allUsesAreCastToInt = true;
      for (OpOperand &use : result.getUses()) {
        auto userCast =
            dyn_cast<UnrealizedConversionCastOp>(use.getOwner());
        if (!userCast || userCast.getResults().size() != 1 ||
            !userCast.getResults()[0].getType().isSignlessInteger()) {
          allUsesAreCastToInt = false;
          break;
        }
      }
      if (!allUsesAreCastToInt)
        continue;

      // All uses of the iter_arg inside the body must be through casts,
      // the yield, or scf.if yields (which we rewrite together).
      bool iterArgOk = true;
      for (OpOperand &use : iterArg.getUses()) {
        Operation *owner = use.getOwner();
        if (owner == yieldOp)
          continue;
        if (isa<UnrealizedConversionCastOp>(owner))
          continue;
        // Allow use as scf.if yield operand (iter_arg passthrough).
        if (isa<scf::YieldOp>(owner) &&
            owner->getParentOp() != forOp.getOperation())
          continue;
        iterArgOk = false;
        break;
      }
      if (!iterArgOk)
        continue;

      convertIndices.push_back(i);
    }

    if (convertIndices.empty())
      continue;

    for (unsigned idx : convertIndices) {
      Value initInt = unwrapIntToFloatCast(forOp.getInitArgs()[idx]);
      BlockArgument iterArg = forOp.getRegionIterArg(idx);
      Type intTy = initInt.getType();

      // Update init operand.
      forOp.getInitArgsMutable()[idx].set(initInt);

      // Change the iter_arg block argument type.
      iterArg.setType(intTy);

      // Update the for yield. If it's an scf.if result, rewrite the
      // scf.if to yield integer too.
      Value forYieldVal = yieldOp.getOperand(idx);
      if (Value intVal = unwrapIntToFloatCast(forYieldVal)) {
        yieldOp.setOperand(idx, intVal);
      } else if (auto innerIf =
                     forYieldVal.getDefiningOp<scf::IfOp>()) {
        unsigned ifResultIdx = 0;
        for (unsigned r = 0; r < innerIf.getNumResults(); ++r) {
          if (innerIf.getResult(r) == forYieldVal) {
            ifResultIdx = r;
            break;
          }
        }

        // Build new scf.if with integer result type at this index.
        SmallVector<Type> newIfTypes;
        for (unsigned r = 0; r < innerIf.getNumResults(); ++r)
          newIfTypes.push_back(innerIf.getResult(r).getType());
        newIfTypes[ifResultIdx] = intTy;

        OpBuilder builder(innerIf);
        auto newIf = scf::IfOp::create(builder, innerIf.getLoc(),
                                       newIfTypes, innerIf.getCondition(),
                                       /*withElseRegion=*/true);
        newIf.getThenRegion().takeBody(innerIf.getThenRegion());
        newIf.getElseRegion().takeBody(innerIf.getElseRegion());

        // Patch yields in both branches.
        auto patchYield = [&](scf::YieldOp branchYield) {
          Value branchVal = branchYield.getOperand(ifResultIdx);
          if (Value intSrc = unwrapIntToFloatCast(branchVal)) {
            branchYield.setOperand(ifResultIdx, intSrc);
          }
          // If the operand is the iter_arg, its type already changed to
          // iN, so no update needed.
        };
        patchYield(newIf.thenYield());
        patchYield(newIf.elseYield());

        // Replace all results of old scf.if.
        for (unsigned r = 0; r < innerIf.getNumResults(); ++r)
          innerIf.getResult(r).replaceAllUsesWith(newIf.getResult(r));
        innerIf.erase();

        // The for yield now consumes the new scf.if result (i32).
        yieldOp.setOperand(idx, newIf.getResult(ifResultIdx));
      }

      // Update the for result type.
      forOp.getResult(idx).setType(intTy);

      // Replace external consumer casts.
      Value result = forOp.getResult(idx);
      SmallVector<UnrealizedConversionCastOp> castsToErase;
      for (OpOperand &use : result.getUses()) {
        auto userCast =
            dyn_cast<UnrealizedConversionCastOp>(use.getOwner());
        if (userCast && userCast.getResults().size() == 1 &&
            userCast.getResults()[0].getType() == result.getType()) {
          userCast.getResults()[0].replaceAllUsesWith(result);
          castsToErase.push_back(userCast);
        }
      }
      for (auto c : castsToErase)
        c.erase();
    }

    changed = true;
  }

  return changed;
}

/// DCE unrealized_conversion_cast ops that are now dead after
/// rewriting (the i32->f32 casts whose only user was scf.yield).
static void cleanupDeadCasts(ModuleOp mod) {
  SmallVector<UnrealizedConversionCastOp> deadCasts;
  mod.walk([&](UnrealizedConversionCastOp cast) {
    if (cast.use_empty())
      deadCasts.push_back(cast);
  });
  for (auto cast : deadCasts)
    cast.erase();
}

struct TTLLowerScalarFpTypesPass
    : impl::TTLLowerScalarFpTypesBase<TTLLowerScalarFpTypesPass> {
  using TTLLowerScalarFpTypesBase::TTLLowerScalarFpTypesBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    bool hadError = false;

    // Phase 1: Lower arith.cmpf to TTKernel soft-float comparison ops.
    // This runs first so that iter_arg uses by cmpf are replaced with
    // unrealized_conversion_casts, enabling scf.for propagation in
    // phase 2.
    mod.walk([&](arith::CmpFOp cmpOp) {
      Type floatTy = cmpOp.getLhs().getType();

      unsigned bitWidth;
      if (floatTy.isF32()) {
        bitWidth = 32;
      } else if (floatTy.isBF16()) {
        bitWidth = 16;
      } else {
        cmpOp.emitOpError("unsupported float type for scalar comparison: ")
            << floatTy;
        hadError = true;
        return;
      }

      OpBuilder builder(cmpOp);
      Location loc = cmpOp.getLoc();
      auto intTy = IntegerType::get(builder.getContext(), bitWidth);

      auto lhsInt =
          utils::materializeIntBits(cmpOp.getLhs(), intTy, builder, loc);
      auto rhsInt =
          utils::materializeIntBits(cmpOp.getRhs(), intTy, builder, loc);

      if (failed(lhsInt) || failed(rhsInt)) {
        cmpOp.emitOpError(
            "could not resolve float operand to integer bit pattern; "
            "operands must come from raw_element_read or float constants");
        hadError = true;
        return;
      }

      Value result;
      auto pred = cmpOp.getPredicate();

      switch (pred) {
      case arith::CmpFPredicate::OGT: {
        if (bitWidth == 32) {
          result = ttk::Float32GreaterOp::create(
              builder, loc, builder.getI1Type(), *lhsInt, *rhsInt);
        } else {
          result = ttk::Bfloat16GreaterOp::create(
              builder, loc, builder.getI1Type(), *lhsInt, *rhsInt);
        }
        break;
      }
      case arith::CmpFPredicate::OLT: {
        if (bitWidth == 32) {
          result = ttk::Float32GreaterOp::create(
              builder, loc, builder.getI1Type(), *rhsInt, *lhsInt);
        } else {
          result = ttk::Bfloat16GreaterOp::create(
              builder, loc, builder.getI1Type(), *rhsInt, *lhsInt);
        }
        break;
      }
      default:
        cmpOp.emitOpError("unsupported cmpf predicate for soft-float "
                          "lowering; only ogt and olt are "
                          "currently supported");
        hadError = true;
        return;
      }

      cmpOp.replaceAllUsesWith(result);
      cmpOp.erase();
    });

    // Phase 2: Propagate integer types through control flow. Now that
    // cmpf has been lowered, iter_args are only used through casts and
    // scf.if yields, enabling the propagation to match.
    bool propagated = true;
    while (propagated) {
      propagated = false;
      propagated |= propagateIntTypeThroughScfIf(mod);
      propagated |= propagateIntTypeThroughScfFor(mod);
      cleanupDeadCasts(mod);
    }

    // Final cleanup: remove any remaining dead casts.
    cleanupDeadCasts(mod);

    if (hadError) {
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::tt::ttl
