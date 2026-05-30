// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTKernel Insert L1 Accumulation
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

#define DEBUG_TYPE "ttkernel-insert-l1-accumulation"

namespace mlir::tt::ttl {

namespace ttk = mlir::tt::ttkernel;

#define GEN_PASS_DEF_TTKERNELINSERTL1ACCUMULATION
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

namespace ttcore = mlir::tt::ttcore;

/// Build an i32 constant at the builder's insertion point.
static Value buildI32Const(OpBuilder &builder, Location loc, int32_t value) {
  return arith::ConstantOp::create(builder, loc, builder.getI32Type(),
                                   builder.getI32IntegerAttr(value));
}

/// Find the innermost enclosing L1 acc or reduction loop.
/// User-written += loops (kL1AccLoopAttrName) take precedence over
/// compiler-generated reduction loops because the user-specified loop
/// structure determines the accumulation granularity.
static scf::ForOp findL1AccLoop(Operation *op) {
  scf::ForOp reductionFallback;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      if (forOp->hasAttr(kL1AccLoopAttrName)) {
        return forOp;
      }
      if (forOp->hasAttr(kReductionLoopAttrName) && !reductionFallback) {
        reductionFallback = forOp;
      }
    }
  }
  return reductionFallback;
}

/// Return the L1 initial mode declared by the loop producer. TTKernel lowering
/// consumes this semantic contract directly instead of inferring the initial
/// value from surrounding pack/reserve/push operations after conversion.
static FailureOr<AccumulationInitialMode> getL1AccInitialMode(scf::ForOp loop) {
  auto attr =
      loop->getAttrOfType<AccumulationInitialModeAttr>(kL1AccInitialAttrName);
  if (!attr) {
    return failure();
  }
  AccumulationInitialMode mode = attr.getValue();
  if (mode == AccumulationInitialMode::Overwrite ||
      mode == AccumulationInitialMode::AccumulateExisting) {
    return mode;
  }
  return failure();
}

/// Return true when an overwrite-mode loop may execute iteration 1 or later.
/// Only packs from those later iterations should accumulate onto the iteration-0
/// baseline. A known 0- or 1-trip loop has no such pack. Unknown trip counts
/// keep the conditional enable because runtime trip count may exceed one.
static bool mayNeedOverwriteModeEnable(scf::ForOp loop) {
  std::optional<llvm::APInt> tripCount = loop.getStaticTripCount();
  return !tripCount || tripCount->ugt(1);
}

/// Return true when the packer can add a packed tile into an existing L1 tile
/// for this output data type.
static bool isL1AccumulationDataTypeSupported(ttcore::DataType dataType) {
  // Mirrors tt-metal's pack L1-acc format coverage:
  // https://github.com/tenstorrent/tt-metal/blob/9938a888cc4efd766d7652c08ab7eeb8fedd9aaf/tt_metal/tt-llk/tests/python_tests/quasar/test_pack_l1_acc_quasar.py#L42-L51
  // TTCore has no Int8 tile data type, so Int8 is absent here.
  switch (dataType) {
  case ttcore::DataType::Float32:
  case ttcore::DataType::Float16:
  case ttcore::DataType::BFloat16:
  case ttcore::DataType::Int32:
  case ttcore::DataType::UInt8:
    return true;
  default:
    return false;
  }
}

/// Verify that every pack in an L1 accumulation loop targets a supported output
/// format. Packer L1 accumulation is an additive write to the destination data
/// format, and unsupported formats have no valid L1-accumulation behavior.
static LogicalResult verifyL1AccumulationPackFormats(scf::ForOp loop) {
  LogicalResult result = success();
  loop->walk([&](ttk::PackTileOp packOp) {
    auto cbType = dyn_cast<ttk::CBType>(packOp.getOutCb().getType());
    if (!cbType) {
      result = packOp.emitOpError(
          "L1 packer accumulation requires a typed output dataflow buffer");
      return WalkResult::interrupt();
    }

    auto tileType = dyn_cast<ttcore::TileType>(cbType.getElementType());
    if (!tileType) {
      result = packOp.emitOpError(
          "L1 packer accumulation requires a tile output dataflow buffer");
      return WalkResult::interrupt();
    }

    ttcore::DataType dataType = tileType.getDataType();
    if (!isL1AccumulationDataTypeSupported(dataType)) {
      result = packOp.emitOpError()
               << "L1 packer accumulation does not support output data type "
               << ttcore::DataTypeEnumToString(dataType);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

struct TTKernelInsertL1AccumulationPass
    : public impl::TTKernelInsertL1AccumulationBase<
          TTKernelInsertL1AccumulationPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    bool hadFailure = false;

    // Walk from TileRegsAcquireOp upward to find annotated loops; only loops
    // with actual pack activity need L1 accumulation guards.
    SmallVector<scf::ForOp> l1AccLoops;
    llvm::SmallDenseSet<Operation *> visitedLoops;
    moduleOp->walk([&](ttk::TileRegsAcquireOp acquireOp) {
      auto loop = findL1AccLoop(acquireOp);
      if (!loop || !visitedLoops.insert(loop).second) {
        return;
      }
      bool alreadyProcessed = false;
      if (auto *prev = loop->getPrevNode()) {
        alreadyProcessed = isa<ttk::PackReconfigL1AccOp>(prev);
      }
      if (!alreadyProcessed) {
        loop->walk([&](ttk::PackReconfigL1AccOp) {
          alreadyProcessed = true;
          return WalkResult::interrupt();
        });
      }
      if (alreadyProcessed) {
        return;
      }
      if (failed(getL1AccInitialMode(loop))) {
        loop.emitOpError() << "requires " << kL1AccInitialAttrName
                           << " overwrite or accumulate_existing metadata";
        hadFailure = true;
        return;
      }
      // Packer L1 accumulation adds the packed tile into the existing L1 value.
      // Max reduction packs must write the max result instead, so this pass
      // leaves those loops without L1-accumulation guards.
      bool hasMaxReduce = false;
      loop->walk([&](ttk::ReduceTileOp reduceOp) {
        if (reduceOp.getReduceType() == ttk::ReduceType::Max) {
          hasMaxReduce = true;
        }
      });
      if (!hasMaxReduce) {
        if (failed(verifyL1AccumulationPackFormats(loop))) {
          hadFailure = true;
          return;
        }
        l1AccLoops.push_back(loop);
      }
    });
    if (hadFailure) {
      signalPassFailure();
      return;
    }

    // Insertion point for the per-iteration enable: the top-level ancestor
    // of the last tile_regs_release in the loop body, since packs may be
    // nested in subblock loops.
    llvm::SmallDenseMap<Operation *, Operation *> l1AccEnablePoint;
    for (auto loop : l1AccLoops) {
      Operation *lastReleaseAncestor = nullptr;
      loop->walk([&](ttk::TileRegsReleaseOp releaseOp) {
        if (auto *ancestor =
                loop.getBody()->findAncestorOpInBlock(*releaseOp)) {
          lastReleaseAncestor = ancestor;
        }
      });
      if (lastReleaseAncestor) {
        l1AccEnablePoint[loop.getOperation()] = lastReleaseAncestor;
      }
    }

    // Group consecutive sibling loops that pack to the same dataflow buffer.
    auto groups = collectLoopGroups(l1AccLoops, l1AccEnablePoint);

    // Emit guards per group.
    for (auto &group : groups) {
      OpBuilder builder(group.rootLoop->getContext());
      Location disableLoc = group.rootLoop->getLoc();

      // Reconfig L1 acc immediately before the first loop in the group. The
      // semantic loop metadata determines whether iteration 0 overwrites L1 or
      // accumulates onto a value materialized before the group.
      FailureOr<AccumulationInitialMode> initialMode =
          getL1AccInitialMode(group.rootLoop);
      assert(succeeded(initialMode) && "validated before grouping");
      bool l1HasPriorValue =
          *initialMode == AccumulationInitialMode::AccumulateExisting;

      builder.setInsertionPoint(group.rootLoop);
      Value beforeGroupFlag =
          buildI32Const(builder, disableLoc, l1HasPriorValue ? 1 : 0);
      ttk::PackReconfigL1AccOp::create(builder, disableLoc, beforeGroupFlag);

      for (size_t idx = 0; idx < group.loops.size(); ++idx) {
        scf::ForOp loop = group.loops[idx];
        auto iter = l1AccEnablePoint.find(loop.getOperation());
        if (iter == l1AccEnablePoint.end()) {
          continue;
        }

        if (idx == 0 && l1HasPriorValue) {
          continue;
        }

        // Init ops between sibling loops reset packer state, so each
        // non-first loop needs an unconditional enable.
        if (idx > 0) {
          builder.setInsertionPoint(loop);
          Value enableFlag = buildI32Const(builder, loop->getLoc(), 1);
          ttk::PackReconfigL1AccOp::create(builder, loop->getLoc(), enableFlag);
        }

        if (!mayNeedOverwriteModeEnable(loop)) {
          continue;
        }

        // Iteration 0 creates the baseline output tile. Enable after its last
        // pack so later iterations add into that L1 value.
        Operation *afterOp = iter->second;
        Location loc = afterOp->getLoc();
        builder.setInsertionPointAfter(afterOp);
        Value firstIter =
            arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::eq,
                                  loop.getInductionVar(), loop.getLowerBound());
        auto ifOp = scf::IfOp::create(builder, loc, firstIter);
        builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
        Value enableFlag = buildI32Const(builder, loc, 1);
        ttk::PackReconfigL1AccOp::create(builder, loc, enableFlag);
      }

      // Disable L1 acc after the group's scope end (typically cb_push_back).
      builder.setInsertionPointAfter(group.scopeEnd);
      Value afterGroupFlag = l1HasPriorValue
                                 ? buildI32Const(builder, disableLoc, 0)
                                 : beforeGroupFlag;
      ttk::PackReconfigL1AccOp::create(builder, disableLoc, afterGroupFlag);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
