// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTKernel Insert L1 Accumulation
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/AccumulationUtils.h"

#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

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

/// Return true when `loop` carries metadata for a packer L1 accumulation loop.
static bool isL1AccumulationLoop(scf::ForOp loop) {
  return loop->hasAttr(kL1AccLoopAttrName) ||
         loop->hasAttr(kReductionLoopAttrName);
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

/// Return the scope id declared by the loop producer. Loops with the same id
/// belong to one semantic accumulation scope and share one packer L1
/// accumulation lifecycle.
static FailureOr<int64_t> getL1AccScopeId(scf::ForOp loop) {
  auto attr = loop->getAttrOfType<IntegerAttr>(kL1AccScopeIdAttrName);
  if (!attr) {
    return failure();
  }
  return attr.getInt();
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

/// Verify metadata that must be supplied by TTL accumulation strategy lowering.
static LogicalResult verifyL1AccLoopMetadata(scf::ForOp loop) {
  if (failed(getL1AccInitialMode(loop))) {
    return loop.emitOpError() << "requires " << kL1AccInitialAttrName
                              << " metadata with value overwrite or "
                                 "accumulate_existing; run "
                                 "ttl-lower-accumulation-scopes before "
                                 "ttkernel-insert-l1-accumulation";
  }
  if (failed(getL1AccScopeId(loop))) {
    return loop.emitOpError()
           << "requires " << kL1AccScopeIdAttrName
           << " metadata; run ttl-lower-accumulation-scopes before "
              "ttkernel-insert-l1-accumulation";
  }
  return success();
}

/// Return true when an overwrite-mode loop may execute iteration 1 or later.
/// Only packs from those later iterations should accumulate onto the
/// iteration-0 baseline. A known 0- or 1-trip loop has no such pack. Unknown
/// trip counts keep the conditional enable because runtime trip count may
/// exceed one.
static bool mayNeedOverwriteModeEnable(scf::ForOp loop) {
  std::optional<int64_t> tripCount = getStaticAccumulationTripCount(loop);
  return !tripCount || *tripCount > 1;
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

static void addPackCBs(scf::ForOp loop,
                       llvm::SmallDenseSet<Value, 2> &packCBs) {
  llvm::SmallDenseSet<Value, 2> loopPackCBs = getPackTileCBs(loop);
  packCBs.insert(loopPackCBs.begin(), loopPackCBs.end());
}

static bool packsToAnyCB(Operation *operation,
                         const llvm::SmallDenseSet<Value, 2> &packCBs) {
  if (auto packOp = dyn_cast<ttk::PackTileOp>(operation)) {
    return packCBs.contains(packOp.getOutCb());
  }
  if (auto packOp = dyn_cast<ttk::PackTileBlockOp>(operation)) {
    return packCBs.contains(packOp.getOutCb());
  }
  return false;
}

static void addScopeOutputCBs(Operation *operation,
                              const llvm::SmallDenseSet<Value, 2> &packCBs,
                              llvm::SmallDenseSet<Value, 2> &scopeOutputCBs) {
  auto pushOp = dyn_cast<ttk::CBPushBackOp>(operation);
  if (pushOp && packCBs.contains(pushOp.getCb())) {
    scopeOutputCBs.insert(pushOp.getCb());
  }
}

static bool containsAnyPack(Operation *operation) {
  bool found = false;
  operation->walk([&](Operation *nested) {
    if (isa<ttk::PackTileOp, ttk::PackTileBlockOp>(nested)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool mayResetPackerL1Acc(Operation *operation) {
  // Copy initialization does not reset packer L1-accumulation state.
  return isa<ttk::PackReconfigDataFormatOp>(operation) ||
         (operation->hasTrait<ttk::TTKernelInitOpTrait>() &&
          !isa<ttk::CopyTileInitOp>(operation));
}

static bool containsPackReconfigL1Acc(Operation *operation) {
  bool found = false;
  operation->walk([&](ttk::PackReconfigL1AccOp) {
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

static bool hasPackerL1AccResetSinceReconfig(Operation *operation,
                                             scf::ForOp boundaryLoop) {
  for (Operation *cursor = operation; cursor;) {
    Block *block = cursor->getBlock();
    if (!block) {
      return false;
    }
    for (auto iter = Block::reverse_iterator(Block::iterator(cursor));
         iter != block->rend(); ++iter) {
      Operation *candidate = &*iter;
      if (containsPackReconfigL1Acc(candidate)) {
        return false;
      }
      if (mayResetPackerL1Acc(candidate)) {
        return true;
      }
    }
    Operation *parent = block->getParentOp();
    if (!parent || parent == boundaryLoop.getOperation()) {
      return false;
    }
    cursor = parent;
  }
  return false;
}

static void
insertLocalL1AccEnableAfterReset(OpBuilder &builder, scf::ForOp loop,
                                 const llvm::SmallDenseSet<Value, 2> &packCBs,
                                 AccumulationInitialMode initialMode) {
  llvm::SmallDenseSet<Operation *, 4> visitedPacks;
  loop->walk([&](Operation *operation) {
    if (!packsToAnyCB(operation, packCBs) ||
        !visitedPacks.insert(operation).second ||
        findL1AccLoop(operation) != loop ||
        !hasPackerL1AccResetSinceReconfig(operation, loop)) {
      return;
    }

    Location loc = operation->getLoc();
    builder.setInsertionPoint(operation);
    if (initialMode == AccumulationInitialMode::AccumulateExisting) {
      Value enableFlag = buildI32Const(builder, loc, 1);
      ttk::PackReconfigL1AccOp::create(builder, loc, enableFlag);
      return;
    }

    if (!mayNeedOverwriteModeEnable(loop)) {
      return;
    }

    Value laterIteration =
        arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                              loop.getInductionVar(), loop.getLowerBound());
    auto ifOp = scf::IfOp::create(builder, loc, laterIteration);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    Value enableFlag = buildI32Const(builder, loc, 1);
    ttk::PackReconfigL1AccOp::create(builder, loc, enableFlag);
  });
}

static void insertL1AccEnableAfterPack(OpBuilder &builder, scf::ForOp loop,
                                       Operation *operation,
                                       AccumulationInitialMode initialMode) {
  Location loc = operation->getLoc();
  builder.setInsertionPointAfter(operation);
  if (initialMode == AccumulationInitialMode::AccumulateExisting) {
    Value enableFlag = buildI32Const(builder, loc, 1);
    ttk::PackReconfigL1AccOp::create(builder, loc, enableFlag);
    return;
  }

  if (!mayNeedOverwriteModeEnable(loop)) {
    return;
  }

  Value laterIteration =
      arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::ne,
                            loop.getInductionVar(), loop.getLowerBound());
  auto ifOp = scf::IfOp::create(builder, loc, laterIteration);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  Value enableFlag = buildI32Const(builder, loc, 1);
  ttk::PackReconfigL1AccOp::create(builder, loc, enableFlag);
}

static void insertNonScopePackL1AccGuards(
    OpBuilder &builder, scf::ForOp loop,
    const llvm::SmallDenseSet<Value, 2> &scopeOutputCBs,
    AccumulationInitialMode initialMode) {
  SmallVector<Operation *> nonScopePacks;
  loop->walk([&](Operation *operation) {
    if (findL1AccLoop(operation) != loop) {
      return;
    }
    if (auto packOp = dyn_cast<ttk::PackTileOp>(operation)) {
      if (!scopeOutputCBs.contains(packOp.getOutCb())) {
        nonScopePacks.push_back(operation);
      }
      return;
    }
    if (auto packOp = dyn_cast<ttk::PackTileBlockOp>(operation)) {
      if (!scopeOutputCBs.contains(packOp.getOutCb())) {
        nonScopePacks.push_back(operation);
      }
    }
  });

  for (Operation *operation : nonScopePacks) {
    Location loc = operation->getLoc();
    builder.setInsertionPoint(operation);
    Value disableFlag = buildI32Const(builder, loc, 0);
    ttk::PackReconfigL1AccOp::create(builder, loc, disableFlag);
    insertL1AccEnableAfterPack(builder, loop, operation, initialMode);
  }
}

/// Verify that every accumulating pack targets a supported output format.
/// Packer L1 accumulation is an additive write to the destination data format.
static LogicalResult verifyL1AccumulationPackFormats(
    scf::ForOp loop, const llvm::SmallDenseSet<Value, 2> &scopeOutputCBs) {
  LogicalResult result = success();
  auto verifyPackOutput = [&](Operation *packOp, Value outCB) {
    if (!scopeOutputCBs.contains(outCB)) {
      return WalkResult::advance();
    }

    auto cbType = dyn_cast<ttk::CBType>(outCB.getType());
    if (!cbType) {
      result = packOp->emitOpError(
          "L1 packer accumulation requires the pack output to be a typed "
          "dataflow buffer");
      return WalkResult::interrupt();
    }

    auto tileType = dyn_cast<ttcore::TileType>(cbType.getElementType());
    if (!tileType) {
      result = packOp->emitOpError(
          "L1 packer accumulation requires the pack output dataflow buffer to "
          "hold tile elements");
      return WalkResult::interrupt();
    }

    ttcore::DataType dataType = tileType.getDataType();
    if (!isL1AccumulationDataTypeSupported(dataType)) {
      result = packOp->emitOpError()
               << "L1 packer accumulation does not support output data type "
               << ttcore::DataTypeEnumToString(dataType)
               << "; use a supported output data type or select another "
                  "accumulation strategy";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  };

  loop->walk([&](Operation *operation) {
    if (auto packOp = dyn_cast<ttk::PackTileOp>(operation)) {
      return verifyPackOutput(packOp.getOperation(), packOp.getOutCb());
    }
    if (auto packOp = dyn_cast<ttk::PackTileBlockOp>(operation)) {
      return verifyPackOutput(packOp.getOperation(), packOp.getOutCb());
    }
    return WalkResult::advance();
  });
  return result;
}

struct L1AccumulationLoopGroup {
  scf::ForOp rootLoop;
  SmallVector<scf::ForOp> loops;
  llvm::SmallDenseSet<Value, 2> packCBs;
  llvm::SmallDenseSet<Value, 2> scopeOutputCBs;
  Operation *scopeEnd = nullptr;
};

/// Return the outermost annotated loop that participates in `scopeId`.
/// This lowering currently models one active packer L1 accumulation
/// configuration per lexical loop nest. Nested independent scopes require
/// explicit state transitions when entering and leaving the inner scope.
// TODO(#648): Model explicit packer L1 accumulation state transitions before
// allowing nested independent scope ids.
static FailureOr<scf::ForOp> findScopeRoot(scf::ForOp loop, int64_t scopeId) {
  scf::ForOp rootLoop = loop;
  for (Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto parentLoop = dyn_cast<scf::ForOp>(parent);
    if (!parentLoop || !isL1AccumulationLoop(parentLoop)) {
      continue;
    }
    if (failed(verifyL1AccLoopMetadata(parentLoop))) {
      return failure();
    }
    FailureOr<int64_t> parentScopeId = getL1AccScopeId(parentLoop);
    assert(succeeded(parentScopeId) && "verified above");
    if (*parentScopeId != scopeId) {
      loop.emitOpError()
          << "nested independent L1 accumulation scopes are not supported "
             "(#648); nested loops that belong to one accumulation must use "
             "matching "
          << kL1AccScopeIdAttrName << " metadata";
      return failure();
    }
    rootLoop = parentLoop;
  }
  return rootLoop;
}

/// Return true when `loop` is an annotated loop in `scopeId`.
static bool isLoopInScope(scf::ForOp loop, int64_t scopeId) {
  FailureOr<int64_t> loopScopeId = getL1AccScopeId(loop);
  return succeeded(loopScopeId) && *loopScopeId == scopeId &&
         isL1AccumulationLoop(loop);
}

/// Collect loops by explicit scope id. The scope id encodes semantic grouping;
/// the remaining sibling scan only finds adjacent loops with the same id and
/// the dataflow-buffer push that bounds packer L1 accumulation.
static FailureOr<SmallVector<L1AccumulationLoopGroup>>
collectL1AccumulationLoopGroups(
    ArrayRef<scf::ForOp> l1AccLoops,
    const llvm::SmallDenseMap<Operation *, Operation *> &enablePointPerLoop) {
  SmallVector<L1AccumulationLoopGroup> groups;
  llvm::SmallDenseSet<Operation *> assignedLoops;
  llvm::SmallDenseSet<Operation *> candidateLoops;
  for (scf::ForOp loop : l1AccLoops) {
    candidateLoops.insert(loop.getOperation());
  }

  for (scf::ForOp loop : l1AccLoops) {
    if (!enablePointPerLoop.count(loop.getOperation()) ||
        assignedLoops.contains(loop.getOperation())) {
      continue;
    }

    FailureOr<int64_t> scopeId = getL1AccScopeId(loop);
    if (failed(scopeId)) {
      loop.emitOpError()
          << "requires " << kL1AccScopeIdAttrName
          << " metadata; run ttl-lower-accumulation-scopes before "
             "ttkernel-insert-l1-accumulation";
      return failure();
    }

    FailureOr<scf::ForOp> rootLoop = findScopeRoot(loop, *scopeId);
    if (failed(rootLoop)) {
      return failure();
    }

    L1AccumulationLoopGroup group;
    group.rootLoop = *rootLoop;
    group.scopeEnd = group.rootLoop;
    group.loops.push_back(loop);
    addPackCBs(loop, group.packCBs);
    assignedLoops.insert(loop.getOperation());

    llvm::SmallDenseSet<Operation *> groupRootLoops;
    groupRootLoops.insert(group.rootLoop.getOperation());

    for (Operation *operation = group.rootLoop->getNextNode(); operation;
         operation = operation->getNextNode()) {
      if (isa<ttk::CBPushBackOp, ttk::CBReserveBackOp>(operation)) {
        break;
      }

      auto siblingLoop = dyn_cast<scf::ForOp>(operation);
      if (!siblingLoop) {
        if (containsAnyPack(operation)) {
          break;
        }
        continue;
      }
      if (!isL1AccumulationLoop(siblingLoop)) {
        if (containsAnyPack(siblingLoop)) {
          break;
        }
        continue;
      }

      if (failed(verifyL1AccLoopMetadata(siblingLoop))) {
        return failure();
      }
      FailureOr<int64_t> siblingScopeId = getL1AccScopeId(siblingLoop);
      assert(succeeded(siblingScopeId) && "verified above");
      if (*siblingScopeId != *scopeId) {
        break;
      }

      groupRootLoops.insert(siblingLoop.getOperation());
      if (candidateLoops.contains(siblingLoop.getOperation()) &&
          !assignedLoops.contains(siblingLoop.getOperation()) &&
          enablePointPerLoop.count(siblingLoop.getOperation())) {
        group.loops.push_back(siblingLoop);
        addPackCBs(siblingLoop, group.packCBs);
        assignedLoops.insert(siblingLoop.getOperation());
      }
    }

    for (Operation *operation = group.rootLoop->getNextNode(); operation;
         operation = operation->getNextNode()) {
      if (isa<ttk::CBPushBackOp>(operation)) {
        addScopeOutputCBs(operation, group.packCBs, group.scopeOutputCBs);
        if (!group.scopeOutputCBs.empty()) {
          group.scopeEnd = operation;
        }
        continue;
      }
      if (isa<ttk::CBReserveBackOp>(operation)) {
        // A later reserve may introduce packs for another output before this
        // scope's push. Disabling before that reserve is conservative because
        // L1-acc state affects subsequent packs, not cb_push_back itself.
        break;
      }
      if (!isa<scf::ForOp>(operation) && containsAnyPack(operation)) {
        break;
      }

      auto siblingLoop = dyn_cast<scf::ForOp>(operation);
      if (!siblingLoop) {
        continue;
      }
      if (groupRootLoops.contains(siblingLoop.getOperation())) {
        group.scopeEnd = operation;
        continue;
      }
      if (containsAnyPack(operation)) {
        break;
      }
      if (isL1AccumulationLoop(siblingLoop) &&
          !isLoopInScope(siblingLoop, *scopeId)) {
        break;
      }
    }

    if (group.scopeOutputCBs.empty()) {
      group.scopeOutputCBs.insert(group.packCBs.begin(), group.packCBs.end());
    }
    groups.push_back(std::move(group));
  }

  return groups;
}

struct TTKernelInsertL1AccumulationPass
    : public impl::TTKernelInsertL1AccumulationBase<
          TTKernelInsertL1AccumulationPass> {
  void runOnOperation() override {
    auto moduleOp = getOperation();
    bool hadFailure = false;

    // Walk from TileRegsAcquireOp upward to find annotated loops; only loops
    // with actual pack activity need packer L1 accumulation reconfiguration.
    SmallVector<scf::ForOp> l1AccLoops;
    llvm::SmallDenseSet<Operation *> visitedLoops;
    moduleOp->walk([&](ttk::TileRegsAcquireOp acquireOp) {
      auto loop = findL1AccLoop(acquireOp);
      if (!loop || !visitedLoops.insert(loop).second) {
        return;
      }
      if (failed(verifyL1AccLoopMetadata(loop))) {
        hadFailure = true;
        return;
      }
      // Packer L1 accumulation adds the packed tile into the existing L1 value.
      // Max reduction packs must write the max result instead, so this pass
      // leaves those loops without packer L1 accumulation reconfiguration.
      bool hasMaxReduce = false;
      loop->walk([&](ttk::ReduceTileOp reduceOp) {
        if (reduceOp.getReduceType() == ttk::ReduceType::Max) {
          hasMaxReduce = true;
        }
      });
      if (!hasMaxReduce) {
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

    FailureOr<SmallVector<L1AccumulationLoopGroup>> groups =
        collectL1AccumulationLoopGroups(l1AccLoops, l1AccEnablePoint);
    if (failed(groups)) {
      signalPassFailure();
      return;
    }

    for (const auto &group : *groups) {
      for (scf::ForOp loop : group.loops) {
        if (failed(
                verifyL1AccumulationPackFormats(loop, group.scopeOutputCBs))) {
          signalPassFailure();
          return;
        }
      }
    }

    // Emit packer L1 accumulation reconfiguration for each semantic scope.
    for (auto &group : *groups) {
      // The marker after the semantic scope end is unambiguous. The marker
      // before the root loop may be the disable from a preceding independent
      // scope when scopes are adjacent.
      bool alreadyProcessed = false;
      if (auto *next = group.scopeEnd->getNextNode()) {
        while (next && isa<arith::ConstantOp>(next)) {
          next = next->getNextNode();
        }
        alreadyProcessed = isa<ttk::PackReconfigL1AccOp>(next);
      }
      if (!alreadyProcessed) {
        group.rootLoop->walk([&](ttk::PackReconfigL1AccOp) {
          alreadyProcessed = true;
          return WalkResult::interrupt();
        });
      }
      if (alreadyProcessed) {
        continue;
      }

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

      for (scf::ForOp loop : group.loops) {
        FailureOr<AccumulationInitialMode> loopInitialMode =
            getL1AccInitialMode(loop);
        assert(succeeded(loopInitialMode) && "validated before grouping");
        insertNonScopePackL1AccGuards(builder, loop, group.scopeOutputCBs,
                                      *loopInitialMode);
        insertLocalL1AccEnableAfterReset(builder, loop, group.scopeOutputCBs,
                                         *loopInitialMode);
      }

      // Disable L1 acc after the group's scope end (typically cb_push_back).
      // The disable flag is always 0; reuse beforeGroupFlag when overwrite mode
      // already built a 0, otherwise build a fresh 0 for accumulate-existing.
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
