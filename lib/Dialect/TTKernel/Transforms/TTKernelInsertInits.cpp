// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Implementation of the TTKernelInsertInits pass, which inserts both common
// inits (init_sfpu, binary_op_init_common) that configure UNPACK + PACK data
// format routing, and per-op inits (exp_tile_init, add_tiles_init, etc.) that
// configure the MATH pipeline.
//
// Two phases:
//   1. Common inits: one per sync region, hoisted above enclosing loops.
//      Scans each tile_regs_acquire -> tile_regs_release region to determine
//      the compute category (FPU binary vs SFPU/copy/bcast) and derives
//      input/output CBs from compute and pack ops.
//   2. Per-op inits: placed at the consumer. A uniform scf region with no
//      DST sync takes one init immediately before the region. Mixed
//      consumers keep an init immediately before each consumer, inside the
//      region. DST synchronization preserves the configuration.
//      reduce_uninit is placed on the transition out of a definite reduce
//      configuration.
//
// TODO(#329): Emit init_short variants for cheaper re-inits on type switches.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Analysis/ConfigFlow.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelConfigEffects.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelTraits.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"

#define DEBUG_TYPE "ttkernel-insert-inits"

namespace mlir::tt::ttl {

namespace ttk = mlir::tt::ttkernel;

#define GEN_PASS_DEF_TTKERNELINSERTINITS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Compute-to-Init mapping
//===----------------------------------------------------------------------===//

/// Resolve an output CB Value from a CB index attribute on a compute op.
/// Looks up the ttkernel.get_compile_time_arg_val with the matching index.
/// TODO: cache the index→Value map per function to avoid O(N) walk per call.
static Value resolveOutputCB(Operation *computeOp, StringRef attrName) {
  auto cbIdxAttr = computeOp->getAttrOfType<IntegerAttr>(attrName);
  if (!cbIdxAttr) {
    return Value();
  }
  int64_t cbIdx = cbIdxAttr.getInt();
  auto funcOp = computeOp->getParentOfType<func::FuncOp>();
  Value result;
  funcOp->walk([&](ttk::GetCompileArgValOp argOp) {
    if (static_cast<int64_t>(argOp.getArgIndex()) == cbIdx) {
      result = argOp.getResult();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

static void createManualInit(OpBuilder &builder, Location loc,
                             ttk::AddTilesOp add) {
  ttk::AddTilesInitOp::create(builder, loc, add.getOperand(0),
                              add.getOperand(1));
}

static void createManualInit(OpBuilder &builder, Location loc,
                             ttk::SubTilesOp sub) {
  ttk::SubTilesInitOp::create(builder, loc, sub.getOperand(0),
                              sub.getOperand(1));
}

static void createManualInit(OpBuilder &builder, Location loc,
                             ttk::MulTilesOp mul) {
  ttk::MulTilesInitOp::create(builder, loc, mul.getOperand(0),
                              mul.getOperand(1));
}

static void createManualInit(OpBuilder &builder, Location loc,
                             ttk::MulIntTileOp multiply) {
  ttk::MulIntTileInitOp::create(builder, loc, multiply.getDtypeAttr());
}

static void createManualInit(OpBuilder &builder, Location loc,
                             ttk::CopyTileOp copy) {
  ttk::CopyTileInitOp::create(builder, loc, copy.getOperand(0));
}

static void createManualInit(OpBuilder &builder, Location loc,
                             ttk::BinaryDestReuseTilesOp reuse) {
  ttk::BinaryDestReuseTilesInitOp::create(builder, loc, reuse.getInCb(),
                                          reuse.getEltwiseBinaryTypeAttr(),
                                          reuse.getReuseTypeAttr());
}

static void createManualInit(OpBuilder &builder, Location loc,
                             ttk::MatmulBlockOp matmul) {
  ttk::MatmulBlockInitShortOp::create(builder, loc, matmul.getIn0CbId(),
                                      matmul.getIn1CbId(),
                                      matmul.getTranspose(), matmul.getCtDim(),
                                      matmul.getRtDim(), matmul.getKtDim());
}

static void createManualInit(OpBuilder &builder, Location loc,
                             ttk::ExperimentalMatmulBlockOp matmul) {
  ttk::MatmulBlockInitShortOp::create(builder, loc, matmul.getIn0CbId(),
                                      matmul.getIn1CbId(),
                                      matmul.getTranspose(), matmul.getCtDim(),
                                      matmul.getRtDim(), matmul.getKtDim());
}

static void createManualInit(OpBuilder &builder, Location loc,
                             ttk::UnaryBcastTileOp bcast) {
  Value outputCB = resolveOutputCB(bcast, kBcastOutputCBIndexAttrName);
  assert(outputCB && "output CB required for unary_bcast_init");
  ttk::UnaryBcastInitOp::create(builder, loc, bcast.getInCb(), outputCB,
                                bcast.getBcastTypeAttr());
}

static void createManualInit(OpBuilder &builder, Location loc,
                             ttk::ReduceTileOp reduce) {
  Value outputCB = resolveOutputCB(reduce, kReduceOutputCBIndexAttrName);
  assert(outputCB && "output CB required for reduce_init");
  ttk::ReduceInitOp::create(
      builder, loc, reduce.getInCb(), reduce.getScalingCb(), outputCB,
      reduce.getReduceTypeAttr(), reduce.getReduceDimAttr());
}

static void createManualInit(OpBuilder &builder, Location loc,
                             ttk::TypecastTileOp typecast) {
  ttk::TypecastTileInitOp::create(builder, loc, typecast.getInDtypeAttr(),
                                  typecast.getOutDtypeAttr());
}

static void createManualInit(OpBuilder &builder, Location loc,
                             ttk::ExpTileOp exp) {
  ttk::ExpTileInitOp::create(builder, loc, exp.getApproxAttr(),
                             exp.getScaleAttr(), exp.getInputClampingAttr());
}

static void createManualInit(OpBuilder &builder, Location loc,
                             ttk::TransposeTileOp transpose) {
  Value outputCB = resolveOutputCB(transpose, kTransposeOutputCBIndexAttrName);
  assert(outputCB && "output CB required for transpose_wh_init");
  ttk::TransposeInitOp::create(builder, loc, transpose.getIcb(), outputCB);
}

static void createManualInit(OpBuilder &builder, Location loc,
                             ttk::SFPUReduceTileOp reduce) {
  ttk::SFPUReduceInitOp::create(builder, loc, reduce.getReduceTypeAttr(),
                                reduce.getDataFormatAttr());
}

static void createPerOpInit(OpBuilder &builder, Location loc, Operation *op) {
  TypeSwitch<Operation *>(op)
#define MATH_INIT_AUTO(COMPUTE, INIT, COPS, IOPS, ATTRS, EXP)                  \
  .Case<ttk::COMPUTE>([&](ttk::COMPUTE) { ttk::INIT::create(builder, loc); })
#define MATH_INIT_MANUAL(COMPUTE, INIT, COPS, IOPS, ATTRS, EXP)                \
  .Case<ttk::COMPUTE>(                                                         \
      [&](ttk::COMPUTE compute) { createManualInit(builder, loc, compute); })
#include "ttlang/Dialect/TTKernel/IR/TTKernelMathInits.def"
      .Default([](Operation *) {
        llvm_unreachable("compute operation has no init builder");
      });
}

static bool isReduceInit(const ttk::MathInitDescriptor &descriptor) {
  return descriptor.init.getValue() == ttk::ReduceInitOp::getOperationName();
}

/// DST sync. MATH configuration is preserved across it, except a sync marked
/// as the first one after a definite reduce configuration.
static bool isSyncBoundary(Operation *op) {
  return isa<ttk::TileRegsAcquireOp, ttk::TileRegsCommitOp, ttk::TileRegsWaitOp,
             ttk::TileRegsReleaseOp>(op);
}

//===----------------------------------------------------------------------===//
// Common init insertion
//===----------------------------------------------------------------------===//

/// Scan a sync region (acquire -> release) including nested regions to find
/// input CBs, output CBs, and determine the compute category.
/// Returns true if FPU binary ops are present, false if not, failure on
/// error (missing tile_regs_release or mismatched output CB data formats).
///
/// Multiple output CBs are allowed when they share the same element type
/// (PACK data format routing is identical). The first output CB encountered
/// is returned for the common init.
/// Result of analyzing a sync region for common init insertion.
struct SyncRegionAnalysis {
  bool hasFPUBinary = false;
  bool hasMatmul = false;
  // For matmul: block dimensions from the first matmul_block op found.
  Value matmulTranspose, matmulCt, matmulRt, matmulKt;
};

static FailureOr<SyncRegionAnalysis>
analyzeSyncRegion(ttk::TileRegsAcquireOp acquireOp, Value &inputCB,
                  Value &in0CB, Value &in1CB, Value &outputCB) {
  Block *block = acquireOp->getBlock();
  SyncRegionAnalysis result;
  bool foundRelease = false;
  bool hadError = false;

  for (auto it = std::next(acquireOp->getIterator()); it != block->end();
       ++it) {
    if (isa<ttk::TileRegsReleaseOp>(&*it)) {
      foundRelease = true;
      break;
    }

    // Walk this op and all nested regions (e.g., scf.for bodies).
    (&*it)->walk([&](Operation *inner) {
      if (auto copy = dyn_cast<ttk::CopyTileOp>(inner)) {
        // copy_tile always precedes SFPU ops -- data must enter DST from a
        // CB before any SFPU/bcast compute can operate on it.
        if (!inputCB) {
          inputCB = copy.getCb0();
        }
      } else if (isa<ttk::AddTilesOp, ttk::SubTilesOp, ttk::MulTilesOp>(
                     inner)) {
        result.hasFPUBinary = true;
        if (!in0CB) {
          in0CB = inner->getOperand(0);
          in1CB = inner->getOperand(1);
        }
      } else if (auto binaryDestReuseOp =
                     dyn_cast<ttk::BinaryDestReuseTilesOp>(inner)) {
        // binary_dest_reuse_tiles uses the FPU binary unpack path for its DFB
        // operand even though the accumulator operand is already in DST, so
        // binary_op_init_common must be selected for the sync region.
        result.hasFPUBinary = true;
        if (!in0CB) {
          in0CB = binaryDestReuseOp.getInCb();
          in1CB = binaryDestReuseOp.getInCb();
        }
      } else if (auto matmul = dyn_cast<ttk::MatmulBlockOp>(inner)) {
        result.hasMatmul = true;
        if (!in0CB) {
          in0CB = matmul.getIn0CbId();
          in1CB = matmul.getIn1CbId();
        }
        if (!result.matmulTranspose) {
          result.matmulTranspose = matmul.getTranspose();
          result.matmulCt = matmul.getCtDim();
          result.matmulRt = matmul.getRtDim();
          result.matmulKt = matmul.getKtDim();
        }
      } else if (auto bcast = dyn_cast<ttk::UnaryBcastTileOp>(inner)) {
        if (!inputCB) {
          inputCB = bcast.getInCb();
        }
      } else if (auto reduce = dyn_cast<ttk::ReduceTileOp>(inner)) {
        if (!inputCB) {
          inputCB = reduce.getInCb();
        }
      } else if (auto transpose = dyn_cast<ttk::TransposeTileOp>(inner)) {
        if (!inputCB) {
          inputCB = transpose.getIcb();
        }
      } else if (auto normalization =
                     dyn_cast<ttk::ExperimentalRowNormalizationBlockOp>(
                         inner)) {
        if (!inputCB) {
          inputCB = normalization.getInputCb();
        }
        if (!outputCB) {
          outputCB = normalization.getOutputCb();
        }
      }
      // Collect output CB from pack ops (both single-tile and block variants).
      auto collectOutputCB = [&](Value packCB, Operation *packOp) {
        if (!outputCB) {
          outputCB = packCB;
        } else if (outputCB != packCB) {
          // PACK initialization depends on the DFB element type; capacity does
          // not affect the configured data format.
          mlir::Type outputElementType =
              mlir::cast<ttk::CBType>(outputCB.getType()).getElementType();
          mlir::Type packElementType =
              mlir::cast<ttk::CBType>(packCB.getType()).getElementType();
          if (outputElementType != packElementType) {
            packOp->emitOpError(
                "sync region packs to output CBs with different data formats; "
                "common init cannot configure multiple PACK formats");
            hadError = true;
          }
        }
      };
      if (auto pack = dyn_cast<ttk::PackTileOp>(inner)) {
        collectOutputCB(pack.getOutCb(), pack);
      } else if (auto pack = dyn_cast<ttk::PackWaitedTileOp>(inner)) {
        collectOutputCB(pack.getOutCb(), pack);
      } else if (auto packBlock = dyn_cast<ttk::PackTileBlockOp>(inner)) {
        collectOutputCB(packBlock.getOutCb(), packBlock);
      }
    });
  }

  if (!foundRelease) {
    acquireOp->emitOpError(
        "tile_regs_acquire without matching tile_regs_release");
    return failure();
  }
  if (hadError) {
    return failure();
  }
  return result;
}

/// Find the outermost enclosing insertion point by walking up through
/// loops with invariant CB configurations: compiler-generated tile/subblock
/// loops (ttl.tile_loop_stride, ttl.subblock_loop_stride) and L1
/// accumulation loops (ttl.l1_acc_loop). All use fixed CBs across
/// iterations, so init hoisting is safe. Stops at unmarked loops to avoid
/// hoisting past user loops with varying CB configurations.
static Operation *hoistAboveCompilerLoops(Operation *op) {
  Operation *insertBefore = op;
  while (auto *parentOp = insertBefore->getParentOp()) {
    if (isa<scf::ForOp>(parentOp) &&
        (parentOp->hasAttr(kTileLoopStrideAttrName) ||
         parentOp->hasAttr(kSubblockLoopStrideAttrName) ||
         parentOp->hasAttr(kL1AccLoopAttrName))) {
      insertBefore = parentOp;
    } else {
      break;
    }
  }
  return insertBefore;
}

/// Insert common init ops (init_sfpu or binary_op_init_common) before each
/// sync region. These configure UNPACK + PACK data format routing.
static LogicalResult insertCommonInits(ModuleOp moduleOp) {
  bool hadError = false;
  moduleOp->walk([&](ttk::TileRegsAcquireOp acquireOp) {
    Value inputCB, in0CB, in1CB, outputCB;
    auto analysisResult =
        analyzeSyncRegion(acquireOp, inputCB, in0CB, in1CB, outputCB);
    if (failed(analysisResult)) {
      hadError = true;
      return;
    }
    SyncRegionAnalysis analysis = *analysisResult;

    // No output CB means the sync region has no pack ops -- nothing to
    // configure for UNPACK + PACK routing.
    if (!outputCB) {
      return;
    }

    Operation *insertBefore = hoistAboveCompilerLoops(acquireOp);
    OpBuilder builder(insertBefore);
    Location loc = acquireOp->getLoc();

    // Fill-only regions have no copy_tile or bcast; route both sides of
    // init_sfpu through outputCB since fill writes directly to DST.
    if (!inputCB && outputCB) {
      inputCB = outputCB;
    }

    // Use init_short when sharing an output CB with a preceding sibling
    // annotated loop: the full init reconfigures PACK and clobbers packer
    // state (including L1 acc on Wormhole).
    bool useInitShort = false;
    if (analysis.hasMatmul) {
      if (auto forOp = dyn_cast<scf::ForOp>(insertBefore)) {
        if (forOp->hasAttr(kL1AccLoopAttrName) ||
            forOp->hasAttr(kReductionLoopAttrName)) {
          for (Operation *prev = forOp->getPrevNode(); prev;
               prev = prev->getPrevNode()) {
            if (auto prevFor = dyn_cast<scf::ForOp>(prev)) {
              if ((prevFor->hasAttr(kL1AccLoopAttrName) ||
                   prevFor->hasAttr(kReductionLoopAttrName)) &&
                  sharePackCB(prevFor, forOp)) {
                useInitShort = true;
              }
              break;
            }
          }
        }
      }
    }

    if (analysis.hasMatmul && in0CB && in1CB && useInitShort) {
      ttk::MatmulBlockInitShortOp::create(
          builder, loc, in0CB, in1CB, analysis.matmulTranspose,
          analysis.matmulCt, analysis.matmulRt, analysis.matmulKt);
    } else if (analysis.hasMatmul && in0CB && in1CB) {
      ttk::MatmulBlockInitOp::create(
          builder, loc, in0CB, in1CB, outputCB, analysis.matmulTranspose,
          analysis.matmulCt, analysis.matmulRt, analysis.matmulKt);
    } else if (analysis.hasFPUBinary && in0CB && in1CB) {
      ttk::BinaryOpInitCommonOp::create(builder, loc, in0CB, in1CB, outputCB);
    } else if (inputCB) {
      ttk::InitSFPUOp::create(builder, loc, inputCB, outputCB);
    }
  });
  return hadError ? failure() : success();
}

//===----------------------------------------------------------------------===//
// Per-op init insertion
//===----------------------------------------------------------------------===//

struct PlannedInit {
  Operation *compute = nullptr;
  Operation *insertionPoint = nullptr;
  bool needsReduceUninit = false;
};

/// MathInit state used to place per-op inits. `markSyncs` makes every DST
/// sync write unknown so the first sync after a reduce can be recorded.
/// Otherwise only syncs in `reduceSyncs` write unknown, and a compute op
/// virtually writes the descriptor it reads.
struct InsertSlot {
  using State = std::optional<ttk::MathInitDescriptor>;

  bool markSyncs = false;
  const llvm::DenseSet<Operation *> *reduceSyncs = nullptr;

  static State unknown() { return std::nullopt; }

  State join(const State &lhs, const State &rhs) const {
    if (lhs && rhs && *lhs == *rhs) {
      return lhs;
    }
    return std::nullopt;
  }

  std::optional<State> getWrite(Operation *op) const {
    if (isSyncBoundary(op)) {
      if (markSyncs || (reduceSyncs && reduceSyncs->contains(op))) {
        return std::optional<State>(std::in_place, std::nullopt);
      }
      return std::nullopt;
    }
    ttk::MathInitEffects effects = ttk::getMathInitEffects(op);
    if (effects.write) {
      return std::optional<State>(std::in_place, effects.write->descriptor);
    }
    if (effects.read && effects.read->descriptor) {
      return std::optional<State>(std::in_place, effects.read->descriptor);
    }
    return std::nullopt;
  }
};

struct HoistingSummary {
  bool hasSync = false;
  bool blocksHoisting = false;
  std::optional<ttk::MathInitDescriptor> descriptor;
};

/// Summarizes each operation subtree once. A summary is eligible for hoisting
/// when it has no synchronization, unsupported region, unknown configuration
/// access, or descriptor other than the one being hoisted.
class InitHoistingAnalysis {
public:
  bool canHoistThrough(Operation *scope,
                       const ttk::MathInitDescriptor &required) {
    const HoistingSummary &summary = summarize(scope);
    return !summary.hasSync && !summary.blocksHoisting &&
           (!summary.descriptor || *summary.descriptor == required);
  }

private:
  static bool isModeledRegionOperation(Operation *op) {
    return isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op);
  }

  static void addDescriptor(HoistingSummary &summary,
                            const ttk::MathInitDescriptor &descriptor) {
    if (summary.descriptor && *summary.descriptor != descriptor) {
      summary.blocksHoisting = true;
      return;
    }
    summary.descriptor = descriptor;
  }

  static void merge(HoistingSummary &summary, const HoistingSummary &nested) {
    summary.hasSync |= nested.hasSync;
    summary.blocksHoisting |= nested.blocksHoisting;
    if (nested.descriptor) {
      addDescriptor(summary, *nested.descriptor);
    }
  }

  const HoistingSummary &summarize(Operation *op) {
    auto found = summaries.find(op);
    if (found != summaries.end()) {
      return found->second;
    }

    HoistingSummary summary;
    summary.hasSync = isSyncBoundary(op);
    if (op->getNumRegions() != 0 && !isModeledRegionOperation(op)) {
      summary.blocksHoisting = true;
    }

    ttk::MathInitEffects effects = ttk::getMathInitEffects(op);
    if (effects.read) {
      if (effects.read->descriptor) {
        addDescriptor(summary, *effects.read->descriptor);
      } else {
        summary.blocksHoisting = true;
      }
    }
    if (effects.write) {
      if (effects.write->descriptor) {
        addDescriptor(summary, *effects.write->descriptor);
      } else {
        summary.blocksHoisting = true;
      }
    }

    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (Operation &nested : block) {
          merge(summary, summarize(&nested));
        }
      }
    }

    return summaries.try_emplace(op, std::move(summary)).first->second;
  }

  llvm::DenseMap<Operation *, HoistingSummary> summaries;
};

static bool operandsAvailableBefore(Operation *scope,
                                    ArrayRef<Value> operands) {
  for (Value operand : operands) {
    Operation *owner = nullptr;
    if (auto arg = dyn_cast<BlockArgument>(operand)) {
      owner = arg.getOwner()->getParentOp();
    } else if (Operation *def = operand.getDefiningOp()) {
      owner = def;
    }
    if (owner && scope->isAncestor(owner)) {
      return false;
    }
  }
  return true;
}

/// A loop or conditional whose reads all require one descriptor can take that
/// init before the region. Mixed descriptors stay at their consumers.
static Operation *initInsertionPoint(Operation *compute,
                                     const ttk::MathInitDescriptor &required,
                                     InitHoistingAnalysis &hoisting) {
  // Canonicalization hoists copy_tile_init. Leaving it at the copy lets that
  // pattern run after L1 accumulation has been inserted.
  if (isa<ttk::CopyTileOp>(compute)) {
    return compute;
  }
  Operation *point = compute;
  while (Operation *parent = point->getParentOp()) {
    if (!isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parent)) {
      break;
    }
    if (!operandsAvailableBefore(parent, required.operands) ||
        !hoisting.canHoistThrough(parent, required)) {
      break;
    }
    point = parent;
  }
  return point;
}

static void insertPerOpInits(func::FuncOp funcOp) {
  InsertSlot marker;
  marker.markSyncs = true;
  llvm::DenseSet<Operation *> reduceSyncs;
  ConfigFlow<InsertSlot> marking(marker);
  marking.run(funcOp.getBody(), InsertSlot::unknown(),
              [&](Operation *op, const InsertSlot::State &incoming) {
                if (isSyncBoundary(op) && incoming && isReduceInit(*incoming)) {
                  reduceSyncs.insert(op);
                }
              });

  InsertSlot planner;
  planner.reduceSyncs = &reduceSyncs;
  SmallVector<PlannedInit> planned;
  // Regions entered with a definite reduce configuration. The body header
  // joins that with the backedge, so the consumer itself may not see it.
  // A reduce that holds on only some paths, such as one branch of scf.if or
  // a dynamic-trip loop, does not emit reduce_uninit.
  llvm::DenseSet<Operation *> reduceRegions;
  ConfigFlow<InsertSlot> planning(planner);
  planning.run(funcOp.getBody(), InsertSlot::unknown(),
               [&](Operation *op, const InsertSlot::State &incoming) {
                 if (isa<scf::ForOp, scf::IfOp, scf::WhileOp>(op)) {
                   if (incoming && isReduceInit(*incoming)) {
                     reduceRegions.insert(op);
                   }
                   return;
                 }
                 ttk::MathInitEffects effects = ttk::getMathInitEffects(op);
                 if (!effects.read || !effects.read->descriptor) {
                   return;
                 }
                 const ttk::MathInitDescriptor &required =
                     *effects.read->descriptor;
                 if (incoming && *incoming == required) {
                   return;
                 }
                 bool leavingReduce = incoming && isReduceInit(*incoming) &&
                                      !isReduceInit(required);
                 if (!leavingReduce && !isReduceInit(required)) {
                   for (Operation *parent = op->getParentOp(); parent;
                        parent = parent->getParentOp()) {
                     if (!reduceRegions.erase(parent)) {
                       continue;
                     }
                     leavingReduce = true;
                     break;
                   }
                 }
                 planned.push_back({op, nullptr, leavingReduce});
               });

  InitHoistingAnalysis hoisting;
  for (PlannedInit &plan : planned) {
    ttk::MathInitEffects effects = ttk::getMathInitEffects(plan.compute);
    assert(effects.read && effects.read->descriptor &&
           "planned compute must declare a MathInit descriptor");
    plan.insertionPoint =
        initInsertionPoint(plan.compute, *effects.read->descriptor, hoisting);
  }

  for (Operation *sync : reduceSyncs) {
    OpBuilder builder(sync);
    ttk::ReduceUninitOp::create(builder, sync->getLoc());
  }
  // Both branches of one conditional can hoist the same init to that
  // conditional. One init at that point covers both.
  struct EmittedInit {
    Operation *point = nullptr;
    ttk::MathInitDescriptor descriptor;
    bool emittedUninit = false;
  };
  SmallVector<EmittedInit, 8> emitted;
  for (const PlannedInit &plan : planned) {
    ttk::MathInitEffects effects = ttk::getMathInitEffects(plan.compute);
    if (!effects.read || !effects.read->descriptor) {
      continue;
    }
    const ttk::MathInitDescriptor &required = *effects.read->descriptor;
    Operation *point = plan.insertionPoint;
    EmittedInit *existing = nullptr;
    for (EmittedInit &done : emitted) {
      if (done.point == point && done.descriptor == required) {
        existing = &done;
        break;
      }
    }
    // The first consumer at a shared point is the one that leaves reduce, so
    // a later plan for the same init never needs an uninit the first omitted.
    if (existing) {
      assert((!plan.needsReduceUninit || existing->emittedUninit) &&
             "shared init disagrees on reduce_uninit");
      continue;
    }
    OpBuilder builder(point);
    if (plan.needsReduceUninit) {
      ttk::ReduceUninitOp::create(builder, point->getLoc());
    }
    createPerOpInit(builder, plan.compute->getLoc(), plan.compute);
    emitted.push_back({point, required, plan.needsReduceUninit});
  }
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct TTKernelInsertInitsPass
    : public impl::TTKernelInsertInitsBase<TTKernelInsertInitsPass> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    if (failed(insertCommonInits(moduleOp))) {
      signalPassFailure();
      return;
    }
    moduleOp->walk([&](func::FuncOp funcOp) {
      if (!funcOp.isExternal()) {
        insertPerOpInits(funcOp);
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
