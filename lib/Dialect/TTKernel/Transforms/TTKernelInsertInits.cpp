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
//   2. Per-op inits: emitted in linear block order whenever the op type
//      changes (unary SFPU, binary SFPU, minmax, FPU binary). The init
//      key is (init op TypeID, operand values). An init is inserted only
//      when the key changes. Tracking resets at sync boundaries.
//
// TODO(#329): Emit init_short variants for cheaper re-inits on type switches.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelTraits.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"

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

/// Information about how to create an init op for a given compute op.
struct InitOpInfo {
  std::function<void(OpBuilder &, Location, Operation *)> createInit;
};

/// Build a static map from TTKernel compute op TypeID to init creation info.
/// Uses the same x-macro table as ConvertTTLTileOpsToTTKernel.
static llvm::DenseMap<mlir::TypeID, InitOpInfo> buildComputeToInitMap() {
  llvm::DenseMap<mlir::TypeID, InitOpInfo> map;

#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  map[mlir::TypeID::get<ttk::TTK_COMPUTE>()] = {                               \
      [](OpBuilder &b, Location l, Operation *) {                              \
        ttk::TTK_INIT::create(b, l);                                           \
      }};
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)             \
  map[mlir::TypeID::get<ttk::TTK_COMPUTE>()] = {                               \
      [](OpBuilder &b, Location l, Operation *) {                              \
        ttk::TTK_INIT::create(b, l);                                           \
      }};
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

#define TTL_BINARY_TILE_OP_MINMAX(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)      \
  map[mlir::TypeID::get<ttk::TTK_COMPUTE>()] = {                               \
      [](OpBuilder &b, Location l, Operation *) {                              \
        ttk::TTK_INIT::create(b, l);                                           \
      }};
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

#define TTL_FPU_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)         \
  map[mlir::TypeID::get<ttk::TTK_COMPUTE>()] = {                               \
      [](OpBuilder &b, Location l, Operation *computeOp) {                     \
        ttk::TTK_INIT::create(b, l, computeOp->getOperand(0),                  \
                              computeOp->getOperand(1));                       \
      }};
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  map[mlir::TypeID::get<ttk::CopyTileOp>()] = {
      [](OpBuilder &b, Location l, Operation *computeOp) {
        ttk::CopyTileInitOp::create(b, l, computeOp->getOperand(0));
      }};

  map[mlir::TypeID::get<ttk::CopyDestValuesOp>()] = {
      [](OpBuilder &b, Location l, Operation *) {
        ttk::CopyDestValuesInitOp::create(b, l);
      }};

  map[mlir::TypeID::get<ttk::MatmulBlockOp>()] = {[](OpBuilder &b, Location l,
                                                     Operation *computeOp) {
    auto matmul = cast<ttk::MatmulBlockOp>(computeOp);
    ttk::MatmulBlockInitShortOp::create(
        b, l, matmul.getIn0CbId(), matmul.getIn1CbId(), matmul.getTranspose(),
        matmul.getCtDim(), matmul.getRtDim(), matmul.getKtDim());
  }};

  map[mlir::TypeID::get<ttk::UnaryBcastTileOp>()] = {
      [](OpBuilder &b, Location l, Operation *computeOp) {
        auto bcastOp = cast<ttk::UnaryBcastTileOp>(computeOp);
        Value outputCB =
            resolveOutputCB(computeOp, kBcastOutputCBIndexAttrName);
        assert(outputCB && "output CB required for unary_bcast_init");
        ttk::UnaryBcastInitOp::create(b, l, bcastOp.getInCb(), outputCB,
                                      bcastOp.getBcastTypeAttr());
      }};

  map[mlir::TypeID::get<ttk::ReduceTileOp>()] = {[](OpBuilder &b, Location l,
                                                    Operation *computeOp) {
    auto reduceOp = cast<ttk::ReduceTileOp>(computeOp);
    Value outputCB = resolveOutputCB(computeOp, kReduceOutputCBIndexAttrName);
    assert(outputCB && "output CB required for reduce_init");
    auto initOp = ttk::ReduceInitOp::create(
        b, l, reduceOp.getInCb(), reduceOp.getScalingCb(), outputCB,
        reduceOp.getReduceTypeAttr(), reduceOp.getReduceDimAttr());
    if (reduceOp.getFullFp32()) {
      initOp.setFullFp32(true);
    }
  }};

  map[mlir::TypeID::get<ttk::FillTileOp>()] = {
      [](OpBuilder &b, Location l, Operation *) {
        ttk::FillTileInitOp::create(b, l);
      }};

  // TypecastTile: init takes the same in_dtype and out_dtype attributes
  // as the compute op so the SFPU is configured for the correct
  // source/destination data formats.
  map[mlir::TypeID::get<ttk::TypecastTileOp>()] = {
      [](OpBuilder &b, Location l, Operation *computeOp) {
        auto typecastOp = cast<ttk::TypecastTileOp>(computeOp);
        ttk::TypecastTileInitOp::create(b, l, typecastOp.getInDtypeAttr(),
                                        typecastOp.getOutDtypeAttr());
      }};

  // Transpose: resolves output CB from annotated attribute.
  map[mlir::TypeID::get<ttk::TransposeTileOp>()] = {
      [](OpBuilder &b, Location l, Operation *computeOp) {
        auto transposeOp = cast<ttk::TransposeTileOp>(computeOp);
        Value outputCB =
            resolveOutputCB(computeOp, kTransposeOutputCBIndexAttrName);
        assert(outputCB && "output CB required for transpose_wh_init");
        ttk::TransposeInitOp::create(b, l, transposeOp.getIcb(), outputCB);
      }};

  return map;
}

/// Init key: consecutive ops with the same key share a single init call.
/// The key captures everything that an init op configures in hardware:
/// the op type (MATH pipeline selection), CB operands (UNPACK source routing),
/// and discriminator (e.g., bcast type). When the key is unchanged, the
/// hardware is already configured correctly and re-init can be skipped.
struct InitKey {
  mlir::TypeID typeId;
  llvm::SmallVector<Value, 2> operands;
  int64_t discriminator = 0; // for attribute differences (e.g., bcast type)

  bool operator==(const InitKey &other) const {
    return typeId == other.typeId && operands == other.operands &&
           discriminator == other.discriminator;
  }
  bool operator!=(const InitKey &other) const { return !(*this == other); }
};

static InitKey computeInitKey(Operation *op) {
  mlir::TypeID typeId = op->getName().getTypeID();

  if (isa<ttk::AddTilesOp, ttk::SubTilesOp, ttk::MulTilesOp>(op)) {
    return {typeId, {op->getOperand(0), op->getOperand(1)}};
  }

  if (isa<ttk::MatmulBlockOp>(op)) {
    return {typeId, {op->getOperand(0), op->getOperand(1)}};
  }

  if (isa<ttk::CopyTileOp>(op)) {
    return {typeId, {op->getOperand(0)}};
  }

  if (auto bcast = dyn_cast<ttk::UnaryBcastTileOp>(op)) {
    return {
        typeId, {bcast.getInCb()}, static_cast<int64_t>(bcast.getBcastType())};
  }

  // Different full_fp32 modes select a different LLK kernel branch and must
  // not share an init.
  if (auto reduce = dyn_cast<ttk::ReduceTileOp>(op)) {
    int64_t disc = (static_cast<int64_t>(reduce.getReduceType()) << 16) |
                   (static_cast<int64_t>(reduce.getReduceDim()) << 8) |
                   static_cast<int64_t>(reduce.getFullFp32());
    return {typeId, {reduce.getInCb(), reduce.getScalingCb()}, disc};
  }

  if (auto transpose = dyn_cast<ttk::TransposeTileOp>(op)) {
    return {typeId, {transpose.getIcb()}};
  }

  // For TypecastTile: key includes in_dtype and out_dtype because the init
  // op configures the SFPU per dtype pair. Distinct dtype combinations must
  // not share an init.
  if (auto typecast = dyn_cast<ttk::TypecastTileOp>(op)) {
    int64_t disc = (static_cast<int64_t>(typecast.getInDtype()) << 16) |
                   static_cast<int64_t>(typecast.getOutDtype());
    return {typeId, {}, disc};
  }

  // For all other ops (SFPU unary/binary, CopyDst): key is just the TypeID.
  return {typeId, {}};
}

/// Check if an operation is a sync boundary that resets init tracking.
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
      }
      // Collect output CB from pack ops (both single-tile and block variants).
      auto collectOutputCB = [&](Value packCB, Operation *packOp) {
        if (!outputCB) {
          outputCB = packCB;
        } else if (outputCB != packCB &&
                   outputCB.getType() != packCB.getType()) {
          packOp->emitOpError(
              "sync region packs to output CBs with different data formats; "
              "common init cannot configure multiple PACK formats");
          hadError = true;
        }
      };
      if (auto pack = dyn_cast<ttk::PackTileOp>(inner)) {
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
// Pass implementation
//===----------------------------------------------------------------------===//

struct TTKernelInsertInitsPass
    : public impl::TTKernelInsertInitsBase<TTKernelInsertInitsPass> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
    constexpr llvm::StringLiteral kInitInserted("ttk.init_inserted");

    if (failed(insertCommonInits(moduleOp))) {
      signalPassFailure();
      return;
    }

    auto computeToInit = buildComputeToInitMap();

    auto emitReduceUninit = [](OpBuilder &builder, Location loc,
                               ttk::ReduceTileOp prevReduce) {
      auto uninit = ttk::ReduceUninitOp::create(builder, loc);
      if (prevReduce && prevReduce.getFullFp32()) {
        uninit.setFullFp32(true);
      }
    };

    auto processOp = [&](Operation &topOp, std::optional<InitKey> &prevKey,
                         ttk::ReduceTileOp &prevReduce) {
      if (isSyncBoundary(&topOp)) {
        if (prevKey &&
            prevKey->typeId == mlir::TypeID::get<ttk::ReduceTileOp>()) {
          OpBuilder builder(&topOp);
          emitReduceUninit(builder, topOp.getLoc(), prevReduce);
        }
        prevKey = std::nullopt;
        prevReduce = nullptr;
        return;
      }

      topOp.walk([&](Operation *inner) {
        auto mapIt = computeToInit.find(inner->getName().getTypeID());
        if (mapIt == computeToInit.end()) {
          return WalkResult::advance();
        }
        InitKey key = computeInitKey(inner);
        if (!prevKey || *prevKey != key) {
          if (prevKey &&
              prevKey->typeId == mlir::TypeID::get<ttk::ReduceTileOp>() &&
              key.typeId != mlir::TypeID::get<ttk::ReduceTileOp>()) {
            OpBuilder builder(&topOp);
            emitReduceUninit(builder, topOp.getLoc(), prevReduce);
          }
          OpBuilder builder(&topOp);
          mapIt->second.createInit(builder, inner->getLoc(), inner);
        }
        prevKey = key;
        prevReduce = dyn_cast<ttk::ReduceTileOp>(inner);
        inner->setAttr(kInitInserted, UnitAttr::get(inner->getContext()));
        return WalkResult::interrupt();
      });
    };

    moduleOp->walk([&](ttk::TileRegsAcquireOp acquireOp) {
      Block *block = acquireOp->getBlock();
      std::optional<InitKey> prevKey;
      ttk::ReduceTileOp prevReduce;
      for (auto it = std::next(acquireOp->getIterator()); it != block->end();
           ++it) {
        if (isa<ttk::TileRegsReleaseOp>(&*it)) {
          break;
        }
        processOp(*it, prevKey, prevReduce);
      }
    });

    moduleOp->walk([&](Operation *op) { op->removeAttr(kInitInserted); });
  }
};

} // namespace

} // namespace mlir::tt::ttl
