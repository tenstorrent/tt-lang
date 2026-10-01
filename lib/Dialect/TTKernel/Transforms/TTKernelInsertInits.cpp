// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//
// Implementation of the TTKernelInsertInits pass, which inserts both common
// inits (init_sfpu, binary_op_init_common) that configure UNPACK + PACK data
// format routing, and per-op inits (exp_tile_init, add_tiles_init, etc.) that
// configure the MATH pipeline.
//
// Three phases:
//   1. Common inits: one per sync region, hoisted above enclosing loops.
//      Scans each tile_regs_acquire -> tile_regs_release region to determine
//      the compute category (FPU binary vs SFPU/copy/bcast) and derives
//      input/output CBs from compute and pack ops. This programs the SrcA
//      data format once.
//   2. Per-op inits: emitted when the init key changes. The key is the init
//      op type plus the operands it programs. Tracking follows scf.for,
//      scf.if, and scf.while, and resets at sync boundaries. The init is
//      placed immediately before the compute op. A loop that keeps one
//      non-copy key gets that init once, before the loop. copy_tile_init
//      stays next to the copy so a SrcA reconfig can precede it.
//   3. SrcA format reconfiguration: copy_tile_init does not change the
//      unpack data format. A copy from a CB whose element type differs from
//      the programmed SrcA operand gets reconfig_data_format_srca. Loop
//      backedges and branch joins merge that operand per edge.
//
// TODO(#329): Emit init_short variants for cheaper re-inits on type switches.
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelTraits.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/SmallVector.h"

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

  map[mlir::TypeID::get<ttk::AddIntTileOp>()] = {
      [](OpBuilder &builder, Location location, Operation *) {
        ttk::AddIntTileInitOp::create(builder, location);
      }};
  map[mlir::TypeID::get<ttk::SubIntTileOp>()] = {
      [](OpBuilder &builder, Location location, Operation *) {
        ttk::SubIntTileInitOp::create(builder, location);
      }};
  map[mlir::TypeID::get<ttk::MulIntTileOp>()] = {
      [](OpBuilder &builder, Location location, Operation *computeOp) {
        auto multiply = cast<ttk::MulIntTileOp>(computeOp);
        ttk::MulIntTileInitOp::create(builder, location,
                                      multiply.getDtypeAttr());
      }};

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

  // copy_block_matmul_partials uses the same unpack-A init as copy_tile.
  map[mlir::TypeID::get<ttk::CopyBlockMatmulPartialsOp>()] = {
      [](OpBuilder &b, Location l, Operation *computeOp) {
        ttk::CopyTileInitOp::create(b, l, computeOp->getOperand(0));
      }};

  // Destination reuse supplies one binary operand from DST, so the init needs
  // the dataflow buffer operand and the exact elementwise/reuse modes.
  map[mlir::TypeID::get<ttk::BinaryDestReuseTilesOp>()] = {
      [](OpBuilder &b, Location l, Operation *computeOp) {
        auto binaryDestReuseOp = cast<ttk::BinaryDestReuseTilesOp>(computeOp);
        ttk::BinaryDestReuseTilesInitOp::create(
            b, l, binaryDestReuseOp.getInCb(),
            binaryDestReuseOp.getEltwiseBinaryTypeAttr(),
            binaryDestReuseOp.getReuseTypeAttr());
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
    ttk::ReduceInitOp::create(b, l, reduceOp.getInCb(), reduceOp.getScalingCb(),
                              outputCB, reduceOp.getReduceTypeAttr(),
                              reduceOp.getReduceDimAttr());
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

  // ExpTile: exp_tile_init configures the SFPU per exp flags. It takes approx,
  // scale (the fp32 scale factor template), and input_clamping read off the
  // exp_tile op. scale_en / iterations are compute-only and not part of the
  // init. (exp is excluded from the generic unary macro above for this reason.)
  map[mlir::TypeID::get<ttk::ExpTileOp>()] = {
      [](OpBuilder &b, Location l, Operation *computeOp) {
        auto expOp = cast<ttk::ExpTileOp>(computeOp);
        ttk::ExpTileInitOp::create(b, l, expOp.getApproxAttr(),
                                   expOp.getScaleAttr(),
                                   expOp.getInputClampingAttr());
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

  if (isa<ttk::CopyTileOp, ttk::CopyBlockMatmulPartialsOp>(op)) {
    return {typeId, {op->getOperand(0)}};
  }

  if (auto binaryDestReuseOp = dyn_cast<ttk::BinaryDestReuseTilesOp>(op)) {
    // The destination operand is not a dataflow buffer and cannot distinguish
    // the LLK init. The binary type and reuse mode select the kernel variant.
    int64_t discriminator =
        (static_cast<int64_t>(binaryDestReuseOp.getEltwiseBinaryType()) << 8) |
        static_cast<int64_t>(binaryDestReuseOp.getReuseType());
    return {typeId, {binaryDestReuseOp.getInCb()}, discriminator};
  }

  if (auto bcast = dyn_cast<ttk::UnaryBcastTileOp>(op)) {
    return {
        typeId, {bcast.getInCb()}, static_cast<int64_t>(bcast.getBcastType())};
  }

  if (auto reduce = dyn_cast<ttk::ReduceTileOp>(op)) {
    int64_t disc = (static_cast<int64_t>(reduce.getReduceType()) << 16) |
                   static_cast<int64_t>(reduce.getReduceDim());
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

  if (auto multiply = dyn_cast<ttk::MulIntTileOp>(op)) {
    return {typeId, {}, static_cast<int64_t>(multiply.getDtype())};
  }

  // For exp: distinct flag combinations configure exp_tile_init differently
  // and must not share an init. The init depends on approx, input_clamping,
  // and the fp32 scale template, so encode all three in the discriminator.
  // scale_en / iterations are compute-only and do not affect the init.
  if (auto exp = dyn_cast<ttk::ExpTileOp>(op)) {
    uint32_t scaleBits = 0x3F800000u; // default 1.0f for exp_tile_init.
    if (auto scaleAttr = exp.getScaleAttr()) {
      scaleBits = static_cast<uint32_t>(scaleAttr.getInt());
    }
    BoolAttr approxAttr = exp.getApproxAttr();
    bool approx = approxAttr && approxAttr.getValue();
    int64_t inputClamping =
        static_cast<int64_t>(ttk::InputClamping::ClampToNegative);
    if (auto inputClampingAttr = exp.getInputClampingAttr()) {
      inputClamping = static_cast<int64_t>(inputClampingAttr.getValue());
    }
    int64_t disc = (static_cast<int64_t>(scaleBits) << 8) |
                   (static_cast<int64_t>(approx) << 1) | inputClamping;
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

// mm_block_init_short is used when this matmul loop shares a pack CB with the
// preceding annotated loop. The short init does not program the unpack format.
static bool matmulCommonInitIsShort(Operation *insertBefore) {
  auto forOp = dyn_cast<scf::ForOp>(insertBefore);
  if (!forOp) {
    return false;
  }
  if (!forOp->hasAttr(kL1AccLoopAttrName) &&
      !forOp->hasAttr(kReductionLoopAttrName)) {
    return false;
  }
  for (Operation *prev = forOp->getPrevNode(); prev;
       prev = prev->getPrevNode()) {
    if (auto prevFor = dyn_cast<scf::ForOp>(prev)) {
      return (prevFor->hasAttr(kL1AccLoopAttrName) ||
              prevFor->hasAttr(kReductionLoopAttrName)) &&
             sharePackCB(prevFor, forOp);
    }
  }
  return false;
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
    bool useInitShort =
        analysis.hasMatmul && matmulCommonInitIsShort(insertBefore);

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
// SrcA data-format reconfiguration
//===----------------------------------------------------------------------===//

static Type unpackElementType(Value cb) {
  return cast<ttk::CBType>(cb.getType()).getElementType();
}

// The common init is the operand whose format hardware actually programs.
// Full matmul init uses SrcOrder::Reverse, so SrcA is in1 and SrcB is in0.
// FPU binary init programs SrcA from in0. mm_block_init_short programs neither.
static Value initialSrcA(const SyncRegionAnalysis &analysis, Value inputCB,
                         Value in0CB, Value in1CB, Value outputCB,
                         bool matmulInitIsShort) {
  if (outputCB && analysis.hasMatmul && in0CB && in1CB) {
    return matmulInitIsShort ? Value() : in1CB;
  }
  if (outputCB && analysis.hasFPUBinary && in0CB && in1CB) {
    return in0CB;
  }
  if (!inputCB && outputCB) {
    return outputCB;
  }
  return inputCB;
}

// Full configures replace the programmed SrcA operand. mm_block_init_short,
// copy_tile_init, binary_dest_reuse_tiles_init, add/sub/mul_tiles_init, and
// reduce_init do not. The tiles inits assert the unpacker format already
// matches. reduce_init expects a preceding reconfig and does not write it.
static Value srcAProgrammedBy(Operation *op) {
  if (auto init = dyn_cast<ttk::TransposeInitOp>(op)) {
    return init.getCbIn();
  }
  if (auto init = dyn_cast<ttk::UnaryBcastInitOp>(op)) {
    return init.getInCb();
  }
  if (auto init = dyn_cast<ttk::BinaryOpInitCommonOp>(op)) {
    return init.getIn0Cb();
  }
  if (auto init = dyn_cast<ttk::InitSFPUOp>(op)) {
    return init.getIcb();
  }
  if (auto init = dyn_cast<ttk::MatmulInitOp>(op)) {
    return init.getIn1Cb();
  }
  if (auto init = dyn_cast<ttk::MatmulBlockInitOp>(op)) {
    return init.getIn1Cb();
  }
  if (auto init = dyn_cast<ttk::ComputeKernelHWStartupOp>(op)) {
    return init.getIcb0();
  }
  if (auto reconfig = dyn_cast<ttk::ReconfigDataFormatSrcaOp>(op)) {
    return reconfig.getSrcaNew();
  }
  return Value();
}

// Phase 2 places copy_tile_init immediately before the copy that needs it.
// Keep the reconfig in front of that init.
static Operation *reconfigInsertionPoint(Operation *copy, Value srcCB) {
  if (auto init = dyn_cast_or_null<ttk::CopyTileInitOp>(copy->getPrevNode())) {
    if (init.getCb0() == srcCB) {
      return init;
    }
  }
  return copy;
}

struct SrcAReconfig {
  Operation *insertBefore;
  Value oldCB;
  Value newCB;
  Location loc;
};

// A null CB means the programmed SrcA operand is not unique. The one-operand
// reconfig always applies; a guessed old operand can skip a required one.
struct SrcAState {
  Value cb;
};

static bool sameSrcA(SrcAState lhs, SrcAState rhs) {
  if (!lhs.cb || !rhs.cb) {
    return !lhs.cb && !rhs.cb;
  }
  return unpackElementType(lhs.cb) == unpackElementType(rhs.cb);
}

static SrcAState mergeSrcA(SrcAState lhs, SrcAState rhs) {
  return sameSrcA(lhs, rhs) ? lhs : SrcAState{};
}

static SrcAState
recordCopyReconfig(Operation *copy, Value srcCB, SrcAState state, bool record,
                   llvm::SmallVectorImpl<SrcAReconfig> &planned) {
  bool mismatch =
      !state.cb || unpackElementType(state.cb) != unpackElementType(srcCB);
  if (record && mismatch) {
    planned.push_back(
        {reconfigInsertionPoint(copy, srcCB), state.cb, srcCB, copy->getLoc()});
  }
  return {srcCB};
}

static SrcAState processRegion(Region &region, SrcAState entry, bool record,
                               llvm::SmallVectorImpl<SrcAReconfig> &planned);

// The lattice is a known element type or unknown, so a region is simulated a
// constant number of times rather than once per iteration.
static constexpr int kSrcAFixedPointLimit = 4;

static SrcAState processFor(scf::ForOp forOp, SrcAState entry, bool record,
                            llvm::SmallVectorImpl<SrcAReconfig> &planned) {
  SrcAState bodyEntry = entry;
  SrcAState bodyExit = entry;
  bool stable = false;
  for (int i = 0; i < kSrcAFixedPointLimit; ++i) {
    SrcAState nextEntry = mergeSrcA(entry, bodyExit);
    SrcAState nextExit =
        processRegion(forOp.getRegion(), nextEntry, false, planned);
    if (sameSrcA(nextEntry, bodyEntry) && sameSrcA(nextExit, bodyExit)) {
      bodyEntry = nextEntry;
      bodyExit = nextExit;
      stable = true;
      break;
    }
    bodyEntry = nextEntry;
    bodyExit = nextExit;
  }
  if (!stable) {
    bodyEntry = {};
    bodyExit = processRegion(forOp.getRegion(), bodyEntry, false, planned);
  }
  if (record) {
    processRegion(forOp.getRegion(), bodyEntry, true, planned);
  }
  std::optional<llvm::APInt> tripCount = forOp.getStaticTripCount();
  bool mustRun = tripCount && !tripCount->isZero();
  return mustRun ? bodyExit : mergeSrcA(entry, bodyExit);
}

static SrcAState processIf(scf::IfOp ifOp, SrcAState entry, bool record,
                           llvm::SmallVectorImpl<SrcAReconfig> &planned) {
  SrcAState thenExit =
      processRegion(ifOp.getThenRegion(), entry, record, planned);
  SrcAState elseExit =
      ifOp.getElseRegion().empty()
          ? entry
          : processRegion(ifOp.getElseRegion(), entry, record, planned);
  return mergeSrcA(thenExit, elseExit);
}

static SrcAState processWhile(scf::WhileOp whileOp, SrcAState entry,
                              bool record,
                              llvm::SmallVectorImpl<SrcAReconfig> &planned) {
  SrcAState beforeEntry = entry;
  SrcAState beforeExit = entry;
  SrcAState afterExit = entry;
  bool stable = false;
  for (int i = 0; i < kSrcAFixedPointLimit; ++i) {
    SrcAState nextBeforeEntry = mergeSrcA(entry, afterExit);
    SrcAState nextBeforeExit =
        processRegion(whileOp.getBefore(), nextBeforeEntry, false, planned);
    SrcAState nextAfterExit =
        processRegion(whileOp.getAfter(), nextBeforeExit, false, planned);
    if (sameSrcA(nextBeforeEntry, beforeEntry) &&
        sameSrcA(nextBeforeExit, beforeExit) &&
        sameSrcA(nextAfterExit, afterExit)) {
      beforeEntry = nextBeforeEntry;
      beforeExit = nextBeforeExit;
      afterExit = nextAfterExit;
      stable = true;
      break;
    }
    beforeEntry = nextBeforeEntry;
    beforeExit = nextBeforeExit;
    afterExit = nextAfterExit;
  }
  if (!stable) {
    beforeEntry = {};
    beforeExit =
        processRegion(whileOp.getBefore(), beforeEntry, false, planned);
    afterExit = processRegion(whileOp.getAfter(), beforeExit, false, planned);
  }
  if (record) {
    processRegion(whileOp.getBefore(), beforeEntry, true, planned);
    processRegion(whileOp.getAfter(), beforeExit, true, planned);
  }
  // The before region runs on every exit, including the failing condition.
  return beforeExit;
}

static SrcAState
processOperation(Operation *op, SrcAState state, bool record,
                 llvm::SmallVectorImpl<SrcAReconfig> &planned) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    return processFor(forOp, state, record, planned);
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    return processIf(ifOp, state, record, planned);
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    return processWhile(whileOp, state, record, planned);
  }
  if (op->getNumRegions() != 0) {
    for (Region &region : op->getRegions()) {
      processRegion(region, {}, record, planned);
    }
    return {};
  }
  if (auto copy = dyn_cast<ttk::CopyTileOp>(op)) {
    return recordCopyReconfig(copy, copy.getCb0(), state, record, planned);
  }
  if (auto copyBlock = dyn_cast<ttk::CopyBlockMatmulPartialsOp>(op)) {
    return recordCopyReconfig(copyBlock, copyBlock.getCb(), state, record,
                              planned);
  }
  if (Value programmed = srcAProgrammedBy(op)) {
    return {programmed};
  }
  return state;
}

static SrcAState processRegion(Region &region, SrcAState entry, bool record,
                               llvm::SmallVectorImpl<SrcAReconfig> &planned) {
  if (region.empty()) {
    return entry;
  }
  if (!region.hasOneBlock()) {
    if (record) {
      region.walk([&](Operation *op) {
        if (auto copy = dyn_cast<ttk::CopyTileOp>(op)) {
          recordCopyReconfig(copy, copy.getCb0(), {}, true, planned);
        } else if (auto copyBlock =
                       dyn_cast<ttk::CopyBlockMatmulPartialsOp>(op)) {
          recordCopyReconfig(copyBlock, copyBlock.getCb(), {}, true, planned);
        }
      });
    }
    return {};
  }
  SrcAState state = entry;
  for (Operation &op : region.front().without_terminator()) {
    state = processOperation(&op, state, record, planned);
  }
  return state;
}

/// Insert reconfig_data_format_srca before copies whose CB element type
/// differs from the SrcA format programmed for this sync region.
/// copy_tile_init does not program that format.
static LogicalResult insertSrcAReconfigs(ModuleOp moduleOp) {
  bool hadError = false;
  moduleOp->walk([&](ttk::TileRegsAcquireOp acquireOp) {
    Value inputCB, in0CB, in1CB, outputCB;
    FailureOr<SyncRegionAnalysis> analysis =
        analyzeSyncRegion(acquireOp, inputCB, in0CB, in1CB, outputCB);
    if (failed(analysis)) {
      hadError = true;
      return;
    }
    bool matmulInitIsShort =
        analysis->hasMatmul &&
        matmulCommonInitIsShort(hoistAboveCompilerLoops(acquireOp));
    SrcAState state{initialSrcA(*analysis, inputCB, in0CB, in1CB, outputCB,
                                matmulInitIsShort)};
    llvm::SmallVector<SrcAReconfig> planned;
    Block *block = acquireOp->getBlock();
    for (auto it = std::next(acquireOp->getIterator()); it != block->end();
         ++it) {
      if (isa<ttk::TileRegsReleaseOp>(&*it)) {
        break;
      }
      state = processOperation(&*it, state, true, planned);
    }

    OpBuilder builder(acquireOp.getContext());
    for (const SrcAReconfig &item : planned) {
      builder.setInsertionPoint(item.insertBefore);
      ttk::ReconfigDataFormatSrcaOp::create(builder, item.loc, item.oldCB,
                                            item.newCB);
    }
  });
  return hadError ? failure() : success();
}

//===----------------------------------------------------------------------===//
// Per-op init insertion
//===----------------------------------------------------------------------===//

using ComputeInitMap = llvm::DenseMap<mlir::TypeID, InitOpInfo>;

// A null key means the programmed init is not unique. The next compute must
// be initialized again; reusing either predecessor can skip a required init.
struct InitState {
  std::optional<InitKey> key;
};

struct InitSim {
  InitState state;
  bool changed = false;
};

struct InitInsertCtx {
  const ComputeInitMap &inits;
  llvm::StringRef insertedAttr;
};

// The lattice is one key or unknown, so a region is simulated a constant
// number of times rather than once per iteration.
static constexpr int kInitFixedPointLimit = 4;

static InitSim processInitRegion(Region &region, InitState entry, bool record,
                                 const InitInsertCtx &ctx);
static InitSim processInitFor(scf::ForOp forOp, InitState entry, bool record,
                              const InitInsertCtx &ctx);
static InitSim processInitIf(scf::IfOp ifOp, InitState entry, bool record,
                             const InitInsertCtx &ctx);
static InitSim processInitWhile(scf::WhileOp whileOp, InitState entry,
                                bool record, const InitInsertCtx &ctx);

static bool sameInit(InitState lhs, InitState rhs) {
  if (!lhs.key || !rhs.key) {
    return !lhs.key && !rhs.key;
  }
  return *lhs.key == *rhs.key;
}

static InitState mergeInit(InitState lhs, InitState rhs) {
  return sameInit(lhs, rhs) ? lhs : InitState{};
}

static bool isCopyInitKey(const InitKey &key) {
  return key.typeId == mlir::TypeID::get<ttk::CopyTileOp>() ||
         key.typeId == mlir::TypeID::get<ttk::CopyBlockMatmulPartialsOp>();
}

static bool isReduceKey(InitState state) {
  return state.key &&
         state.key->typeId == mlir::TypeID::get<ttk::ReduceTileOp>();
}

static void markInitInserted(Operation *compute, const InitInsertCtx &ctx) {
  compute->setAttr(ctx.insertedAttr, UnitAttr::get(compute->getContext()));
}

static bool initValuesDefinedOutside(scf::ForOp forOp, Operation *compute) {
  InitKey key = computeInitKey(compute);
  for (Value operand : key.operands) {
    if (!forOp.isDefinedOutsideOfLoop(operand)) {
      return false;
    }
  }
  // The short matmul init also reads the dimension operands. Those are not
  // part of the key because they do not select a different kernel variant.
  if (auto matmul = dyn_cast<ttk::MatmulBlockOp>(compute)) {
    Value extras[] = {matmul.getTranspose(), matmul.getCtDim(),
                      matmul.getRtDim(), matmul.getKtDim()};
    for (Value operand : extras) {
      if (!forOp.isDefinedOutsideOfLoop(operand)) {
        return false;
      }
    }
  }
  return true;
}

static Operation *findComputeWithKey(Region &region, const InitKey &key,
                                     const InitInsertCtx &ctx) {
  Operation *found = nullptr;
  region.walk([&](Operation *op) {
    if (ctx.inits.find(op->getName().getTypeID()) == ctx.inits.end()) {
      return WalkResult::advance();
    }
    if (computeInitKey(op) == key) {
      found = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static InitSim processInitSequence(Block::iterator begin, Block::iterator end,
                                   InitState entry, bool record,
                                   const InitInsertCtx &ctx) {
  InitSim sim;
  sim.state = entry;
  for (auto it = begin; it != end; ++it) {
    Operation &op = *it;
    if (isSyncBoundary(&op)) {
      if (isReduceKey(sim.state)) {
        sim.changed = true;
        if (record) {
          OpBuilder builder(&op);
          ttk::ReduceUninitOp::create(builder, op.getLoc());
        }
      }
      sim.state = {};
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      InitSim nested = processInitFor(forOp, sim.state, record, ctx);
      sim.changed |= nested.changed;
      sim.state = nested.state;
      continue;
    }
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      InitSim nested = processInitIf(ifOp, sim.state, record, ctx);
      sim.changed |= nested.changed;
      sim.state = nested.state;
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      InitSim nested = processInitWhile(whileOp, sim.state, record, ctx);
      sim.changed |= nested.changed;
      sim.state = nested.state;
      continue;
    }
    if (op.getNumRegions() != 0) {
      for (Region &region : op.getRegions()) {
        InitSim nested = processInitRegion(region, {}, record, ctx);
        sim.changed |= nested.changed;
      }
      sim.state = {};
      continue;
    }
    auto mapIt = ctx.inits.find(op.getName().getTypeID());
    if (mapIt == ctx.inits.end()) {
      continue;
    }
    InitKey key = computeInitKey(&op);
    bool reduceUninit = isReduceKey(sim.state) &&
                        key.typeId != mlir::TypeID::get<ttk::ReduceTileOp>();
    if (!sim.state.key || *sim.state.key != key) {
      sim.changed = true;
      if (record) {
        OpBuilder builder(&op);
        if (reduceUninit) {
          ttk::ReduceUninitOp::create(builder, op.getLoc());
        }
        mapIt->second.createInit(builder, op.getLoc(), &op);
      }
    }
    if (record) {
      markInitInserted(&op, ctx);
    }
    sim.state.key = key;
  }
  return sim;
}

// copy_tile_init stays in the body. A later SrcA reconfig has to run before
// it, and this phase cannot yet see that reconfig.
static InitSim processInitFor(scf::ForOp forOp, InitState entry, bool record,
                              const InitInsertCtx &ctx) {
  Region &body = forOp.getRegion();
  InitSim cold = processInitRegion(body, {}, false, ctx);
  bool preserves = false;
  if (cold.state.key) {
    InitSim warm = processInitRegion(body, cold.state, false, ctx);
    preserves = !warm.changed && sameInit(warm.state, cold.state);
  }

  std::optional<llvm::APInt> tripCount = forOp.getStaticTripCount();
  bool mustRun = tripCount && !tripCount->isZero();
  auto atExit = [&](InitSim inner) {
    if (!mustRun) {
      inner.state = mergeInit(entry, inner.state);
    }
    return inner;
  };

  if (preserves && !isCopyInitKey(*cold.state.key)) {
    Operation *representative = findComputeWithKey(body, *cold.state.key, ctx);
    if (representative && initValuesDefinedOutside(forOp, representative)) {
      bool needsInit = !sameInit(entry, cold.state);
      if (record && needsInit) {
        auto mapIt = ctx.inits.find(representative->getName().getTypeID());
        OpBuilder builder(forOp);
        if (isReduceKey(entry) && !isReduceKey(cold.state)) {
          ttk::ReduceUninitOp::create(builder, forOp.getLoc());
        }
        mapIt->second.createInit(builder, representative->getLoc(),
                                 representative);
      }
      if (record) {
        processInitRegion(body, cold.state, true, ctx);
      }
      InitSim result;
      result.state = mustRun ? cold.state : mergeInit(entry, cold.state);
      result.changed = needsInit;
      return result;
    }
  }

  if (preserves) {
    InitState bodyEntry = mergeInit(entry, cold.state);
    return atExit(processInitRegion(body, bodyEntry, record, ctx));
  }

  InitState bodyEntry = entry;
  InitState bodyExit = entry;
  bool stable = false;
  for (int i = 0; i < kInitFixedPointLimit; ++i) {
    InitState nextEntry = mergeInit(entry, bodyExit);
    InitSim next = processInitRegion(body, nextEntry, false, ctx);
    if (sameInit(nextEntry, bodyEntry) && sameInit(next.state, bodyExit)) {
      bodyEntry = nextEntry;
      bodyExit = next.state;
      stable = true;
      break;
    }
    bodyEntry = nextEntry;
    bodyExit = next.state;
  }
  if (!stable) {
    bodyEntry = {};
    bodyExit = processInitRegion(body, bodyEntry, false, ctx).state;
  }
  InitSim inner = processInitRegion(body, bodyEntry, record, ctx);
  inner.state = mustRun ? bodyExit : mergeInit(entry, bodyExit);
  return inner;
}

static InitSim processInitIf(scf::IfOp ifOp, InitState entry, bool record,
                             const InitInsertCtx &ctx) {
  InitSim thenSim = processInitRegion(ifOp.getThenRegion(), entry, record, ctx);
  InitSim elseSim =
      ifOp.getElseRegion().empty()
          ? InitSim{entry, false}
          : processInitRegion(ifOp.getElseRegion(), entry, record, ctx);
  InitSim result;
  result.state = mergeInit(thenSim.state, elseSim.state);
  result.changed = thenSim.changed || elseSim.changed;
  return result;
}

static InitSim processInitWhile(scf::WhileOp whileOp, InitState entry,
                                bool record, const InitInsertCtx &ctx) {
  InitState beforeEntry = entry;
  InitState beforeExit = entry;
  InitState afterExit = entry;
  bool stable = false;
  for (int i = 0; i < kInitFixedPointLimit; ++i) {
    InitState nextBeforeEntry = mergeInit(entry, afterExit);
    InitSim nextBefore =
        processInitRegion(whileOp.getBefore(), nextBeforeEntry, false, ctx);
    InitSim nextAfter =
        processInitRegion(whileOp.getAfter(), nextBefore.state, false, ctx);
    if (sameInit(nextBeforeEntry, beforeEntry) &&
        sameInit(nextBefore.state, beforeExit) &&
        sameInit(nextAfter.state, afterExit)) {
      beforeEntry = nextBeforeEntry;
      beforeExit = nextBefore.state;
      afterExit = nextAfter.state;
      stable = true;
      break;
    }
    beforeEntry = nextBeforeEntry;
    beforeExit = nextBefore.state;
    afterExit = nextAfter.state;
  }
  if (!stable) {
    beforeEntry = {};
    beforeExit =
        processInitRegion(whileOp.getBefore(), beforeEntry, false, ctx).state;
    afterExit =
        processInitRegion(whileOp.getAfter(), beforeExit, false, ctx).state;
  }
  InitSim beforeSim =
      processInitRegion(whileOp.getBefore(), beforeEntry, record, ctx);
  InitSim afterSim =
      processInitRegion(whileOp.getAfter(), beforeExit, record, ctx);
  // The before region runs on every exit, including the failing condition.
  InitSim result;
  result.state = beforeSim.state;
  result.changed = beforeSim.changed || afterSim.changed;
  return result;
}

static InitSim processInitRegion(Region &region, InitState entry, bool record,
                                 const InitInsertCtx &ctx) {
  if (region.empty()) {
    return {entry, false};
  }
  if (!region.hasOneBlock()) {
    InitSim sim;
    region.walk([&](Operation *op) {
      auto mapIt = ctx.inits.find(op->getName().getTypeID());
      if (mapIt == ctx.inits.end()) {
        return;
      }
      sim.changed = true;
      if (!record) {
        return;
      }
      OpBuilder builder(op);
      mapIt->second.createInit(builder, op->getLoc(), op);
      markInitInserted(op, ctx);
    });
    return sim;
  }
  Block &block = region.front();
  return processInitSequence(block.begin(), block.end(), entry, record, ctx);
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
    InitInsertCtx ctx{computeToInit, kInitInserted};

    moduleOp->walk([&](ttk::TileRegsAcquireOp acquireOp) {
      Block *block = acquireOp->getBlock();
      Block::iterator begin = std::next(acquireOp->getIterator());
      Block::iterator end = block->end();
      for (Block::iterator it = begin; it != block->end(); ++it) {
        if (isa<ttk::TileRegsReleaseOp>(&*it)) {
          end = it;
          break;
        }
      }
      processInitSequence(begin, end, {}, true, ctx);
    });

    if (failed(insertSrcAReconfigs(moduleOp))) {
      signalPassFailure();
      return;
    }

    moduleOp->walk([&](Operation *op) { op->removeAttr(kInitInserted); });
  }
};

} // namespace

} // namespace mlir::tt::ttl
