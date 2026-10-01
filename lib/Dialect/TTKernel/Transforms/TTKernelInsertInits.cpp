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
//   2. Per-op inits: emitted in linear block order whenever the op type
//      changes (unary SFPU, binary SFPU, minmax, FPU binary). The init
//      key is (init op TypeID, operand values). An init is inserted only
//      when the key changes. Tracking resets at sync boundaries.
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
// copy_tile_init, and binary_dest_reuse_tiles_init do not.
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

// Keep reconfig immediately before copy_tile_init. That init is the previous
// op when phase 2 inserted it beside the copy.
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
                               ttk::ReduceTileOp) {
      ttk::ReduceUninitOp::create(builder, loc);
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

    if (failed(insertSrcAReconfigs(moduleOp))) {
      signalPassFailure();
      return;
    }

    moduleOp->walk([&](Operation *op) { op->removeAttr(kInitInserted); });
  }
};

} // namespace

} // namespace mlir::tt::ttl
