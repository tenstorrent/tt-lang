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
//      placed immediately before the compute op. A loop whose every
//      iteration computes with one key and loop-invariant operands gets
//      that init once, before the loop.
//   3. SrcA format reconfiguration: copy_tile_init does not change the
//      unpack data format, and the short inits assert that it already
//      matches. A copy, add, sub, mul, dest-reuse, reduce, or matmul short
//      init whose required SrcA element type differs gets
//      reconfig_data_format_srca immediately before it. The format persists
//      across sync regions, so the dataflow runs over the whole function and
//      reads the inits placed by phases 1 and 2. Loop backedges and branch
//      joins merge the programmed operand per edge.
//
// Phases 2 and 3 summarize each region once (see Transfer), so their cost is
// linear in the number of operations at any loop nesting depth.
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
#include "mlir/IR/Matchers.h"
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
  // Non-CB init operands. Constants compare by value because lowering
  // materializes one constant per op and this pass runs before CSE.
  llvm::SmallVector<OpFoldResult, 4> parameters = {};

  bool operator==(const InitKey &other) const {
    return typeId == other.typeId && operands == other.operands &&
           discriminator == other.discriminator &&
           parameters == other.parameters;
  }
  bool operator!=(const InitKey &other) const { return !(*this == other); }
};

// mm_block_init_short forwards these to the unpack and math inits.
static llvm::SmallVector<Value, 4> initParameterValues(Operation *op) {
  if (auto matmul = dyn_cast<ttk::MatmulBlockOp>(op)) {
    return {matmul.getTranspose(), matmul.getCtDim(), matmul.getRtDim(),
            matmul.getKtDim()};
  }
  return {};
}

static OpFoldResult initParameter(Value value) {
  Attribute constant;
  if (matchPattern(value, m_Constant(&constant))) {
    return constant;
  }
  return value;
}

static InitKey computeInitKey(Operation *op) {
  mlir::TypeID typeId = op->getName().getTypeID();

  if (isa<ttk::AddTilesOp, ttk::SubTilesOp, ttk::MulTilesOp>(op)) {
    return {typeId, {op->getOperand(0), op->getOperand(1)}};
  }

  if (auto matmul = dyn_cast<ttk::MatmulBlockOp>(op)) {
    InitKey key{typeId, {matmul.getIn0CbId(), matmul.getIn1CbId()}};
    for (Value parameter : initParameterValues(op)) {
      key.parameters.push_back(initParameter(parameter));
    }
    return key;
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
// Region transfer summaries
//===----------------------------------------------------------------------===//

static bool alwaysEntersBody(scf::ForOp forOp) {
  std::optional<llvm::APInt> tripCount = forOp.getStaticTripCount();
  return tripCount && !tripCount->isZero();
}

/// Transfer function of a region in a forward analysis where every operation
/// either keeps the state or replaces it with a constant. These functions are
/// closed under sequencing and join, so each one is the identity, `s -> c`, or
/// `s -> join(s, c)`, and the loop fixed point `x = join(s, f(x))` is
/// `join(s, f(s))`. Summarizing each region once and then visiting each
/// operation once keeps an analysis linear in the number of operations at any
/// loop depth. `joinStates` must be associative, commutative, and idempotent.
template <typename State>
struct Transfer {
  enum class Kind { Identity, Assign, Join };

  Kind kind = Kind::Identity;
  State value{};

  static Transfer assign(State state) {
    return {Kind::Assign, std::move(state)};
  }

  State apply(const State &state) const {
    switch (kind) {
    case Kind::Identity:
      return state;
    case Kind::Assign:
      return value;
    case Kind::Join:
      return joinStates(state, value);
    }
    llvm_unreachable("unknown transfer kind");
  }

  /// Returns the transfer that runs this one and then `next`.
  Transfer then(const Transfer &next) const {
    if (next.kind == Kind::Identity) {
      return *this;
    }
    if (next.kind == Kind::Assign || kind == Kind::Identity) {
      return next;
    }
    return {kind, joinStates(value, next.value)};
  }

  /// Returns the pointwise join, the transfer of a control-flow merge.
  Transfer join(const Transfer &other) const {
    if (other.kind == Kind::Identity) {
      return kind == Kind::Identity ? *this : Transfer{Kind::Join, value};
    }
    if (kind == Kind::Identity) {
      return {Kind::Join, other.value};
    }
    Kind joined = kind == Kind::Assign && other.kind == Kind::Assign
                      ? Kind::Assign
                      : Kind::Join;
    return {joined, joinStates(value, other.value)};
  }
};

/// Transfer from the entry of a loop to the entry of its body.
template <typename State>
static Transfer<State> loopEntryTransfer(const Transfer<State> &body) {
  return Transfer<State>().join(body);
}

template <typename State>
static Transfer<State> forTransfer(const Transfer<State> &body,
                                   bool entersBody) {
  Transfer<State> exit = loopEntryTransfer(body).then(body);
  return entersBody ? exit : Transfer<State>().join(exit);
}

// The before region runs on every exit, including the failing condition.
template <typename State>
static Transfer<State> whileTransfer(const Transfer<State> &before,
                                     const Transfer<State> &after) {
  return loopEntryTransfer(before.then(after)).then(before);
}

//===----------------------------------------------------------------------===//
// SrcA data-format reconfiguration
//===----------------------------------------------------------------------===//

static Type unpackElementType(Value cb) {
  return cast<ttk::CBType>(cb.getType()).getElementType();
}

// Full configures replace the programmed SrcA operand. Full matmul init uses
// SrcOrder::Reverse, so SrcA is in1. mm_block_init_short, copy_tile_init,
// binary_dest_reuse_tiles_init, add/sub/mul_tiles_init, and reduce_init do
// not write the format. The tiles inits assert the unpacker format already
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

// Short inits assert that unpacker A already has this operand's format, so
// they are the SrcA consumers. Phase 2 places a compute op's init before it
// or before its loop with no SrcA write in between.
// add/sub/mul take it from in0. Dest reuse unpacks its one CB through SrcA.
// Row sum and row average swap the scaler into SrcA. Other reduces use the
// data CB. Matmul's short init keeps SrcA on in1.
static Value srcARequiredBy(Operation *op) {
  if (auto init = dyn_cast<ttk::CopyTileInitOp>(op)) {
    return init.getCb0();
  }
  if (auto init = dyn_cast<ttk::AddTilesInitOp>(op)) {
    return init.getIn0Cb();
  }
  if (auto init = dyn_cast<ttk::SubTilesInitOp>(op)) {
    return init.getIn0Cb();
  }
  if (auto init = dyn_cast<ttk::MulTilesInitOp>(op)) {
    return init.getIn0Cb();
  }
  if (auto init = dyn_cast<ttk::BinaryDestReuseTilesInitOp>(op)) {
    return init.getInCb();
  }
  if (auto init = dyn_cast<ttk::ReduceInitOp>(op)) {
    bool scalerIsSrcA = init.getReduceDim() == ttk::ReduceDim::Row &&
                        init.getReduceType() != ttk::ReduceType::Max;
    return scalerIsSrcA ? init.getScalingCb() : init.getInCb();
  }
  if (auto init = dyn_cast<ttk::MatmulBlockInitShortOp>(op)) {
    return init.getIn1Cb();
  }
  return Value();
}

// A null CB means the programmed SrcA operand is not unique. The one-operand
// reconfig always applies; a guessed old operand can skip a required one.
struct SrcAState {
  Value cb;
};

static SrcAState joinStates(SrcAState lhs, SrcAState rhs) {
  if (lhs.cb && rhs.cb &&
      unpackElementType(lhs.cb) == unpackElementType(rhs.cb)) {
    return lhs;
  }
  return {};
}

using SrcATransfer = Transfer<SrcAState>;

struct SrcAReconfig {
  Operation *init;
  Value oldCB;
  Value newCB;
};

/// Plans reconfig_data_format_srca before every short init whose required
/// SrcA element type differs from the programmed one.
class SrcAReconfigPlanner {
public:
  SrcAState plan(Region &region, SrcAState entry);
  ArrayRef<SrcAReconfig> getReconfigs() const { return reconfigs; }

private:
  SrcATransfer summarize(Region &region);
  SrcATransfer summarize(Operation *op);
  SrcAState plan(Operation *op, SrcAState state);

  llvm::DenseMap<Region *, SrcATransfer> summaries;
  llvm::SmallVector<SrcAReconfig> reconfigs;
};

SrcATransfer SrcAReconfigPlanner::summarize(Region &region) {
  if (region.empty()) {
    return {};
  }
  auto it = summaries.find(&region);
  if (it != summaries.end()) {
    return it->second;
  }
  SrcATransfer transfer = SrcATransfer::assign({});
  if (region.hasOneBlock()) {
    transfer = {};
    for (Operation &op : region.front().without_terminator()) {
      transfer = transfer.then(summarize(&op));
    }
  }
  summaries[&region] = transfer;
  return transfer;
}

SrcATransfer SrcAReconfigPlanner::summarize(Operation *op) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    return forTransfer(summarize(forOp.getRegion()), alwaysEntersBody(forOp));
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    return summarize(ifOp.getThenRegion())
        .join(summarize(ifOp.getElseRegion()));
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    return whileTransfer(summarize(whileOp.getBefore()),
                         summarize(whileOp.getAfter()));
  }
  if (op->getNumRegions() != 0) {
    return SrcATransfer::assign({});
  }
  if (Value cb = srcARequiredBy(op)) {
    return SrcATransfer::assign({cb});
  }
  if (Value cb = srcAProgrammedBy(op)) {
    return SrcATransfer::assign({cb});
  }
  return {};
}

SrcAState SrcAReconfigPlanner::plan(Region &region, SrcAState entry) {
  if (region.empty()) {
    return entry;
  }
  if (!region.hasOneBlock()) {
    region.walk([&](Operation *op) {
      if (Value cb = srcARequiredBy(op)) {
        reconfigs.push_back({op, Value(), cb});
      }
    });
    return {};
  }
  SrcAState state = entry;
  for (Operation &op : region.front().without_terminator()) {
    state = plan(&op, state);
  }
  return state;
}

SrcAState SrcAReconfigPlanner::plan(Operation *op, SrcAState state) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    Region &body = forOp.getRegion();
    SrcAState exit =
        plan(body, loopEntryTransfer(summarize(body)).apply(state));
    return alwaysEntersBody(forOp) ? exit : joinStates(state, exit);
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    SrcAState thenExit = plan(ifOp.getThenRegion(), state);
    return joinStates(thenExit, plan(ifOp.getElseRegion(), state));
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    SrcATransfer iteration =
        summarize(whileOp.getBefore()).then(summarize(whileOp.getAfter()));
    SrcAState beforeExit =
        plan(whileOp.getBefore(), loopEntryTransfer(iteration).apply(state));
    plan(whileOp.getAfter(), beforeExit);
    return beforeExit;
  }
  if (op->getNumRegions() != 0) {
    for (Region &region : op->getRegions()) {
      plan(region, {});
    }
    return {};
  }
  if (Value cb = srcARequiredBy(op)) {
    if (!state.cb || unpackElementType(state.cb) != unpackElementType(cb)) {
      reconfigs.push_back({op, state.cb, cb});
    }
    return {cb};
  }
  if (Value cb = srcAProgrammedBy(op)) {
    return {cb};
  }
  return state;
}

/// Insert reconfig_data_format_srca before a short init whose required SrcA
/// element type differs from the programmed operand.
/// The SrcA format persists across sync regions, and a common init hoisted
/// above a compiler loop runs once for every region in that loop, so the
/// analysis covers the whole function and reads the inits present in the IR.
static void insertSrcAReconfigs(ModuleOp moduleOp) {
  moduleOp->walk([&](func::FuncOp funcOp) {
    SrcAReconfigPlanner planner;
    planner.plan(funcOp.getBody(), {});

    OpBuilder builder(funcOp.getContext());
    for (const SrcAReconfig &reconfig : planner.getReconfigs()) {
      builder.setInsertionPoint(reconfig.init);
      ttk::ReconfigDataFormatSrcaOp::create(builder, reconfig.init->getLoc(),
                                            reconfig.oldCB, reconfig.newCB);
    }
  });
}

//===----------------------------------------------------------------------===//
// Per-op init insertion
//===----------------------------------------------------------------------===//

using ComputeInitMap = llvm::DenseMap<mlir::TypeID, InitOpInfo>;

// A null key means the programmed init is not unique. The next compute must
// be initialized again; reusing either predecessor can skip a required init.
// mayReduce stays set when any incoming path can still have the packer edge
// mask from reduce_init. Merging keys must not drop that fact.
struct InitState {
  std::optional<InitKey> key;
  bool mayReduce = false;
};

static InitState joinStates(const InitState &lhs, const InitState &rhs) {
  InitState result;
  result.mayReduce = lhs.mayReduce || rhs.mayReduce;
  if (lhs.key && rhs.key && *lhs.key == *rhs.key) {
    result.key = lhs.key;
  }
  return result;
}

using InitTransfer = Transfer<InitState>;

static InitState stateAfter(Operation *compute) {
  InitKey key = computeInitKey(compute);
  bool isReduce = key.typeId == mlir::TypeID::get<ttk::ReduceTileOp>();
  return {key, isReduce};
}

static bool initValuesDefinedOutside(scf::ForOp forOp, Operation *compute) {
  InitKey key = computeInitKey(compute);
  for (Value operand : key.operands) {
    if (!forOp.isDefinedOutsideOfLoop(operand)) {
      return false;
    }
  }
  // The hoisted init consumes the original SSA values, not the folded key.
  for (Value parameter : initParameterValues(compute)) {
    if (!forOp.isDefinedOutsideOfLoop(parameter)) {
      return false;
    }
  }
  return true;
}

struct InitSummary {
  InitTransfer transfer;
  /// First compute in the region, or null when it has none.
  Operation *representative = nullptr;
  /// Every compute shares the representative's key and no operation discards
  /// the programmed init.
  bool uniform = true;

  static InitSummary discarding() {
    return {InitTransfer::assign({}), nullptr, false};
  }

  void absorbUniformity(const InitSummary &other) {
    uniform = uniform && other.uniform;
    if (!other.representative) {
      return;
    }
    if (!representative) {
      representative = other.representative;
      return;
    }
    if (computeInitKey(representative) !=
        computeInitKey(other.representative)) {
      uniform = false;
    }
  }
};

/// Inserts per-op inits and reduce_uninit.
class InitInserter {
public:
  explicit InitInserter(const ComputeInitMap &inits) : inits(inits) {}

  InitState insert(Block::iterator begin, Block::iterator end, InitState state);

private:
  bool isCompute(Operation *op) const {
    return inits.contains(op->getName().getTypeID());
  }
  InitSummary summarize(Region &region);
  InitSummary summarize(Operation *op);
  Operation *hoistedCompute(scf::ForOp forOp);
  InitState initialize(Operation *compute, InitState state,
                       Operation *insertBefore);
  InitState insert(Region &region, InitState entry);
  InitState insert(Operation *op, InitState state);

  const ComputeInitMap &inits;
  llvm::DenseMap<Region *, InitSummary> summaries;
};

InitSummary InitInserter::summarize(Region &region) {
  if (region.empty()) {
    return {};
  }
  auto it = summaries.find(&region);
  if (it != summaries.end()) {
    return it->second;
  }
  InitSummary summary = InitSummary::discarding();
  if (region.hasOneBlock()) {
    summary = {};
    for (Operation &op : region.front().without_terminator()) {
      InitSummary next = summarize(&op);
      summary.transfer = summary.transfer.then(next.transfer);
      summary.absorbUniformity(next);
    }
  }
  summaries[&region] = summary;
  return summary;
}

InitSummary InitInserter::summarize(Operation *op) {
  if (isSyncBoundary(op)) {
    return InitSummary::discarding();
  }
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    InitSummary summary = summarize(forOp.getRegion());
    if (Operation *compute = hoistedCompute(forOp)) {
      summary.transfer = InitTransfer::assign(stateAfter(compute));
    } else {
      summary.transfer = forTransfer(summary.transfer, alwaysEntersBody(forOp));
    }
    return summary;
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    InitSummary summary = summarize(ifOp.getThenRegion());
    InitSummary other = summarize(ifOp.getElseRegion());
    summary.transfer = summary.transfer.join(other.transfer);
    summary.absorbUniformity(other);
    return summary;
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    InitSummary summary = summarize(whileOp.getBefore());
    InitSummary after = summarize(whileOp.getAfter());
    summary.transfer = whileTransfer(summary.transfer, after.transfer);
    summary.absorbUniformity(after);
    return summary;
  }
  if (op->getNumRegions() != 0) {
    return InitSummary::discarding();
  }
  if (!isCompute(op)) {
    return {};
  }
  return {InitTransfer::assign(stateAfter(op)), op, true};
}

// A loop whose every iteration computes with one key and loop-invariant init
// operands gets that init once, before the loop. The init runs whether or not
// the body does, and the SrcA phase places any required reconfig before it.
Operation *InitInserter::hoistedCompute(scf::ForOp forOp) {
  InitSummary body = summarize(forOp.getRegion());
  bool everyIteration = body.transfer.kind == InitTransfer::Kind::Assign;
  if (!body.uniform || !body.representative || !everyIteration ||
      !initValuesDefinedOutside(forOp, body.representative)) {
    return nullptr;
  }
  return body.representative;
}

InitState InitInserter::initialize(Operation *compute, InitState state,
                                   Operation *insertBefore) {
  InitState next = stateAfter(compute);
  OpBuilder builder(insertBefore);
  if (state.mayReduce && !next.mayReduce) {
    ttk::ReduceUninitOp::create(builder, insertBefore->getLoc());
  }
  if (!state.key || *state.key != *next.key) {
    inits.find(compute->getName().getTypeID())
        ->second.createInit(builder, compute->getLoc(), compute);
  }
  return next;
}

InitState InitInserter::insert(Block::iterator begin, Block::iterator end,
                               InitState state) {
  for (Operation &op : llvm::make_range(begin, end)) {
    state = insert(&op, state);
  }
  return state;
}

InitState InitInserter::insert(Region &region, InitState entry) {
  if (region.empty()) {
    return entry;
  }
  if (!region.hasOneBlock()) {
    region.walk([&](Operation *op) {
      if (isCompute(op)) {
        initialize(op, {}, op);
      }
    });
    return {};
  }
  Block &block = region.front();
  return insert(block.begin(), block.end(), entry);
}

InitState InitInserter::insert(Operation *op, InitState state) {
  if (isSyncBoundary(op)) {
    if (state.mayReduce) {
      OpBuilder builder(op);
      ttk::ReduceUninitOp::create(builder, op->getLoc());
    }
    return {};
  }
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    Region &body = forOp.getRegion();
    if (Operation *compute = hoistedCompute(forOp)) {
      InitState programmed = initialize(compute, state, forOp);
      insert(body, programmed);
      return programmed;
    }
    InitState exit =
        insert(body, loopEntryTransfer(summarize(body).transfer).apply(state));
    return alwaysEntersBody(forOp) ? exit : joinStates(state, exit);
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    InitState thenExit = insert(ifOp.getThenRegion(), state);
    return joinStates(thenExit, insert(ifOp.getElseRegion(), state));
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
    InitTransfer before = summarize(whileOp.getBefore()).transfer;
    InitTransfer after = summarize(whileOp.getAfter()).transfer;
    InitState beforeExit =
        insert(whileOp.getBefore(),
               loopEntryTransfer(before.then(after)).apply(state));
    insert(whileOp.getAfter(), beforeExit);
    return beforeExit;
  }
  if (op->getNumRegions() != 0) {
    for (Region &region : op->getRegions()) {
      insert(region, {});
    }
    return {};
  }
  if (!isCompute(op)) {
    return state;
  }
  return initialize(op, state, op);
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

    auto computeToInit = buildComputeToInitMap();
    InitInserter inserter(computeToInit);

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
      inserter.insert(begin, end, {});
    });

    insertSrcAReconfigs(moduleOp);
  }
};

} // namespace

} // namespace mlir::tt::ttl
