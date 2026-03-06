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

/// Information about how to create an init op for a given compute op.
struct InitOpInfo {
  /// Creates the init op before the given compute op.
  /// The compute op is passed so operands (e.g., CBs) can be extracted.
  std::function<void(OpBuilder &, Location, Operation *)> createInit;
};

/// Build a static map from TTKernel compute op TypeID to init creation info.
/// Uses the same x-macro table as ConvertTTLTileOpsToTTKernel.
static llvm::DenseMap<mlir::TypeID, InitOpInfo> buildComputeToInitMap() {
  llvm::DenseMap<mlir::TypeID, InitOpInfo> map;

  // Unary SFPU ops: init takes no arguments.
#define TTL_UNARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)              \
  map[mlir::TypeID::get<ttk::TTK_COMPUTE>()] = {                               \
      [](OpBuilder &b, Location l, Operation *) {                              \
        b.create<ttk::TTK_INIT>(l);                                            \
      }};
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // Binary SFPU ops: init takes no arguments.
#define TTL_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)             \
  map[mlir::TypeID::get<ttk::TTK_COMPUTE>()] = {                               \
      [](OpBuilder &b, Location l, Operation *) {                              \
        b.create<ttk::TTK_INIT>(l);                                            \
      }};
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // MinMax ops: init takes no arguments.
#define TTL_BINARY_TILE_OP_MINMAX(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)      \
  map[mlir::TypeID::get<ttk::TTK_COMPUTE>()] = {                               \
      [](OpBuilder &b, Location l, Operation *) {                              \
        b.create<ttk::TTK_INIT>(l);                                            \
      }};
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // FPU binary ops: init takes 2 CB arguments (in0_cb, in1_cb).
#define TTL_FPU_BINARY_TILE_OP(TTL_OP, TILE_OP, TTK_INIT, TTK_COMPUTE)         \
  map[mlir::TypeID::get<ttk::TTK_COMPUTE>()] = {                               \
      [](OpBuilder &b, Location l, Operation *computeOp) {                     \
        b.create<ttk::TTK_INIT>(l, computeOp->getOperand(0),                   \
                                computeOp->getOperand(1));                     \
      }};
#include "ttlang/Dialect/TTL/TTLElementwiseOps.def"

  // CopyTile: init takes 1 CB argument (cb0, the first operand).
  map[mlir::TypeID::get<ttk::CopyTileOp>()] = {
      [](OpBuilder &b, Location l, Operation *computeOp) {
        b.create<ttk::CopyTileInitOp>(l, computeOp->getOperand(0));
      }};

  // CopyDestValues: init takes no arguments.
  map[mlir::TypeID::get<ttk::CopyDestValuesOp>()] = {
      [](OpBuilder &b, Location l, Operation *) {
        b.create<ttk::CopyDestValuesInitOp>(l);
      }};

  // UnaryBcast: init takes 2 CB args + bcast_type attr.
  map[mlir::TypeID::get<ttk::UnaryBcastTileOp>()] = {
      [](OpBuilder &b, Location l, Operation *computeOp) {
        auto bcastOp = cast<ttk::UnaryBcastTileOp>(computeOp);
        auto funcOp = computeOp->getParentOfType<func::FuncOp>();
        assert(funcOp && "unary_bcast must be inside a function");

        // Look up output CB via the annotated index (validated in
        // runOnOperation precondition check).
        auto cbIdxAttr =
            computeOp->getAttrOfType<IntegerAttr>(kBcastOutputCBIndexAttrName);
        assert(cbIdxAttr && "expected ttl.bcast_output_cb_index attribute");

        // After TTL -> TTKernel lowering, bind_cb becomes
        // get_compile_time_arg_val with arg_index matching the CB index.
        // TODO: Cache arg_index -> GetCompileArgValOp mapping to avoid O(N*M)
        // walk when many bcast ops exist.
        Value outCB;
        int64_t cbIdx = cbIdxAttr.getInt();
        funcOp->walk([&](ttk::GetCompileArgValOp argOp) {
          if (static_cast<int64_t>(argOp.getArgIndex()) == cbIdx) {
            outCB = argOp.getResult();
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        assert(outCB && "get_compile_time_arg_val must exist for cb_index");

        b.create<ttk::UnaryBcastInitOp>(l, bcastOp.getInCb(), outCB,
                                        bcastOp.getBcastTypeAttr());
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

  // For FPU binary: key includes CB operands (first 2 operands).
  if (isa<ttk::AddTilesOp, ttk::SubTilesOp, ttk::MulTilesOp>(op)) {
    return {typeId, {op->getOperand(0), op->getOperand(1)}};
  }

  // For CopyTile: key includes the CB operand (first operand).
  if (isa<ttk::CopyTileOp>(op)) {
    return {typeId, {op->getOperand(0)}};
  }

  // For UnaryBcast: key includes in_cb AND bcast_type.
  // Different bcast types (COL/ROW/SCALAR) require different inits.
  if (auto bcast = dyn_cast<ttk::UnaryBcastTileOp>(op)) {
    return {
        typeId, {bcast.getInCb()}, static_cast<int64_t>(bcast.getBcastType())};
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
static FailureOr<bool> analyzeSyncRegion(ttk::TileRegsAcquireOp acquireOp,
                                         Value &inputCB, Value &in0CB,
                                         Value &in1CB, Value &outputCB) {
  Block *block = acquireOp->getBlock();
  bool hasFPUBinary = false;
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
        hasFPUBinary = true;
        if (!in0CB) {
          in0CB = inner->getOperand(0);
          in1CB = inner->getOperand(1);
        }
      } else if (auto bcast = dyn_cast<ttk::UnaryBcastTileOp>(inner)) {
        if (!inputCB) {
          inputCB = bcast.getInCb();
        }
      }
      if (auto pack = dyn_cast<ttk::PackTileOp>(inner)) {
        if (!outputCB) {
          outputCB = pack.getOutCb();
        } else if (outputCB != pack.getOutCb() &&
                   outputCB.getType() != pack.getOutCb().getType()) {
          pack->emitOpError(
              "sync region packs to output CBs with different data formats; "
              "common init cannot configure multiple PACK formats");
          hadError = true;
        }
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
  return hasFPUBinary;
}

/// Find the outermost enclosing insertion point by walking up through
/// compiler-generated loops (marked with ttl.tile_loop or
/// ttl.subblock_stride). By construction, these loops iterate over tiles
/// within a single ttl.compute whose input/output CBs are fixed, so the
/// CB configuration is invariant across iterations and hoisting is safe.
/// Stops at unmarked loops to avoid hoisting past user loops that could
/// contain multiple sync regions with different CB configurations.
static Operation *hoistAboveCompilerLoops(Operation *op) {
  Operation *insertBefore = op;
  while (auto *parentOp = insertBefore->getParentOp()) {
    if (isa<scf::ForOp>(parentOp) &&
        (parentOp->hasAttr(kTileLoopAttrName) ||
         parentOp->hasAttr(kSubblockStrideAttrName))) {
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
    auto result = analyzeSyncRegion(acquireOp, inputCB, in0CB, in1CB, outputCB);
    if (failed(result)) {
      hadError = true;
      return;
    }
    bool hasFPUBinary = *result;

    // No output CB means the sync region has no pack ops -- nothing to
    // configure for UNPACK + PACK routing.
    if (!outputCB) {
      return;
    }

    // By construction, each sync region comes from a single ttl.compute
    // body with a fixed set of input CBs. Multiple input CBs in a single
    // region are not expected; if they occur, we use the first one found.

    Operation *insertBefore = hoistAboveCompilerLoops(acquireOp);
    OpBuilder builder(insertBefore);
    Location loc = acquireOp->getLoc();

    if (hasFPUBinary && in0CB && in1CB) {
      builder.create<ttk::BinaryOpInitCommonOp>(loc, in0CB, in1CB, outputCB);
    } else if (inputCB) {
      builder.create<ttk::InitSFPUOp>(loc, inputCB, outputCB);
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

    // Phase 1: Insert common inits (init_sfpu / binary_op_init_common).
    if (failed(insertCommonInits(moduleOp))) {
      signalPassFailure();
      return;
    }

    // Validate preconditions: all unary_bcast ops must carry the output CB
    // index attribute (set during TTL -> TTKernel lowering).
    bool hadError = false;
    moduleOp->walk([&](ttk::UnaryBcastTileOp bcastOp) {
      if (!bcastOp->hasAttr(kBcastOutputCBIndexAttrName)) {
        bcastOp->emitOpError("missing ttl.bcast_output_cb_index attribute; "
                             "cannot insert unary_bcast_init");
        hadError = true;
      }
    });
    if (hadError) {
      signalPassFailure();
      return;
    }

    // Phase 2: Insert per-op inits for compute ops.
    auto computeToInit = buildComputeToInitMap();
    moduleOp->walk([&](Block *block) {
      std::optional<InitKey> prevKey;

      for (Operation &op : *block) {
        // Reset tracking at sync boundaries.
        if (isSyncBoundary(&op)) {
          prevKey = std::nullopt;
          continue;
        }

        // Look up this op in the compute-to-init map.
        auto it = computeToInit.find(op.getName().getTypeID());
        if (it == computeToInit.end()) {
          continue;
        }

        // Compute init key for this op.
        InitKey key = computeInitKey(&op);

        // Insert init if key changed from previous compute op.
        if (!prevKey || *prevKey != key) {
          OpBuilder builder(&op);
          it->second.createInit(builder, op.getLoc(), &op);
        }

        prevKey = key;
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
