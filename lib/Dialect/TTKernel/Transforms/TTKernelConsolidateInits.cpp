// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTKernel Init Consolidation Pass
//===----------------------------------------------------------------------===//
//
// Single source of truth for all init ops in the TTKernel pipeline.
// Inserts both common inits (init_sfpu, binary_op_init_common) that configure
// UNPACK + PACK data format routing, and per-op inits (exp_tile_init,
// add_tiles_init, etc.) that configure the MATH pipeline.
//
// Two phases:
//   1. Common inits: one per sync region, hoisted above enclosing loops.
//      Scans each tile_regs_acquire -> tile_regs_release region to determine
//      the compute category (FPU binary vs SFPU/copy/bcast) and derives
//      input/output CBs from compute and pack ops.
//   2. Per-op inits: one per consecutive group of same-type compute ops.
//      The init key is (init op TypeID, operand values). An init is inserted
//      only when the key changes. Tracking resets at sync boundaries.
//
// TODO: Consecutive same-type ops still get a full init (e.g., add_tiles_init)
// on every type switch. tt-metal exposes init_short variants
// (add_tiles_init_short, mul_tiles_init_short) that reconfigure UNPACK+MATH
// without touching PACK, but TTKernel only models mm_init_short /
// mm_block_init_short (matmul). Adding the elementwise init_short ops to
// TTKernel would let this pass emit cheaper re-inits when only the op type
// changes but the output CB stays the same.
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

#define DEBUG_TYPE "ttkernel-consolidate-inits"

namespace mlir::tt::ttl {

namespace ttk = mlir::tt::ttkernel;

#define GEN_PASS_DEF_TTKERNELCONSOLIDATEINITS
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
        // Bcast init needs in_cb and out_cb. The compute op only has in_cb.
        // Derive out_cb from pack_tile (always present when bcast result is
        // packed to output), falling back to init_sfpu/binary_op_init_common.
        auto funcOp = computeOp->getParentOfType<func::FuncOp>();
        Value outCB;
        if (funcOp) {
          // Primary: derive from pack_tile (works for bcast-only functions).
          funcOp->walk([&](ttk::PackTileOp pack) {
            outCB = pack.getOutCb();
            return WalkResult::interrupt();
          });
          // Fallback: derive from existing init ops.
          if (!outCB) {
            funcOp->walk([&](Operation *op) {
              if (auto sfpu = dyn_cast<ttk::InitSFPUOp>(op)) {
                outCB = sfpu.getOcb();
                return WalkResult::interrupt();
              }
              if (auto binary = dyn_cast<ttk::BinaryOpInitCommonOp>(op)) {
                outCB = binary.getOutCb();
                return WalkResult::interrupt();
              }
              return WalkResult::advance();
            });
          }
        }
        if (outCB) {
          b.create<ttk::UnaryBcastInitOp>(l, bcastOp.getInCb(), outCB,
                                          bcastOp.getBcastTypeAttr());
        } else {
          computeOp->emitWarning(
              "cannot derive output CB for unary_bcast_init; "
              "no pack_tile, init_sfpu, or binary_op_init_common found");
        }
      }};

  return map;
}

/// Compute a key that identifies when an init op needs to change.
/// Init key: identifies which compute ops can share a single init call.
/// Two consecutive ops with the same key need only one init before the group.
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
/// Returns true if FPU binary ops are present in the region.
static bool analyzeSyncRegion(ttk::TileRegsAcquireOp acquireOp, Value &inputCB,
                              Value &in0CB, Value &in1CB, Value &outputCB) {
  Block *block = acquireOp->getBlock();
  bool hasFPUBinary = false;

  for (auto it = std::next(acquireOp->getIterator()); it != block->end();
       ++it) {
    if (isa<ttk::TileRegsReleaseOp>(&*it)) {
      break;
    }

    // Walk this op and all nested regions (e.g., scf.for bodies).
    (&*it)->walk([&](Operation *inner) {
      if (auto copy = dyn_cast<ttk::CopyTileOp>(inner)) {
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
        } else if (outputCB != pack.getOutCb()) {
          // TODO: Extend to emit one common init per distinct output CB.
          pack->emitWarning("sync region packs to multiple output CBs; "
                            "common init only configured for the first");
        }
      }
    });
  }

  return hasFPUBinary;
}

/// Find the outermost enclosing insertion point by walking up through
/// compiler-generated loops (marked with ttl.tile_loop or
/// ttl.subblock_stride). These loops have invariant CB configuration across
/// iterations, so hoisting the common init above them is always safe.
/// Stops at unmarked loops to avoid hoisting past user loops that could
/// contain multiple sync regions with different init types.
///
/// TODO: A more aggressive strategy would analyze all sync regions inside an
/// unmarked loop and hoist if they all need the same init type. This would
/// allow hoisting through user loops that wrap compiler loops (e.g., the
/// streaming pattern in test_large_dram_streaming.py).
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
static void insertCommonInits(ModuleOp moduleOp) {
  moduleOp->walk([&](ttk::TileRegsAcquireOp acquireOp) {
    Value inputCB, in0CB, in1CB, outputCB;
    bool hasFPUBinary =
        analyzeSyncRegion(acquireOp, inputCB, in0CB, in1CB, outputCB);

    if (!outputCB) {
      return;
    }

    Operation *insertBefore = hoistAboveCompilerLoops(acquireOp);
    OpBuilder builder(insertBefore);
    Location loc = acquireOp->getLoc();

    if (hasFPUBinary && in0CB && in1CB) {
      builder.create<ttk::BinaryOpInitCommonOp>(loc, in0CB, in1CB, outputCB);
    } else if (inputCB) {
      builder.create<ttk::InitSFPUOp>(loc, inputCB, outputCB);
    }
  });
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct TTKernelConsolidateInitsPass
    : public impl::TTKernelConsolidateInitsBase<TTKernelConsolidateInitsPass> {

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Phase 1: Insert common inits (init_sfpu / binary_op_init_common).
    insertCommonInits(moduleOp);

    // Phase 2: Insert per-op inits for compute ops.
    auto computeToInit = buildComputeToInitMap();

    // Walk all blocks (including nested ones inside scf.for loops).
    // Each block tracks init state independently - sync regions don't cross
    // block boundaries.
    moduleOp->walk([&](Block *block) {
      std::optional<InitKey> prevKey;

      for (Operation &op : llvm::make_early_inc_range(*block)) {
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
