// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// TTKernel Init Consolidation Pass
//===----------------------------------------------------------------------===//
//
// Walks TTKernel IR and inserts the minimal set of init ops before compute ops.
// After convert-ttl-to-ttkernel (which no longer emits inits), this pass
// inserts one init op per consecutive group of same-type compute ops.
//
// The init key is (init op TypeID, operand values). An init is inserted only
// when the key changes between consecutive compute ops. Tracking resets at
// sync boundaries (tile_regs_acquire/commit/wait/release).
//
//===----------------------------------------------------------------------===//

#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelTraits.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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
        // The out_cb comes from the init_sfpu op. We search for it.
        auto funcOp = computeOp->getParentOfType<func::FuncOp>();
        Value outCB;
        if (funcOp) {
          funcOp->walk([&](ttk::InitSFPUOp initOp) {
            outCB = initOp.getOcb();
            return WalkResult::interrupt();
          });
        }
        if (outCB) {
          b.create<ttk::UnaryBcastInitOp>(l, bcastOp.getInCb(), outCB,
                                          bcastOp.getBcastTypeAttr());
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
    return {typeId,
            {bcast.getInCb()},
            static_cast<int64_t>(bcast.getBcastType())};
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
// Pass implementation
//===----------------------------------------------------------------------===//

struct TTKernelConsolidateInitsPass
    : public impl::TTKernelConsolidateInitsBase<TTKernelConsolidateInitsPass> {

  void runOnOperation() override {
    auto moduleOp = getOperation();
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
