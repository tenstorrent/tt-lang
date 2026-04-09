// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Conversion/StableHLOToTTL/StableHLOToTTL.h"
#include "ttlang/Conversion/StableHLOToTTL/ShardingUtils.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsEnums.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "shardy/dialect/sdy/ir/dialect.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::tt;

namespace mlir::tt::ttl {
#define GEN_PASS_DEF_CONVERTSTABLEHLOTOTLL
#include "ttlang/Dialect/TTL/Passes.h.inc"
} // namespace mlir::tt::ttl

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static constexpr int64_t kTileSize = 32;
static constexpr int64_t kDefaultBlockCount = 2;

/// Map a scalar element type (bf16, f16, f32) to a ttcore::TileType.
static ttcore::TileType getTileType(Type scalarType) {
  return ttcore::TileType::get(scalarType);
}

/// Build a ranked tensor type of tiles from a tile shape.
/// E.g., tileShape={4,1}, scalarType=bf16 -> tensor<4x1x!ttcore.tile<32x32,bf16>>
static RankedTensorType getTiledTensorType(MLIRContext *ctx,
                                           ArrayRef<int64_t> tileShape,
                                           Type scalarType) {
  return RankedTensorType::get(tileShape, getTileType(scalarType));
}

/// Build a DRAM layout tensor type (for func args).
/// E.g., tileShape={4,1}, scalarType=bf16 ->
///   tensor<4x1x!ttcore.tile<32x32,bf16>, #ttl.layout<...dram...>>
static RankedTensorType getDRAMTensorType(MLIRContext *ctx,
                                          ArrayRef<int64_t> tileShape,
                                          ArrayRef<int64_t> localShape,
                                          Type scalarType) {
  auto tileType = getTileType(scalarType);
  auto layout = ttl::LayoutAttr::get(ctx, localShape, tileType,
                                     ttl::BufferType::DRAM,
                                     /*grid=*/{1, 1},
                                     ttl::TensorMemoryLayout::Interleaved);
  return RankedTensorType::get(tileShape, tileType, layout);
}

/// Build CB type for a given tile shape.
static ttl::CircularBufferType getCBType(MLIRContext *ctx,
                                         ArrayRef<int64_t> tileShape,
                                         Type scalarType) {
  return ttl::CircularBufferType::get(ctx, tileShape, getTileType(scalarType),
                                      kDefaultBlockCount);
}

/// Create a func.func with the given name, type, and kernel thread attribute.
static func::FuncOp createKernelFunc(OpBuilder &builder, Location loc,
                                     StringRef name, FunctionType funcType,
                                     ttkernel::ThreadType thread) {
  auto funcOp = func::FuncOp::create(builder, loc, name, funcType);
  funcOp->setAttr("ttl.kernel_thread",
                  ttkernel::ThreadTypeAttr::get(builder.getContext(), thread));
  funcOp.setPrivate();
  return funcOp;
}

// ---------------------------------------------------------------------------
// Op classification
// ---------------------------------------------------------------------------

enum class OpKind { BinaryElementwise, UnaryElementwise, Matmul, Unsupported };

static OpKind classifyOp(Operation *op) {
  // Binary elementwise
#define SHLO_TO_TTL_BINARY(SHLO_OP, TTL_OP)                                   \
  if (isa<stablehlo::SHLO_OP>(op))                                            \
    return OpKind::BinaryElementwise;
#define SHLO_TO_TTL_UNARY(SHLO_OP, TTL_OP)
#include "ttlang/Conversion/StableHLOToTTL/StableHLOToTTLOps.def"

  // Unary elementwise
#define SHLO_TO_TTL_BINARY(SHLO_OP, TTL_OP)
#define SHLO_TO_TTL_UNARY(SHLO_OP, TTL_OP)                                    \
  if (isa<stablehlo::SHLO_OP>(op))                                            \
    return OpKind::UnaryElementwise;
#include "ttlang/Conversion/StableHLOToTTL/StableHLOToTTLOps.def"

  if (isa<stablehlo::DotGeneralOp>(op))
    return OpKind::Matmul;

  return OpKind::Unsupported;
}

/// Emit the TTL tensor-level op corresponding to a StableHLO op.
/// For binary: ttl.add(%lhs, %rhs), for unary: ttl.exp(%operand).
/// Returns the result Value.
static FailureOr<Value> emitTTLOp(OpBuilder &builder, Location loc,
                                  Operation *shloOp, ValueRange operands,
                                  RankedTensorType resultType) {
  // Binary elementwise
#define SHLO_TO_TTL_BINARY(SHLO_OP, TTL_OP)                                   \
  if (isa<stablehlo::SHLO_OP>(shloOp))                                        \
    return ttl::TTL_OP::create(builder, loc, resultType, operands[0],          \
                               operands[1])                                    \
        ->getResult(0);
#define SHLO_TO_TTL_UNARY(SHLO_OP, TTL_OP)
#include "ttlang/Conversion/StableHLOToTTL/StableHLOToTTLOps.def"

  // Unary elementwise
#define SHLO_TO_TTL_BINARY(SHLO_OP, TTL_OP)
#define SHLO_TO_TTL_UNARY(SHLO_OP, TTL_OP)                                    \
  if (isa<stablehlo::SHLO_OP>(shloOp))                                        \
    return ttl::TTL_OP::create(builder, loc, resultType, operands[0])          \
        ->getResult(0);
#include "ttlang/Conversion/StableHLOToTTL/StableHLOToTTLOps.def"

  return emitError(loc, "unsupported op for TTL: ") << shloOp->getName();
}

// ---------------------------------------------------------------------------
// Matmul validation
// ---------------------------------------------------------------------------

static LogicalResult validateDotGeneral(stablehlo::DotGeneralOp op) {
  auto dimNums = op.getDotDimensionNumbers();
  auto lhsContract = dimNums.getLhsContractingDimensions();
  auto rhsContract = dimNums.getRhsContractingDimensions();

  if (lhsContract.size() != 1 || rhsContract.size() != 1)
    return op.emitError("only single contraction dimension supported");
  if (lhsContract[0] != 1 || rhsContract[0] != 0)
    return op.emitError("only contracting_dims [1]x[0] supported");
  if (!dimNums.getLhsBatchingDimensions().empty())
    return op.emitError("batched matmul not yet supported");

  return success();
}

// ---------------------------------------------------------------------------
// Reader function generation
// ---------------------------------------------------------------------------

/// Generate dm_read function: one DMA per input, copies full local shard to CB.
static func::FuncOp
generateReader(OpBuilder &moduleBuilder, Location loc, StringRef baseName,
               ArrayRef<ttl::TensorShardInfo> inputInfos, MLIRContext *ctx) {

  // Build function type: (dram_tensor0, dram_tensor1, ...) -> ()
  SmallVector<Type> argTypes;
  for (auto &info : inputInfos)
    argTypes.push_back(
        getDRAMTensorType(ctx, info.tileShape, info.localShape,
                          info.elementType));

  auto funcType = FunctionType::get(ctx, argTypes, {});
  auto funcOp = createKernelFunc(moduleBuilder, loc,
                                 (baseName + "_dm_read").str(), funcType,
                                 ttkernel::ThreadType::Noc);

  Block *body = funcOp.addEntryBlock();
  OpBuilder builder(body, body->end());
  auto c0 = arith::ConstantIndexOp::create(builder, loc, 0);

  for (auto [cbIdx, info] : llvm::enumerate(inputInfos)) {
    auto cbType = getCBType(ctx, info.tileShape, info.elementType);
    auto cb = ttl::BindCBOp::create(builder, loc, cbType,
                                    builder.getIndexAttr(cbIdx),
                                    builder.getI64IntegerAttr(kDefaultBlockCount));

    auto tensorType = getTiledTensorType(ctx, info.tileShape, info.elementType);
    auto reserve =
        ttl::CBReserveOp::create(builder, loc, tensorType, cb, IntegerAttr{});

    // Build index list for tensor_slice at origin
    SmallVector<Value> indices(info.tileShape.size(), c0);
    auto srcArg = body->getArgument(cbIdx);
    auto sliceType = RankedTensorType::get(
        info.tileShape, getTileType(info.elementType),
        cast<RankedTensorType>(srcArg.getType()).getEncoding());
    auto slice =
        ttl::TensorSliceOp::create(builder, loc, sliceType, srcArg, indices);

    auto xfType = ttl::TransferHandleType::get(ctx, ttl::TransferKind::read);
    auto copy = ttl::CopyOp::create(builder, loc, xfType, slice, cb);
    ttl::WaitOp::create(builder, loc, copy);
    ttl::CBPushOp::create(builder, loc, cb, IntegerAttr{});
  }

  func::ReturnOp::create(builder, loc, ValueRange{});
  return funcOp;
}

// ---------------------------------------------------------------------------
// Compute function generation
// ---------------------------------------------------------------------------

static FailureOr<func::FuncOp>
generateCompute(OpBuilder &moduleBuilder, Location loc, StringRef baseName,
                ArrayRef<ttl::TensorShardInfo> inputInfos,
                ArrayRef<ttl::TensorShardInfo> outputInfos,
                Block &shloBody, MLIRContext *ctx) {

  auto funcType = FunctionType::get(ctx, {}, {});
  auto funcOp = createKernelFunc(moduleBuilder, loc,
                                 (baseName + "_compute").str(), funcType,
                                 ttkernel::ThreadType::Compute);

  Block *body = funcOp.addEntryBlock();
  OpBuilder builder(body, body->end());

  // Bind input CBs and wait for data
  unsigned cbIdx = 0;
  IRMapping valueMap;
  SmallVector<Value> inputCBs;

  for (auto [i, info] : llvm::enumerate(inputInfos)) {
    auto cbType = getCBType(ctx, info.tileShape, info.elementType);
    auto cb = ttl::BindCBOp::create(builder, loc, cbType,
                                    builder.getIndexAttr(cbIdx),
                                    builder.getI64IntegerAttr(kDefaultBlockCount));
    inputCBs.push_back(cb);

    auto tensorType = getTiledTensorType(ctx, info.tileShape, info.elementType);
    auto waited = ttl::CBWaitOp::create(builder, loc, tensorType, cb);
    auto attached = ttl::AttachCBOp::create(builder, loc, tensorType, waited, cb);

    // Map sdy.manual_computation block arg -> attached CB tensor
    valueMap.map(shloBody.getArgument(i), attached);
    cbIdx++;
  }

  // Bind output CBs and reserve space
  SmallVector<Value> outputCBs;
  SmallVector<Value> outputReserves;

  for (auto &info : outputInfos) {
    auto cbType = getCBType(ctx, info.tileShape, info.elementType);
    auto cb = ttl::BindCBOp::create(builder, loc, cbType,
                                    builder.getIndexAttr(cbIdx),
                                    builder.getI64IntegerAttr(kDefaultBlockCount));
    outputCBs.push_back(cb);

    auto tensorType = getTiledTensorType(ctx, info.tileShape, info.elementType);
    auto reserve =
        ttl::CBReserveOp::create(builder, loc, tensorType, cb, IntegerAttr{});
    outputReserves.push_back(reserve);
    cbIdx++;
  }

  // Walk the StableHLO body and emit corresponding TTL tensor-level ops
  for (auto &op : shloBody.without_terminator()) {
    if (isa<sdy::ReturnOp>(&op))
      continue;

    auto kind = classifyOp(&op);
    if (kind == OpKind::Unsupported) {
      // Check for CCL ops
      if (isa<sdy::AllGatherOp, sdy::AllReduceOp, sdy::AllSliceOp,
              sdy::AllToAllOp, sdy::CollectivePermuteOp, sdy::ReduceScatterOp>(
              &op))
        return op.emitError("CCL ops not yet supported (future: map to TTNN "
                            "CCLs): ")
               << op.getName();
      return op.emitError("unsupported op for TTL: ") << op.getName();
    }

    // Gather mapped operands
    SmallVector<Value> mappedOperands;
    for (auto operand : op.getOperands())
      mappedOperands.push_back(valueMap.lookup(operand));

    if (kind == OpKind::Matmul) {
      auto dotOp = cast<stablehlo::DotGeneralOp>(&op);
      if (failed(validateDotGeneral(dotOp)))
        return failure();

      // Result type: [M, N] tiles
      auto lhsInfo = inputInfos[0]; // placeholder, actual shape from operands
      auto lhsTy = cast<RankedTensorType>(mappedOperands[0].getType());
      auto rhsTy = cast<RankedTensorType>(mappedOperands[1].getType());
      int64_t M = lhsTy.getShape()[0];
      int64_t N = rhsTy.getShape()[1];
      auto resultType = getTiledTensorType(ctx, {M, N},
                                           lhsTy.getElementType());
      auto mm = ttl::MatmulOp::create(builder, loc, resultType,
                                       mappedOperands[0], mappedOperands[1]);
      valueMap.map(op.getResult(0), mm->getResult(0));
    } else {
      // Elementwise: result type matches first operand's type
      auto operandType = cast<RankedTensorType>(mappedOperands[0].getType());
      auto result = emitTTLOp(builder, loc, &op, mappedOperands, operandType);
      if (failed(result))
        return failure();
      valueMap.map(op.getResult(0), *result);
    }
  }

  // Store results to output CB reserves, then push/pop
  auto *terminator = shloBody.getTerminator();
  for (auto [i, returnVal] : llvm::enumerate(terminator->getOperands())) {
    auto mapped = valueMap.lookup(returnVal);
    ttl::StoreOp::create(builder, loc, mapped, outputReserves[i]);
  }

  for (auto cb : inputCBs)
    ttl::CBPopOp::create(builder, loc, cb);
  for (auto cb : outputCBs)
    ttl::CBPushOp::create(builder, loc, cb, IntegerAttr{});

  func::ReturnOp::create(builder, loc, ValueRange{});
  return funcOp;
}

// ---------------------------------------------------------------------------
// Writer function generation
// ---------------------------------------------------------------------------

static func::FuncOp
generateWriter(OpBuilder &moduleBuilder, Location loc, StringRef baseName,
               ArrayRef<ttl::TensorShardInfo> outputInfos,
               unsigned outputCBStart, MLIRContext *ctx) {

  SmallVector<Type> argTypes;
  for (auto &info : outputInfos)
    argTypes.push_back(
        getDRAMTensorType(ctx, info.tileShape, info.localShape,
                          info.elementType));

  auto funcType = FunctionType::get(ctx, argTypes, {});
  auto funcOp = createKernelFunc(moduleBuilder, loc,
                                 (baseName + "_dm_write").str(), funcType,
                                 ttkernel::ThreadType::Noc);

  Block *body = funcOp.addEntryBlock();
  OpBuilder builder(body, body->end());
  auto c0 = arith::ConstantIndexOp::create(builder, loc, 0);

  for (auto [i, info] : llvm::enumerate(outputInfos)) {
    unsigned cbIdx = outputCBStart + i;
    auto cbType = getCBType(ctx, info.tileShape, info.elementType);
    auto cb = ttl::BindCBOp::create(builder, loc, cbType,
                                    builder.getIndexAttr(cbIdx),
                                    builder.getI64IntegerAttr(kDefaultBlockCount));

    auto tensorType = getTiledTensorType(ctx, info.tileShape, info.elementType);
    auto waited = ttl::CBWaitOp::create(builder, loc, tensorType, cb);

    SmallVector<Value> indices(info.tileShape.size(), c0);
    auto dstArg = body->getArgument(i);
    auto sliceType = RankedTensorType::get(
        info.tileShape, getTileType(info.elementType),
        cast<RankedTensorType>(dstArg.getType()).getEncoding());
    auto slice =
        ttl::TensorSliceOp::create(builder, loc, sliceType, dstArg, indices);

    auto xfType = ttl::TransferHandleType::get(ctx, ttl::TransferKind::write);
    auto copy = ttl::CopyOp::create(builder, loc, xfType, cb, slice);
    ttl::WaitOp::create(builder, loc, copy);
    ttl::CBPopOp::create(builder, loc, cb);
  }

  func::ReturnOp::create(builder, loc, ValueRange{});
  return funcOp;
}

// ---------------------------------------------------------------------------
// Main pass
// ---------------------------------------------------------------------------

struct ConvertStableHLOToTTLPass
    : public ttl::impl::ConvertStableHLOToTTLBase<ConvertStableHLOToTTLPass> {

  void runOnOperation() final {
    auto module = getOperation();
    auto *ctx = module.getContext();

    // Find the sdy.mesh op
    sdy::MeshOp meshOp;
    module.walk([&](sdy::MeshOp op) { meshOp = op; });
    if (!meshOp) {
      module.emitError("expected sdy.mesh op in module");
      return signalPassFailure();
    }
    auto meshInfo = ttl::parseMesh(meshOp);

    // Collect manual_computation ops (can't modify while walking)
    SmallVector<sdy::ManualComputationOp> manualOps;
    module.walk(
        [&](sdy::ManualComputationOp op) { manualOps.push_back(op); });

    if (manualOps.empty()) {
      module.emitError(
          "expected sdy.manual_computation (run Shardy propagation first)");
      return signalPassFailure();
    }

    for (auto manualOp : manualOps) {
      Location loc = manualOp.getLoc();
      Block &body = manualOp.getBody().front();

      // Parse input tensor infos from block arg types
      SmallVector<ttl::TensorShardInfo> inputInfos;
      for (auto arg : body.getArguments()) {
        auto tensorType = dyn_cast<RankedTensorType>(arg.getType());
        if (!tensorType) {
          arg.getLoc().print(llvm::errs());
          manualOp.emitError("expected ranked tensor block argument");
          return signalPassFailure();
        }
        auto info = ttl::parseTensorInfo(tensorType, arg.getLoc());
        if (failed(info))
          return signalPassFailure();
        inputInfos.push_back(*info);
      }

      // Parse output tensor infos from terminator operand types
      auto *terminator = body.getTerminator();
      SmallVector<ttl::TensorShardInfo> outputInfos;
      for (auto result : terminator->getOperands()) {
        auto tensorType = dyn_cast<RankedTensorType>(result.getType());
        if (!tensorType) {
          manualOp.emitError("expected ranked tensor return type");
          return signalPassFailure();
        }
        auto info = ttl::parseTensorInfo(tensorType, loc);
        if (failed(info))
          return signalPassFailure();
        outputInfos.push_back(*info);
      }

      // Validate all body ops are supported
      for (auto &op : body.without_terminator()) {
        if (isa<sdy::ReturnOp>(&op))
          continue;
        if (classifyOp(&op) == OpKind::Unsupported) {
          if (isa<sdy::AllGatherOp, sdy::AllReduceOp, sdy::AllSliceOp,
                  sdy::AllToAllOp, sdy::CollectivePermuteOp,
                  sdy::ReduceScatterOp>(&op)) {
            op.emitError("CCL ops not yet supported (future: map to TTNN "
                         "CCLs): ")
                << op.getName();
          } else {
            op.emitError("unsupported op for TTL: ") << op.getName();
          }
          return signalPassFailure();
        }
      }

      // Generate a base name from the parent func or a default
      StringRef baseName = "kernel";
      if (auto parentFunc = manualOp->getParentOfType<func::FuncOp>())
        baseName = parentFunc.getName();

      // Insert generated functions into the module
      OpBuilder moduleBuilder(ctx);
      moduleBuilder.setInsertionPoint(manualOp->getParentOfType<func::FuncOp>());

      auto reader = generateReader(moduleBuilder, loc, baseName, inputInfos, ctx);
      module.push_back(reader);

      auto computeOrErr = generateCompute(moduleBuilder, loc, baseName,
                                          inputInfos, outputInfos, body, ctx);
      if (failed(computeOrErr))
        return signalPassFailure();
      module.push_back(*computeOrErr);

      unsigned outputCBStart = inputInfos.size();
      auto writer = generateWriter(moduleBuilder, loc, baseName, outputInfos,
                                   outputCBStart, ctx);
      module.push_back(writer);
    }
  }
};

} // namespace

namespace mlir::tt::ttl {

std::unique_ptr<mlir::Pass> createConvertStableHLOToTTL() {
  return std::make_unique<ConvertStableHLOToTTLPass>();
}

} // namespace mlir::tt::ttl
