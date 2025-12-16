// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h" // IWYU pragma: keep

#include "ttlang/Dialect/TTKernel/Transforms/TTKernelCleanupPatterns.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsEnums.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/Utils/ConversionUtils.h"
#include "ttlang/Dialect/Utils/LayoutUtils.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h" // IWYU pragma: keep
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"

namespace mlir::tt::ttl {
#define GEN_PASS_DEF_TTLCONVERTTTLTOTTKERNEL
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

using mlir::LogicalResult;
using mlir::PatternRewriter;
using mlir::RewritePatternSet;
using mlir::TypeConverter;
using mlir::UnrealizedConversionCastOp;
using mlir::ValueRange;
using mlir::func::FuncOp;
namespace ttk = mlir::tt::ttkernel;

class TTLToTTKernelTypeConverter : public TypeConverter {
public:
  TTLToTTKernelTypeConverter() {
    // Specific conversions first; identity fallback last.
    // CB: lower to TTKernel CB type with flattened element count.
    addConversion([](CircularBufferType t) -> Type {
      return ttk::CBType::get(t.getContext(), t.getTotalElements(),
                              t.getElementType());
    });
    // Tensor -> TensorAccessor for TTKernel when TTNN layout is present.
    addConversion([](RankedTensorType t) -> Type {
      if (t.getEncoding() &&
          mlir::isa<tt::ttnn::TTNNLayoutAttr>(t.getEncoding())) {
        return ttk::TensorAccessorType::get(t.getContext());
      }
      return t;
    });
    // Transfer handle: map to an i32 (TRID placeholder until TTKernel TRID ops
    // are wired through).
    addConversion([](TransferHandleType t) -> Type {
      return IntegerType::get(t.getContext(), 32);
    });
    // Identity fallback must be last.
    addConversion([](Type t) { return t; });

    auto castMaterialization = [](OpBuilder &builder, Type resultType,
                                  ValueRange inputs, Location loc) -> Value {
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };
    addSourceMaterialization(castMaterialization);
    addTargetMaterialization(castMaterialization);
  }
};

//===----------------------------------------------------------------------===//
// Helper utilities.
//===----------------------------------------------------------------------===//

static std::optional<ttk::ThreadType> getKernelThreadType(Operation *op) {
  if (auto a = op->getAttrOfType<ttk::ThreadTypeAttr>("ttl.kernel_thread")) {
    return a.getValue();
  }
  return std::nullopt;
}

static bool isNocKernel(Operation *op) {
  return getKernelThreadType(op) == ttk::ThreadType::Noc;
}

static Value buildTensorAccessor(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 Value rowStride, Value colStride,
                                 Value bankBase, Value pageSize) {
  auto args =
      rewriter.create<ttk::TensorAccessorArgsOp>(loc, rowStride, colStride);
  auto accessor = rewriter.create<ttk::TensorAccessorOp>(loc, args.getResult(),
                                                         bankBase, pageSize);
  return accessor.getResult();
}

template <typename FuncLike>
static bool eraseUnusedArguments(FuncLike funcLike) {
  if (funcLike.getNumArguments() == 0) {
    return false;
  }
  if (llvm::any_of(funcLike.getArguments(),
                   [](BlockArgument arg) { return !arg.use_empty(); })) {
    return false;
  }

  llvm::BitVector argsToErase(funcLike.getNumArguments());
  for (unsigned idx = 0; idx < funcLike.getNumArguments(); ++idx) {
    argsToErase.set(idx);
  }
  if (failed(funcLike.eraseArguments(argsToErase))) {
    return false;
  }

  auto newType = FunctionType::get(funcLike.getContext(), TypeRange{},
                                   funcLike.getFunctionType().getResults());
  funcLike.setType(newType);
  return true;
}

struct CreateCBLowering : OpConversionPattern<CreateCBOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CreateCBOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter = this->getTypeConverter();
    if (!typeConverter) {
      return rewriter.notifyMatchFailure(op, "no type converter");
    }

    auto converted = typeConverter->convertType(op.getResult().getType());
    if (!converted) {
      return rewriter.notifyMatchFailure(op, "failed to convert CB type");
    }

    // For now, fabricate a CB value via unrealized cast; real lowering would
    // supply a concrete CB handle.
    auto zero = rewriter.create<arith::ConstantIntOp>(op.getLoc(), 0, 32);
    auto cast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), converted, ValueRange{zero});
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};

enum class CopySourceKind { TensorAccessor, CircularBuffer, Pipe, Unknown };
enum class CopyDestKind { TensorAccessor, CircularBuffer, Pipe, Unknown };

static bool isTensorAccessorLike(Type t) {
  return llvm::isa<ttk::TensorAccessorType>(t) ||
         llvm::isa<RankedTensorType>(t);
}

static CopySourceKind classifySrc(Value v) {
  if (llvm::isa<CircularBufferType>(v.getType())) {
    return CopySourceKind::CircularBuffer;
  }
  if (isTensorAccessorLike(v.getType())) {
    return CopySourceKind::TensorAccessor;
  }
  return CopySourceKind::Unknown;
}

static CopyDestKind classifyDst(Value v) {
  if (llvm::isa<CircularBufferType>(v.getType())) {
    return CopyDestKind::CircularBuffer;
  }
  if (isTensorAccessorLike(v.getType())) {
    return CopyDestKind::TensorAccessor;
  }
  return CopyDestKind::Unknown;
}

static Value emitPlaceholderCB(ValueRange inputs,
                               ConversionPatternRewriter &rewriter,
                               Location loc, Type targetType) {
  // TODO(ttl): Emit a real TTKernel CB handle instead of an unrealized cast.
  // Issue: #78.
  auto cast =
      rewriter.create<UnrealizedConversionCastOp>(loc, targetType, inputs);
  return cast.getResult(0);
}

static Value makeZeroI32(Location loc, ConversionPatternRewriter &rewriter) {
  return rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
}

static std::optional<TransferKind> getTransferKindFromHandleType(Type t) {
  auto transferHandle = llvm::dyn_cast<TransferHandleType>(t);
  if (!transferHandle) {
    return std::nullopt;
  }
  return transferHandle.getKind();
}

static FailureOr<Value>
materializeTensorAccessor(Value tensor, ConversionPatternRewriter &rewriter) {
  if (!llvm::isa<RankedTensorType>(tensor.getType())) {
    return failure();
  }

  auto loc = tensor.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  if (auto barg = llvm::dyn_cast<BlockArgument>(tensor)) {
    // Hoist accessor construction to the entry block when the source is a
    // function argument so the accessor does not end up between generated
    // tile loops for multiple copies.
    Block *block = barg.getParentBlock();
    if (block && block->isEntryBlock()) {
      rewriter.setInsertionPointToStart(block);
    }
  }
  auto tensorTy = mlir::cast<RankedTensorType>(tensor.getType());
  utils::ContiguousLayoutInfo layout;
  if (auto enc = tensorTy.getEncoding()) {
    if (mlir::isa<tt::ttnn::TTNNLayoutAttr>(enc)) {
      // TODO(ttl): Derive strides/page size/bank base from TTNNLayoutAttr.
      // Issue: #81.
    }
  }
  layout = utils::computeContiguousLayout(tensorTy);

  // Strides in elements (row-major placeholder).
  auto rowStride =
      rewriter.create<arith::ConstantIntOp>(loc, layout.rowStrideElems, 32);
  auto colStride =
      rewriter.create<arith::ConstantIntOp>(loc, layout.colStrideElems, 32);

  // Page size placeholder uses contiguous row-major; bank base is still a stub.
  auto pageSize =
      rewriter.create<arith::ConstantIntOp>(loc, layout.pageSizeBytes, 32);
  auto bankBase = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);

  return buildTensorAccessor(loc, rewriter, rowStride, colStride, bankBase,
                             pageSize);
}

static std::pair<int64_t, int64_t>
getTileGridShape(const RankedTensorType &tensorTy) {
  auto dims = tensorTy.getShape();
  assert(dims.size() == 2 && "only rank-2 tensors supported currently");
  auto ceilDiv = [](int64_t num, int64_t den) { return (num + den - 1) / den; };
  int64_t tilesY = ceilDiv(dims[0], kDefaultTileHeight);
  int64_t tilesX = ceilDiv(dims[1], kDefaultTileWidth);
  return {tilesY, tilesX};
}

/// Extract tile grid shape from a Value if it's a static rank-2 tensor.
/// Returns {1, 1} for non-tensor types or dynamic shapes.
static std::pair<int64_t, int64_t> getTileGridShapeFromValue(Value v) {
  auto tensorTy = llvm::dyn_cast<RankedTensorType>(v.getType());
  if (tensorTy && tensorTy.hasStaticShape() && tensorTy.getRank() == 2) {
    return getTileGridShape(tensorTy);
  }
  return {1, 1};
}

// Emit a tile loop (or single tile body) with proper offset computation.
// The emitBody callback receives (builder, location, tileOffset) where
// tileOffset is an i32 linear tile index computed from loop indices.
static void
emitTileLoop(ConversionPatternRewriter &rewriter, Location loc, int64_t tilesY,
             int64_t tilesX,
             llvm::function_ref<void(OpBuilder &, Location, Value)> emitBody) {
  if (tilesY > 1 || tilesX > 1) {
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto yBound = rewriter.create<arith::ConstantIndexOp>(loc, tilesY);
    auto xBound = rewriter.create<arith::ConstantIndexOp>(loc, tilesX);
    auto one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    auto tilesXVal = rewriter.create<arith::ConstantIndexOp>(loc, tilesX);

    scf::buildLoopNest(
        rewriter, loc, ValueRange{zero, zero}, ValueRange{yBound, xBound},
        ValueRange{one, one},
        [&](OpBuilder &b, Location bodyLoc, ValueRange ivs) {
          // Compute linear tile offset: offset = iy * tilesX + ix
          Value iy = ivs[0];
          Value ix = ivs[1];
          Value offsetY = b.create<arith::MulIOp>(bodyLoc, iy, tilesXVal);
          Value offset = b.create<arith::AddIOp>(bodyLoc, offsetY, ix);

          // Convert offset to i32 for TTKernel NOC operations
          Value offset32 =
              b.create<arith::IndexCastOp>(bodyLoc, b.getI32Type(), offset);

          emitBody(b, bodyLoc, offset32);
        });
  } else {
    // Single tile: offset is always 0
    Value zero32 = rewriter.create<arith::ConstantIntOp>(loc, 0, 32);
    emitBody(rewriter, loc, zero32);
  }
}

static LogicalResult lowerTensorToCB(CopyOp op, Value srcAccessor,
                                     ConversionPatternRewriter &rewriter,
                                     const TypeConverter &typeConverter) {
  auto loc = op.getLoc();

  auto [tilesY, tilesX] = getTileGridShapeFromValue(op.getSrc());

  // TODO(ttl): Plumb real NOC coordinates and bank base addresses from tensor
  // accessors and kernel launch metadata. Issue: #84.
  auto nocDst = makeZeroI32(loc, rewriter);

  emitTileLoop(rewriter, loc, tilesY, tilesX,
               [&](OpBuilder &b, Location bodyLoc, Value tileOffset) {
                 // TODO(ttl): Add lowering for CB protocol ops
                 // (reserve/push/wait/pop) once those ops are exposed in the
                 // TTL dialect and wired through to TTKernel. Issue: #78.
                 b.create<ttk::NocAsyncReadTileOp>(bodyLoc, tileOffset,
                                                   srcAccessor, nocDst);
               });

  // Encode direction in the handle type (async-token-like design).
  (void)typeConverter.convertType(
      TransferHandleType::get(rewriter.getContext(), TransferKind::read));
  // TODO(ttl): When TRID-aware TTKernel ops are available, pass a real TRID
  // instead of the zero placeholder here. Issue: #87.
  auto handle = makeZeroI32(loc, rewriter);
  rewriter.replaceOp(op, handle);
  return success();
}

static LogicalResult lowerCBToTensor(CopyOp op, Value dstAccessor,
                                     ConversionPatternRewriter &rewriter,
                                     const TypeConverter &typeConverter) {
  auto loc = op.getLoc();

  auto [tilesY, tilesX] = getTileGridShapeFromValue(op.getDst());

  // TODO(ttl): Lower CB operands to real CB handles and NOC addresses.
  // Issue: #80.
  auto tkCbTy = typeConverter.convertType(op.getSrc().getType());
  auto cbVal = emitPlaceholderCB(ValueRange{makeZeroI32(loc, rewriter)},
                                 rewriter, loc, tkCbTy);
  (void)cbVal;

  auto nocDst = makeZeroI32(loc, rewriter);

  emitTileLoop(rewriter, loc, tilesY, tilesX,
               [&](OpBuilder &b, Location bodyLoc, Value tileOffset) {
                 // TODO(ttl): Add lowering for CB protocol ops
                 // (reserve/push/wait/pop) once those ops are exposed in the
                 // TTL dialect and wired through to TTKernel. Issue: #78.
                 b.create<ttk::NocAsyncWriteTileOp>(bodyLoc, tileOffset,
                                                    dstAccessor, nocDst);
               });

  (void)typeConverter.convertType(
      TransferHandleType::get(rewriter.getContext(), TransferKind::write));
  // TODO(ttl): When TRID-aware TTKernel ops are available, pass a real TRID
  // instead of the zero placeholder here. Issue: #87.
  auto handle = makeZeroI32(loc, rewriter);
  rewriter.replaceOp(op, handle);
  return success();
}

struct CopyLowering : OpConversionPattern<CopyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter = this->getTypeConverter();
    if (!typeConverter) {
      return rewriter.notifyMatchFailure(op, "no type converter");
    }

    auto convertOperand = [&](Value v) -> FailureOr<Value> {
      if (auto acc = materializeTensorAccessor(v, rewriter); succeeded(acc)) {
        return *acc;
      }

      Type targetTy = typeConverter->convertType(v.getType());
      if (!targetTy) {
        return failure();
      }
      if (targetTy == v.getType()) {
        return v;
      }
      auto cast =
          rewriter.create<UnrealizedConversionCastOp>(op.getLoc(), targetTy, v);
      return cast.getResult(0);
    };

    FailureOr<Value> convertedSrc = convertOperand(adaptor.getSrc());
    if (failed(convertedSrc)) {
      return rewriter.notifyMatchFailure(op, "failed to convert src type");
    }
    FailureOr<Value> convertedDst = convertOperand(adaptor.getDst());
    if (failed(convertedDst)) {
      return rewriter.notifyMatchFailure(op, "failed to convert dst type");
    }

    auto srcKind = classifySrc(*convertedSrc);
    auto dstKind = classifyDst(*convertedDst);

    // Tensor accessor -> CB
    if (srcKind == CopySourceKind::TensorAccessor &&
        dstKind == CopyDestKind::CircularBuffer) {
      return lowerTensorToCB(op, *convertedSrc, rewriter, *typeConverter);
    }

    // CB -> Tensor accessor
    if (srcKind == CopySourceKind::CircularBuffer &&
        dstKind == CopyDestKind::TensorAccessor) {
      return lowerCBToTensor(op, *convertedDst, rewriter, *typeConverter);
    }

    // Unsupported pairs for now.
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << "unsupported ttl.copy src/dst combination: src="
           << op.getSrc().getType() << " dst=" << op.getDst().getType();
    });
  }
};

struct WaitLowering : OpConversionPattern<WaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO(ttl): Lower ttl.wait to TRID-specific barriers keyed by the transfer
    // handle (read vs write barrier based on transfer direction). Issue: #87.
    //
    // MVP behavior: require a direction-typed handle and emit the
    // corresponding global barrier. Untyped handles are rejected by the
    // verifier, but we also fail the rewrite defensively.
    auto kind = getTransferKindFromHandleType(adaptor.getXf().getType());
    if (!kind) {
      return rewriter.notifyMatchFailure(
          op, "requires direction-typed !ttl.transfer_handle<read|write>");
    }
    if (*kind == TransferKind::read) {
      rewriter.create<ttk::NocAsyncReadBarrierOp>(op.getLoc());
    } else if (*kind == TransferKind::write) {
      rewriter.create<ttk::NocAsyncWriteBarrierOp>(op.getLoc());
    } else {
      // Future-proofing: TransferKind is currently {read, write}, but fail
      // explicitly if it ever expands without updating the lowering.
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "unsupported TransferKind for ttl.wait lowering";
      });
    }
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Tile Op Lowering Patterns (ttl.tile_* -> ttkernel.*)
//===----------------------------------------------------------------------===//

// Template for lowering binary tile ops to TTKernel SFPU ops.
// SFPU binary ops: DST[odst] = DST[src0] op DST[src1]
template <typename TTLTileOp, typename TTKernelOp, typename TTKernelInitOp>
struct LowerTileBinaryOp : OpRewritePattern<TTLTileOp> {
  using OpRewritePattern<TTLTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TTLTileOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // TODO: Get DST indices from:
    // - src0/src1: Track which DST registers the operands are in
    // - odst: Use dst_idx attribute from the op
    // For MVP, hardcode: src0=0, src1=1, odst=0 (in-place on src0).
    Value src0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value src1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value odst = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Emit init operation to configure compute unit.
    rewriter.create<TTKernelInitOp>(loc);

    // Emit the SFPU binary operation (operates in-place, no results).
    rewriter.create<TTKernelOp>(loc, src0, src1, odst);

    // TTKernel ops have no results; the TTL tile op result is now implicit.
    // Erase the TTL op since its result is tracked via DST register.
    rewriter.eraseOp(op);
    return success();
  }
};

// Template for lowering unary tile ops to TTKernel SFPU ops.
// Unary SFPU ops: DST[dst_idx] = op(DST[dst_idx])
template <typename TTLTileOp, typename TTKernelOp, typename TTKernelInitOp>
struct LowerTileUnaryOp : OpRewritePattern<TTLTileOp> {
  using OpRewritePattern<TTLTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TTLTileOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // TODO: Get dst_idx from attribute and use for both src and dst.
    // For now, hardcode to 0.
    Value dstIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Emit init operation.
    rewriter.create<TTKernelInitOp>(loc);

    // Emit the SFPU unary operation (in-place, no results).
    rewriter.create<TTKernelOp>(loc, dstIdx);

    // Erase the TTL op since its result is now implicit in DST register.
    rewriter.eraseOp(op);
    return success();
  }
};

// Generate pattern type aliases from TTLToTTKernelOps.def
#define TTL_TILE_BINARY_SFPU_TO_TTKERNEL(TTL_OP, TTKERNEL_OP, TTKERNEL_INIT)   \
  using Lower##TTL_OP =                                                        \
      LowerTileBinaryOp<TTL_OP, ttk::TTKERNEL_OP, ttk::TTKERNEL_INIT>;
#define TTL_TILE_UNARY_TO_TTKERNEL(TTL_OP, TTKERNEL_OP, TTKERNEL_INIT)         \
  using Lower##TTL_OP =                                                        \
      LowerTileUnaryOp<TTL_OP, ttk::TTKERNEL_OP, ttk::TTKERNEL_INIT>;
#include "ttlang/Dialect/TTL/TTLToTTKernelOps.def"

//===----------------------------------------------------------------------===//
// ComputeOp Lowering Pattern
//===----------------------------------------------------------------------===//

struct LowerComputeOp : OpRewritePattern<ComputeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ComputeOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // TODO: Implement full lowering with:
    // 1. tile_regs_acquire()
    // 2. copy_tile from input CBs to DST (using dst_idx_map if present)
    // 3. Inline tile ops from body and lower them
    // 4. tile_regs_commit() + tile_regs_wait()
    // 5. pack_tile from DST to output CB
    // 6. tile_regs_release()

    rewriter.setInsertionPoint(op);
    rewriter.create<ttk::TileRegsAcquireOp>(loc);

    // TODO: Copy input tiles from CBs to DST registers.
    // TODO: Inline and lower tile ops from the body region.

    rewriter.setInsertionPointAfter(op);
    rewriter.create<ttk::TileRegsCommitOp>(loc);
    rewriter.create<ttk::TileRegsWaitOp>(loc);

    // TODO: Pack result tiles from DST to output CBs.

    rewriter.create<ttk::TileRegsReleaseOp>(loc);

    // Don't erase the compute op yet since we haven't fully lowered the body.
    // TODO: Erase after implementing body inlining and lowering.
    return success();
  }
};

struct FuncKernelFinalize : OpRewritePattern<FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter &rewriter) const override {
    if (!isNocKernel(op.getOperation())) {
      return failure();
    }

    // Change ttl.kernel_thread attribute to ttkernel.thread
    if (auto threadAttr =
            op->getAttrOfType<ttk::ThreadTypeAttr>("ttl.kernel_thread")) {
      op->removeAttr("ttl.kernel_thread");
      op->setAttr("ttkernel.thread", threadAttr);
    }

    // If function has arguments, we need to transform them
    if (op.getNumArguments() > 0) {
      // Build arg_spec attribute for compile-time arguments
      // Tensor arguments become buffer_address compile-time args
      llvm::SmallVector<ttk::ArgAttr> ctArgSpecs;
      unsigned operandIndex = 0;
      for (auto arg : op.getArguments()) {
        if (llvm::isa<RankedTensorType>(arg.getType())) {
          auto argAttr = ttk::ArgAttr::get(
              op.getContext(), ttk::ArgType::BufferAddress, operandIndex++);
          ctArgSpecs.push_back(argAttr);
        }
      }

      // Set arg_spec attribute if we have any arguments
      if (!ctArgSpecs.empty()) {
        auto argSpecAttr =
            ttk::ArgSpecAttr::get(op.getContext(),
                                  /*rtArgs=*/ArrayRef<ttk::ArgAttr>{},
                                  /*ctArgs=*/ctArgSpecs);
        op->setAttr("ttkernel.arg_spec", argSpecAttr);
      }

      // Only erase arguments that are now unused after conversion. If any are
      // still used (e.g., until full accessor materialization is wired), keep
      // them to avoid invalid IR.
      eraseUnusedArguments(op);
    }

    return success();
  }
};

struct TTLConvertTTLToTTKernelPass
    : impl::TTLConvertTTLToTTKernelBase<TTLConvertTTLToTTKernelPass> {
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    ModuleOp mod = getOperation();

    TTLToTTKernelTypeConverter typeConverter;

    ConversionTarget target(ctx);
    target.addIllegalDialect<tt::ttl::TTLDialect>();
    target.addLegalDialect<arith::ArithDialect, BuiltinDialect, scf::SCFDialect,
                           func::FuncDialect, tensor::TensorDialect,
                           ttkernel::TTKernelDialect>();

    // Mark compute-related ops as legal during partial conversion since they're
    // handled by the separate greedy rewrite phase (populateTTLComputeToTTKernelPatterns).
    target.addLegalOp<ComputeOp, YieldOp>();
    // Tile ops (handled by greedy phase):
    target.addLegalOp<
        // Binary tile ops
        AddTileOp, SubTileOp, MulTileOp, MaxTileOp,
        // Unary tile ops
        ExpTileOp, LogTileOp, SqrtTileOp, RsqrtTileOp, TanhTileOp,
        SigmoidTileOp, AbsTileOp, NegTileOp, ReluTileOp>();

    target.addDynamicallyLegalOp<ModuleOp>([&](ModuleOp op) {
      return typeConverter.isLegal(&op.getBodyRegion());
    });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    // WaitOp will be lowered; do not mark it legal.

    RewritePatternSet patterns(&ctx);
    patterns.add<CreateCBLowering, CopyLowering, WaitLowering>(typeConverter,
                                                               &ctx);
    populateFunctionOpInterfaceTypeConversionPattern(
        func::FuncOp::getOperationName(), patterns, typeConverter);

    FrozenRewritePatternSet frozen(std::move(patterns));
    std::string diagMessage;
    if (utils::applyPartialConversionWithDiag(mod, target, frozen, getName(),
                                              diagMessage)) {
      mod.emitError() << diagMessage;
      signalPassFailure();
    }

    // Apply post-conversion cleanup patterns (e.g., barrier deduplication).
    RewritePatternSet cleanupPatterns(&ctx);
    ttkernel::populateTTKernelCleanupPatterns(cleanupPatterns);
    if (failed(applyPatternsGreedily(mod, std::move(cleanupPatterns)))) {
      signalPassFailure();
    }

    // Lower ttl.compute and tile ops to TTKernel ops using DialectConversion.
    // This properly handles nested regions (tile ops inside compute body).
    ConversionTarget computeTarget(ctx);
    // TTKernel ops are legal (target dialect)
    computeTarget.addLegalDialect<ttkernel::TTKernelDialect>();
    // Arith ops are legal (used for index constants)
    computeTarget.addLegalDialect<arith::ArithDialect>();
    // Mark compute-related TTL ops as illegal (must be converted)
    computeTarget.addIllegalOp<ComputeOp, YieldOp>();
    computeTarget.addIllegalOp<
        // Binary tile ops
        AddTileOp, SubTileOp, MulTileOp, MaxTileOp,
        // Unary tile ops
        ExpTileOp, LogTileOp, SqrtTileOp, RsqrtTileOp, TanhTileOp,
        SigmoidTileOp, AbsTileOp, NegTileOp, ReluTileOp>();
    // Other dialects are legal (func, tensor, etc.)
    computeTarget.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet computePatterns(&ctx);
    populateTTLComputeToTTKernelPatterns(computePatterns);
    if (failed(applyPartialConversion(mod, computeTarget,
                                       std::move(computePatterns)))) {
      signalPassFailure();
      return;
    }

    // Apply FuncKernelFinalize as a greedy rewrite after tile lowering.
    RewritePatternSet finalizePatterns(&ctx);
    finalizePatterns.add<FuncKernelFinalize>(&ctx);
    if (failed(applyPatternsGreedily(mod, std::move(finalizePatterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
