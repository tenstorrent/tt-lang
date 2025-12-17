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

// Build CB analysis state (block arguments and attached tensors).
CopyTileCBState buildCopyTileCBState(Operation *root) {
  CopyTileCBState state;
  root->walk([&](ComputeOp compute) {
    for (auto it : llvm::enumerate(compute.getInputs())) {
      if (auto attach = it.value().template getDefiningOp<AttachCBOp>()) {
        BlockArgument barg = compute.getBody().front().getArgument(it.index());
        state.blockArgToCb.try_emplace(barg, attach.getCb());
      }
    }
  });
  root->walk([&](AttachCBOp attach) {
    state.tensorToCb.try_emplace(attach.getResult(), attach.getCb());
    state.tensorToCb.try_emplace(attach.getTensor(), attach.getCb());
  });
  return state;
}

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
    // Preserve transfer handle types so ttl.wait can inspect transfer
    // direction. TRID-aware lowering will be added later.
    addConversion([](TransferHandleType t) -> Type { return t; });
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

/// Get the function argument index for a tensor value.
/// Returns the index if the tensor is a block argument of an entry block,
/// otherwise returns failure. Used to map tensors to runtime args.
static FailureOr<unsigned> getTensorFuncArgIndex(Value tensor) {
  auto blockArg = llvm::dyn_cast<BlockArgument>(tensor);
  if (!blockArg) {
    return failure();
  }
  Block *block = blockArg.getParentBlock();
  if (!block || !block->isEntryBlock()) {
    return failure();
  }
  return blockArg.getArgNumber();
}

/// Get the L1 buffer address from runtime args for a tensor function argument.
/// Runtime args are indexed by the tensor's function argument position.
static FailureOr<Value>
getBufferAddressFromRuntimeArg(Value tensor, Location loc,
                               ConversionPatternRewriter &rewriter) {
  auto argIdx = getTensorFuncArgIndex(tensor);
  if (failed(argIdx)) {
    return failure();
  }
  auto idxConst = rewriter.create<arith::ConstantIndexOp>(loc, *argIdx);
  return rewriter
      .create<ttk::GetCommonArgValOp>(loc, rewriter.getI32Type(), idxConst)
      .getResult();
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

/// Convert TTL CircularBufferType to TTKernel CBType.
static ttk::CBType convertToKernelCBType(CircularBufferType ttlCb) {
  return ttk::CBType::get(ttlCb.getContext(), ttlCb.getTotalElements(),
                          ttlCb.getElementType());
}

struct BindCBLowering : OpConversionPattern<BindCBOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(BindCBOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto ttlCbType =
        mlir::dyn_cast<CircularBufferType>(op.getResult().getType());
    if (!ttlCbType) {
      return rewriter.notifyMatchFailure(op,
                                         "result is not CircularBufferType");
    }

    // Convert to TTKernel CB type.
    auto cbType = convertToKernelCBType(ttlCbType);

    // Get the CB index from the bind_cb op attribute.
    int64_t cbIndex = op.getCbIndex().getSExtValue();
    if (cbIndex < 0 || cbIndex >= kMaxCircularBuffers) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "cb_index " << cbIndex << " out of valid range [0, "
             << kMaxCircularBuffers - 1 << "]";
      });
    }

    // Create ttkernel.get_compile_time_arg_val to get the CB handle.
    auto getArgVal = rewriter.create<ttk::GetCompileArgValOp>(
        op.getLoc(), cbType, static_cast<int32_t>(cbIndex));

    // Cast back to TTL CB type for downstream ops that still expect it.
    auto cast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), op.getResult().getType(), ValueRange{getArgVal});
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CB synchronization operation lowering patterns
//===----------------------------------------------------------------------===//

// Trace through unrealized casts to get the original TTL CB type.
static CircularBufferType getTTLCBType(Value cb) {
  if (auto ttlCbTy = mlir::dyn_cast<CircularBufferType>(cb.getType())) {
    return ttlCbTy;
  }
  if (auto castOp = cb.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (castOp.getInputs().size() == 1) {
      if (auto ttlCbTy = mlir::dyn_cast<CircularBufferType>(
              castOp.getInputs()[0].getType())) {
        return ttlCbTy;
      }
    }
  }
  return nullptr;
}

static FailureOr<Value>
convertCBOperand(Value cb, ConversionPatternRewriter &rewriter, Location loc) {
  if (mlir::isa<ttk::CBType>(cb.getType())) {
    return cb;
  }
  auto ttlCbTy = mlir::dyn_cast<CircularBufferType>(cb.getType());
  if (!ttlCbTy) {
    return failure();
  }
  Type tkCbTy =
      ttk::CBType::get(ttlCbTy.getContext(), ttlCbTy.getTotalElements(),
                       ttlCbTy.getElementType());
  auto cast = rewriter.create<UnrealizedConversionCastOp>(loc, tkCbTy, cb);
  return cast.getResult(0);
}

// num_pages = product of CB shape dimensions (elements per block).
static Value computeNumPages(Value cb, ConversionPatternRewriter &rewriter,
                             Location loc) {
  auto ttlCbTy = getTTLCBType(cb);
  int64_t numPages = ttlCbTy ? ttlCbTy.getElementsPerBlock() : 1;
  return rewriter.create<arith::ConstantIntOp>(loc, numPages, 32);
}

template <typename SourceOp, typename TargetOp, bool HasResult>
struct CBOpLowering : OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    Value originalCb = op.getCb();
    auto ttlCbTy = getTTLCBType(originalCb);
    if (!ttlCbTy) {
      return rewriter.notifyMatchFailure(op, "failed to get TTL CB type");
    }

    auto convertedCb = convertCBOperand(adaptor.getCb(), rewriter, loc);
    if (failed(convertedCb)) {
      return rewriter.notifyMatchFailure(op, "failed to convert CB operand");
    }

    Value numPages = computeNumPages(originalCb, rewriter, loc);
    rewriter.create<TargetOp>(loc, *convertedCb, numPages);

    if constexpr (HasResult) {
      auto viewCast = rewriter.create<UnrealizedConversionCastOp>(
          loc, op.getResult().getType(), *convertedCb);
      rewriter.replaceOp(op, viewCast.getResult(0));
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }
};

using CBReserveLowering =
    CBOpLowering<CBReserveOp, ttk::CBReserveBackOp, /*HasResult=*/true>;
using CBPushLowering =
    CBOpLowering<CBPushOp, ttk::CBPushBackOp, /*HasResult=*/false>;
using CBWaitLowering =
    CBOpLowering<CBWaitOp, ttk::CBWaitFrontOp, /*HasResult=*/true>;
using CBPopLowering =
    CBOpLowering<CBPopOp, ttk::CBPopFrontOp, /*HasResult=*/false>;

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

/// Create a TensorAccessor from a tensor type and bank base address.
/// The bankBase should come from runtime args via
/// getBufferAddressFromRuntimeArg.
static FailureOr<Value>
materializeTensorAccessor(Value tensor, Value bankBase,
                          ConversionPatternRewriter &rewriter) {
  auto tensorTy = llvm::dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorTy) {
    return failure();
  }

  auto loc = tensor.getLoc();
  utils::ContiguousLayoutInfo layout = utils::computeContiguousLayout(tensorTy);

  auto rowStride =
      rewriter.create<arith::ConstantIntOp>(loc, layout.rowStrideElems, 32);
  auto colStride =
      rewriter.create<arith::ConstantIntOp>(loc, layout.colStrideElems, 32);
  auto pageSize =
      rewriter.create<arith::ConstantIntOp>(loc, layout.pageSizeBytes, 32);

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

/// Lower tensor->CB copy: read from DRAM/L1 tensor into circular buffer.
static LogicalResult lowerTensorToCB(CopyOp op, Value srcTensor, Value dstCB,
                                     ConversionPatternRewriter &rewriter,
                                     const TypeConverter &typeConverter) {
  auto loc = op.getLoc();

  // Get tensor L1 address from runtime args.
  auto bankBase = getBufferAddressFromRuntimeArg(srcTensor, loc, rewriter);
  if (failed(bankBase)) {
    return rewriter.notifyMatchFailure(
        op, "tensor must be a function argument for runtime arg mapping");
  }

  // Create tensor accessor with actual buffer address.
  auto srcAccessor = materializeTensorAccessor(srcTensor, *bankBase, rewriter);
  if (failed(srcAccessor)) {
    return rewriter.notifyMatchFailure(op, "failed to create tensor accessor");
  }

  // Convert CB to TTKernel type and get write pointer.
  auto cbConverted = convertCBOperand(dstCB, rewriter, loc);
  if (failed(cbConverted)) {
    return rewriter.notifyMatchFailure(op, "failed to convert CB operand");
  }
  auto cbWritePtr = rewriter.create<ttk::GetWritePtrOp>(loc, *cbConverted);

  auto [tilesY, tilesX] = getTileGridShapeFromValue(srcTensor);

  // TODO(#138): Emit single block transfer for contiguous layouts instead of
  // tile loop.
  emitTileLoop(rewriter, loc, tilesY, tilesX,
               [&](OpBuilder &b, Location bodyLoc, Value tileOffset) {
                 b.create<ttk::NocAsyncReadTileOp>(bodyLoc, tileOffset,
                                                   *srcAccessor, cbWritePtr);
               });

  auto handle = makeZeroI32(loc, rewriter);
  rewriter.replaceOp(op, handle);
  return success();
}

/// Lower CB->tensor copy: write from circular buffer to DRAM/L1 tensor.
static LogicalResult lowerCBToTensor(CopyOp op, Value srcCB, Value dstTensor,
                                     ConversionPatternRewriter &rewriter,
                                     const TypeConverter &typeConverter) {
  auto loc = op.getLoc();

  // Get tensor L1 address from runtime args.
  auto bankBase = getBufferAddressFromRuntimeArg(dstTensor, loc, rewriter);
  if (failed(bankBase)) {
    return rewriter.notifyMatchFailure(
        op, "tensor must be a function argument for runtime arg mapping");
  }

  // Create tensor accessor with actual buffer address.
  auto dstAccessor = materializeTensorAccessor(dstTensor, *bankBase, rewriter);
  if (failed(dstAccessor)) {
    return rewriter.notifyMatchFailure(op, "failed to create tensor accessor");
  }

  // Convert CB to TTKernel type and get read pointer.
  auto cbConverted = convertCBOperand(srcCB, rewriter, loc);
  if (failed(cbConverted)) {
    return rewriter.notifyMatchFailure(op, "failed to convert CB operand");
  }
  auto cbReadPtr = rewriter.create<ttk::GetReadPtrOp>(loc, *cbConverted);

  auto [tilesY, tilesX] = getTileGridShapeFromValue(dstTensor);

  // TODO(#138): Emit single block transfer for contiguous layouts instead of
  // tile loop.
  emitTileLoop(rewriter, loc, tilesY, tilesX,
               [&](OpBuilder &b, Location bodyLoc, Value tileOffset) {
                 b.create<ttk::NocAsyncWriteTileOp>(bodyLoc, tileOffset,
                                                    *dstAccessor, cbReadPtr);
               });

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

    // Use original operands for classification since lowering functions
    // handle type conversion internally.
    Value src = op.getSrc();
    Value dst = op.getDst();
    auto srcKind = classifySrc(src);
    auto dstKind = classifyDst(dst);

    // Tensor -> CB: read from tensor into circular buffer.
    if (srcKind == CopySourceKind::TensorAccessor &&
        dstKind == CopyDestKind::CircularBuffer) {
      return lowerTensorToCB(op, src, adaptor.getDst(), rewriter,
                             *typeConverter);
    }

    // CB -> Tensor: write from circular buffer to tensor.
    if (srcKind == CopySourceKind::CircularBuffer &&
        dstKind == CopyDestKind::TensorAccessor) {
      return lowerCBToTensor(op, adaptor.getSrc(), dst, rewriter,
                             *typeConverter);
    }

    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << "unsupported ttl.copy src/dst combination: src=" << src.getType()
           << " dst=" << dst.getType();
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
    // handled by the separate greedy rewrite phase
    // (populateTTLTileOpsToTTKernelPatterns).
    target.addLegalOp<ComputeOp, YieldOp, AttachCBOp>();
    // Tile ops (handled by tile ops phase later):
    target.addLegalOp<
        // Binary tile ops
        AddTileOp, SubTileOp, MulTileOp, MaxTileOp,
        // Unary tile ops
        ExpTileOp, LogTileOp, SqrtTileOp, RsqrtTileOp, TanhTileOp,
        SigmoidTileOp, AbsTileOp, NegTileOp, ReluTileOp,
        // Copy tile op
        CopyTileOp>();

    target.addDynamicallyLegalOp<ModuleOp>([&](ModuleOp op) {
      return typeConverter.isLegal(&op.getBodyRegion());
    });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType()) &&
             typeConverter.isLegal(&op.getBody());
    });
    // WaitOp will be lowered; do not mark it legal.

    RewritePatternSet patterns(&ctx);
    patterns.add<BindCBLowering, CopyLowering, WaitLowering, CBReserveLowering,
                 CBPushLowering, CBWaitLowering, CBPopLowering>(typeConverter,
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

    // Lower tile ops to TTKernel ops using DialectConversion. ttl.compute is
    // kept legal here because full compute lowering happens after loops and
    // bufferization in a later stage.
    ConversionTarget computeTarget(ctx);
    // TTKernel ops are legal (target dialect)
    computeTarget.addLegalDialect<ttkernel::TTKernelDialect>();
    // Arith ops are legal (used for index constants)
    computeTarget.addLegalDialect<arith::ArithDialect>();
    // Keep compute ops legal (tile-only lowering here).
    computeTarget.addLegalOp<ComputeOp, YieldOp>();
    // Other dialects are legal (func, tensor, etc.) EXCEPT tile ops.
    computeTarget.markUnknownOpDynamicallyLegal(
        [](Operation *) { return true; });
    // Mark tile ops as illegal so they get converted.
    computeTarget.addIllegalOp<
        // Binary tile ops
        AddTileOp, SubTileOp, MulTileOp, MaxTileOp,
        // Unary tile ops
        ExpTileOp, LogTileOp, SqrtTileOp, RsqrtTileOp, TanhTileOp,
        SigmoidTileOp, AbsTileOp, NegTileOp, ReluTileOp,
        // Copy tile op
        CopyTileOp>();

    auto cbState = buildCopyTileCBState(mod);

    RewritePatternSet computePatterns(&ctx);
    populateTTLTileOpsToTTKernelPatterns(&typeConverter, &cbState,
                                         computePatterns);
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
