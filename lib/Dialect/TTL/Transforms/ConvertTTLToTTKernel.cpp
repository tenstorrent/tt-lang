// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h" // IWYU pragma: keep

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
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
    // Transfer handle: keep as-is for now (identity).
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
  auto tensorTy = mlir::cast<RankedTensorType>(tensor.getType());
  utils::ContiguousLayoutInfo layout;
  if (auto enc = tensorTy.getEncoding()) {
    if (auto ttnnLayout = mlir::dyn_cast<tt::ttnn::TTNNLayoutAttr>(enc)) {
      // TODO(ttl): Derive strides/page size/bank base from TTNNLayoutAttr.
      // Issue: #81.
      (void)ttnnLayout;
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

static LogicalResult lowerTensorToCB(CopyOp op, Value srcAccessor,
                                     ConversionPatternRewriter &rewriter,
                                     const TypeConverter &typeConverter) {
  auto loc = op.getLoc();

  // TODO(ttl): Plumb real NOC coordinates and bank base addresses from tensor
  // accessors and kernel launch metadata.
  // Issue: #84.
  auto nocSrc = makeZeroI32(loc, rewriter);
  auto nocDst = makeZeroI32(loc, rewriter);

  // Per TT-Lang spec, ttl.copy is non-blocking: it initiates the async DMA
  // transfer and returns a handle that is synchronized by ttl.wait.
  //
  // TODO(ttl): Compute the destination NOC address from the CB operand and the
  // requested tile coordinates (when we model CB handles and CB addressing).
  // Issue: #80.
  rewriter.create<ttk::NocAsyncReadTileOp>(loc, nocSrc, srcAccessor, nocDst);

  // Encode direction in the handle type (async-token-like design).
  auto handleTy = typeConverter.convertType(
      TransferHandleType::get(rewriter.getContext(), TransferKind::read));
  // TODO(ttl): Replace the placeholder unrealized cast with a real transfer
  // handle value (TRID or equivalent). Issue: #87.
  auto handle = rewriter.create<UnrealizedConversionCastOp>(
      loc, handleTy, ValueRange{makeZeroI32(loc, rewriter)});
  rewriter.replaceOp(op, handle.getResult(0));
  return success();
}

static LogicalResult lowerCBToTensor(CopyOp op, Value dstAccessor,
                                     ConversionPatternRewriter &rewriter,
                                     const TypeConverter &typeConverter) {
  auto loc = op.getLoc();

  // TODO(ttl): Lower CB operands to real CB handles and NOC addresses.
  // Issue: #80.
  auto tkCbTy = typeConverter.convertType(op.getSrc().getType());
  auto cbVal = emitPlaceholderCB(ValueRange{makeZeroI32(loc, rewriter)},
                                 rewriter, loc, tkCbTy);
  (void)cbVal;

  // Per TT-Lang spec, ttl.copy is non-blocking: it initiates the async DMA
  // transfer and returns a handle that is synchronized by ttl.wait. This is
  // lowered to a TTKernel NOC async write.
  rewriter.create<ttk::NocAsyncWriteTileOp>(
      loc, makeZeroI32(loc, rewriter), dstAccessor, makeZeroI32(loc, rewriter));

  auto handleTy = typeConverter.convertType(
      TransferHandleType::get(rewriter.getContext(), TransferKind::write));
  // TODO(ttl): Replace the placeholder unrealized cast with a real transfer
  // handle value (TRID or equivalent). Issue: #87.
  auto handle = rewriter.create<UnrealizedConversionCastOp>(
      loc, handleTy, ValueRange{makeZeroI32(loc, rewriter)});
  rewriter.replaceOp(op, handle.getResult(0));
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

struct FuncKernelFinalize : OpRewritePattern<FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter & /*rewriter*/) const override {
    if (!isNocKernel(op.getOperation())) {
      return failure();
    }
    return eraseUnusedArguments(op) ? success() : failure();
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
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<ttk::TTKernelDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
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
    patterns.add<FuncKernelFinalize>(&ctx);

    FrozenRewritePatternSet frozen(std::move(patterns));
    std::string diagMessage;
    if (utils::applyPartialConversionWithDiag(mod, target, frozen, getName(),
                                              diagMessage)) {
      mod.emitError() << diagMessage;
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
