// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h" // IWYU pragma: keep

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
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

// Sentinel values for transfer direction (CopyLowering -> WaitLowering).
constexpr int32_t kTransferReadSentinel = 0;
constexpr int32_t kTransferWriteSentinel = 1;

// Runtime arg counter for tensor copies (reset per function).
static thread_local int64_t gRuntimeArgIndex = 0;
static void resetRuntimeArgCounter() { gRuntimeArgIndex = 0; }
static int64_t getNextRuntimeArgIndex() { return gRuntimeArgIndex++; }

//===----------------------------------------------------------------------===//
// CB Type Utilities
//===----------------------------------------------------------------------===//

static bool isCircularBufferType(Type t) {
  return llvm::isa<CircularBufferType>(t) || llvm::isa<ttk::CBType>(t);
}

static ttk::CBType convertCBTypeToKernel(CircularBufferType cb) {
  return ttk::CBType::get(cb.getContext(), cb.getTotalElements(),
                          cb.getElementType());
}

/// Convert CB value to TTKernel CB type, tracing through casts if needed.
static Value convertCBToKernelType(Value cb,
                                   ConversionPatternRewriter &rewriter) {
  auto srcType = cb.getType();

  if (llvm::isa<ttk::CBType>(srcType)) {
    return cb;
  }

  if (auto ttlCB = llvm::dyn_cast<CircularBufferType>(srcType)) {
    auto targetType = convertCBTypeToKernel(ttlCB);
    // Avoid redundant cast chains.
    if (auto cast = cb.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (cast.getInputs().size() == 1 &&
          cast.getInputs()[0].getType() == targetType) {
        return cast.getInputs()[0];
      }
    }
    return rewriter
        .create<UnrealizedConversionCastOp>(cb.getLoc(), targetType, cb)
        .getResult(0);
  }

  // Trace through unrealized_conversion_cast to find TTKernel CB.
  if (auto cast = cb.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() == 1) {
      return convertCBToKernelType(cast.getInputs()[0], rewriter);
    }
  }

  return nullptr;
}

class TTLToTTKernelTypeConverter : public TypeConverter {
public:
  TTLToTTKernelTypeConverter() {
    // Identity conversion last (addConversion prepends).
    addConversion([](Type t) { return t; });

    // TransferHandle -> i32 (direction encoded as sentinel).
    addConversion([](TransferHandleType t) -> Type {
      return IntegerType::get(t.getContext(), 32);
    });

    // Tensors stay as-is (L1 address from runtime args).
    addConversion([](RankedTensorType t) -> Type { return t; });

    // CB types handled explicitly by patterns (no auto-conversion).
    auto castMaterialization = [](OpBuilder &builder, Type resultType,
                                  ValueRange inputs, Location loc) -> Value {
      if (isCircularBufferType(resultType)) {
        return nullptr;
      }
      if (inputs.size() == 1 && isCircularBufferType(inputs[0].getType())) {
        return nullptr;
      }
      return builder.create<UnrealizedConversionCastOp>(loc, resultType, inputs)
          .getResult(0);
    };
    addSourceMaterialization(castMaterialization);
    addTargetMaterialization(castMaterialization);
  }
};

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

//===----------------------------------------------------------------------===//
// CB synchronization operation lowering patterns
//===----------------------------------------------------------------------===//

struct CBReserveLowering : OpConversionPattern<CBReserveOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CBReserveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto cb = convertCBToKernelType(adaptor.getCb(), rewriter);
    if (!cb) {
      return rewriter.notifyMatchFailure(op, "failed to convert CB type");
    }

    rewriter.create<ttk::CBReserveBackOp>(loc, cb, adaptor.getNumPages());

    // Return tensor view via cast from CB for downstream ops.
    auto viewCast = rewriter.create<UnrealizedConversionCastOp>(
        loc, op.getResult().getType(), cb);
    rewriter.replaceOp(op, viewCast.getResult(0));
    return success();
  }
};

struct CBPushLowering : OpConversionPattern<CBPushOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CBPushOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto cb = convertCBToKernelType(adaptor.getCb(), rewriter);
    if (!cb) {
      return rewriter.notifyMatchFailure(op, "failed to convert CB type");
    }

    rewriter.create<ttk::CBPushBackOp>(op.getLoc(), cb, adaptor.getNumPages());
    rewriter.eraseOp(op);
    return success();
  }
};

struct CBWaitLowering : OpConversionPattern<CBWaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CBWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto cb = convertCBToKernelType(adaptor.getCb(), rewriter);
    if (!cb) {
      return rewriter.notifyMatchFailure(op, "failed to convert CB type");
    }

    rewriter.create<ttk::CBWaitFrontOp>(loc, cb, adaptor.getNumPages());

    // Return tensor view via cast from CB for downstream ops.
    auto viewCast = rewriter.create<UnrealizedConversionCastOp>(
        loc, op.getResult().getType(), cb);
    rewriter.replaceOp(op, viewCast.getResult(0));
    return success();
  }
};

struct CBPopLowering : OpConversionPattern<CBPopOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CBPopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto cb = convertCBToKernelType(adaptor.getCb(), rewriter);
    if (!cb) {
      return rewriter.notifyMatchFailure(op, "failed to convert CB type");
    }

    rewriter.create<ttk::CBPopFrontOp>(op.getLoc(), cb, adaptor.getNumPages());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Copy operation lowering
//===----------------------------------------------------------------------===//

enum class CopySourceKind { TensorAccessor, CircularBuffer, Unknown };
enum class CopyDestKind { TensorAccessor, CircularBuffer, Unknown };

static bool isTensorLike(Type t) {
  return llvm::isa<ttk::TensorAccessorType>(t) ||
         llvm::isa<RankedTensorType>(t);
}

static CopySourceKind classifySrc(Value v) {
  if (isCircularBufferType(v.getType())) {
    return CopySourceKind::CircularBuffer;
  }
  if (isTensorLike(v.getType())) {
    return CopySourceKind::TensorAccessor;
  }
  return CopySourceKind::Unknown;
}

static CopyDestKind classifyDst(Value v) {
  if (isCircularBufferType(v.getType())) {
    return CopyDestKind::CircularBuffer;
  }
  if (isTensorLike(v.getType())) {
    return CopyDestKind::TensorAccessor;
  }
  return CopyDestKind::Unknown;
}

static std::optional<TransferKind> getTransferKindFromHandleType(Type t) {
  auto transferHandle = llvm::dyn_cast<TransferHandleType>(t);
  if (!transferHandle) {
    return std::nullopt;
  }
  return transferHandle.getKind();
}

/// Lower tensor->CB copy using runtime args for L1 address.
static LogicalResult lowerTensorToCB(CopyOp op, Value cb,
                                     ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();

  // Get L1 address from runtime args (populated by host at kernel launch).
  int64_t rtArgIdx = getNextRuntimeArgIndex();
  auto rtArgIndexVal = rewriter.create<arith::ConstantIndexOp>(loc, rtArgIdx);
  auto l1Addr = rewriter.create<ttk::GetCommonArgValOp>(
      loc, rewriter.getI32Type(), rtArgIndexVal);

  // Get CB write pointer for destination.
  auto cbWritePtr = rewriter.create<ttk::GetWritePtrOp>(loc, cb);

  // NOC coordinates: hardcoded for MVP (physical core 0,0 = NOC 18,18).
  auto nocX = rewriter.create<arith::ConstantIndexOp>(loc, 18);
  auto nocY = rewriter.create<arith::ConstantIndexOp>(loc, 18);
  auto nocAddr =
      rewriter.create<ttk::GetNocAddrOp>(loc, nocX, nocY, l1Addr.getResult());

  // Tile size: 32x32 bf16 = 2048 bytes (MVP hardcoded).
  auto tileSize = rewriter.create<arith::ConstantIntOp>(loc, 2048, 32);

  rewriter.create<ttk::NocAsyncReadOp>(loc, nocAddr, cbWritePtr, tileSize);

  // Return sentinel indicating read direction.
  auto sentinel =
      rewriter.create<arith::ConstantIntOp>(loc, kTransferReadSentinel, 32);
  rewriter.replaceOp(op, sentinel.getResult());
  return success();
}

/// Lower CB->tensor copy using runtime args for L1 address.
static LogicalResult lowerCBToTensor(CopyOp op, Value cb,
                                     ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();

  // Get L1 address from runtime args.
  int64_t rtArgIdx = getNextRuntimeArgIndex();
  auto rtArgIndexVal = rewriter.create<arith::ConstantIndexOp>(loc, rtArgIdx);
  auto l1Addr = rewriter.create<ttk::GetCommonArgValOp>(
      loc, rewriter.getI32Type(), rtArgIndexVal);

  // Get CB read pointer for source.
  auto cbReadPtr = rewriter.create<ttk::GetReadPtrOp>(loc, cb);

  // NOC coordinates: hardcoded for MVP.
  auto nocX = rewriter.create<arith::ConstantIndexOp>(loc, 18);
  auto nocY = rewriter.create<arith::ConstantIndexOp>(loc, 18);
  auto nocAddr =
      rewriter.create<ttk::GetNocAddrOp>(loc, nocX, nocY, l1Addr.getResult());

  // Tile size: 32x32 bf16 = 2048 bytes (MVP hardcoded).
  auto tileSize = rewriter.create<arith::ConstantIntOp>(loc, 2048, 32);

  rewriter.create<ttk::NocAsyncWriteOp>(loc, cbReadPtr, nocAddr, tileSize);

  // Return sentinel indicating write direction.
  auto sentinel =
      rewriter.create<arith::ConstantIntOp>(loc, kTransferWriteSentinel, 32);
  rewriter.replaceOp(op, sentinel.getResult());
  return success();
}

struct CopyLowering : OpConversionPattern<CopyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcKind = classifySrc(adaptor.getSrc());
    auto dstKind = classifyDst(adaptor.getDst());

    // Tensor -> CB: read from tensor to CB.
    if (srcKind == CopySourceKind::TensorAccessor &&
        dstKind == CopyDestKind::CircularBuffer) {
      auto cb = convertCBToKernelType(adaptor.getDst(), rewriter);
      if (!cb) {
        return rewriter.notifyMatchFailure(op, "failed to convert dst CB");
      }
      return lowerTensorToCB(op, cb, rewriter);
    }

    // CB -> Tensor: write from CB to tensor.
    if (srcKind == CopySourceKind::CircularBuffer &&
        dstKind == CopyDestKind::TensorAccessor) {
      auto cb = convertCBToKernelType(adaptor.getSrc(), rewriter);
      if (!cb) {
        return rewriter.notifyMatchFailure(op, "failed to convert src CB");
      }
      return lowerCBToTensor(op, cb, rewriter);
    }

    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << "unsupported copy combination: src=" << op.getSrc().getType()
           << " dst=" << op.getDst().getType();
    });
  }
};

//===----------------------------------------------------------------------===//
// Wait operation lowering
//===----------------------------------------------------------------------===//

struct WaitLowering : OpConversionPattern<WaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto xf = adaptor.getXf();

    // Check for sentinel constant from CopyLowering.
    if (auto constOp = xf.getDefiningOp<arith::ConstantIntOp>()) {
      int64_t value = constOp.value();
      if (value == kTransferReadSentinel) {
        rewriter.create<ttk::NocAsyncReadBarrierOp>(op.getLoc());
      } else if (value == kTransferWriteSentinel) {
        rewriter.create<ttk::NocAsyncWriteBarrierOp>(op.getLoc());
      } else {
        return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
          diag << "unknown transfer sentinel: " << value;
        });
      }
      rewriter.eraseOp(op);
      return success();
    }

    // Fallback: check original type for direction.
    auto kind = getTransferKindFromHandleType(op.getXf().getType());
    if (!kind) {
      return rewriter.notifyMatchFailure(
          op, "requires direction-typed handle or sentinel constant");
    }
    if (*kind == TransferKind::read) {
      rewriter.create<ttk::NocAsyncReadBarrierOp>(op.getLoc());
    } else if (*kind == TransferKind::write) {
      rewriter.create<ttk::NocAsyncWriteBarrierOp>(op.getLoc());
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported transfer direction");
    }
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Function CB args conversion
//===----------------------------------------------------------------------===//

/// Convert CB function arguments to get_compile_time_arg_val calls.
/// Tensor args are left in place (CopyLowering uses runtime args for L1 addr).
struct FuncCBArgsToGetArgVal : OpConversionPattern<FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(FuncOp op, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    // Only process functions with TTKernel thread attribute.
    if (!op->hasAttr(ttk::ThreadTypeAttr::name)) {
      return failure();
    }

    // Check if there are any CB arguments to convert.
    bool hasCBArgs = false;
    for (auto argType : op.getArgumentTypes()) {
      if (llvm::isa<CircularBufferType>(argType)) {
        hasCBArgs = true;
        break;
      }
    }
    if (!hasCBArgs) {
      return failure();
    }

    // Reset runtime arg counter for this function.
    resetRuntimeArgCounter();

    Block *block = &op.getCallableRegion()->front();
    unsigned numArgs = block->getNumArguments();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(block);

    // Track CB index (only CBs get compile-time arg indices).
    unsigned cbIndex = 0;
    SmallVector<Type> newArgTypes;
    llvm::BitVector argsToRemove(numArgs);

    for (unsigned i = 0; i < numArgs; ++i) {
      auto arg = block->getArgument(i);
      auto argType = arg.getType();

      if (auto ttlCBType = llvm::dyn_cast<CircularBufferType>(argType)) {
        // CB arg: replace with get_compile_time_arg_val.
        auto kernelCBType = convertCBTypeToKernel(ttlCBType);
        auto kernelCB = rewriter.create<ttk::GetCompileArgValOp>(
            op.getLoc(), kernelCBType, rewriter.getI32IntegerAttr(cbIndex++));

        // Cast back to !ttl.cb for downstream TTL ops.
        auto ttlCB = rewriter.create<UnrealizedConversionCastOp>(
            op.getLoc(), ttlCBType, kernelCB.getResult());
        arg.replaceAllUsesWith(ttlCB.getResult(0));
        argsToRemove.set(i);
      } else {
        // Keep non-CB args (tensor args stay, CopyLowering handles them).
        newArgTypes.push_back(argType);
      }
    }

    // Erase CB arguments (reverse order to maintain indices).
    for (int i = numArgs - 1; i >= 0; --i) {
      if (argsToRemove.test(i)) {
        block->eraseArgument(i);
      }
    }

    // Update function type to reflect removed CB args.
    rewriter.modifyOpInPlace(op, [&]() {
      op.setType(rewriter.getFunctionType(newArgTypes, TypeRange()));
    });
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
    target.addLegalDialect<BuiltinDialect>();
    target.addLegalDialect<ttk::TTKernelDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addDynamicallyLegalOp<ModuleOp>([&](ModuleOp op) {
      return typeConverter.isLegal(&op.getBodyRegion());
    });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      // Kernel functions must have CB args converted (tensor args stay).
      if (op->hasAttr(ttk::ThreadTypeAttr::name)) {
        for (auto argType : op.getArgumentTypes()) {
          if (llvm::isa<CircularBufferType>(argType)) {
            return false;
          }
        }
      }
      return typeConverter.isLegal(&op.getBody());
    });

    RewritePatternSet patterns(&ctx);
    patterns.add<FuncCBArgsToGetArgVal>(typeConverter, &ctx);
    patterns
        .add<CreateCBLowering, CopyLowering, WaitLowering, CBReserveLowering,
             CBPushLowering, CBWaitLowering, CBPopLowering>(typeConverter,
                                                            &ctx);

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
