// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h" // IWYU pragma: keep

#include "DFBAllocationLimits.h"
#include "PipeGraph.h"
#include "PipeLowering.h"
#include "PipePlanning.h"
#include "PipeTransferExpansion.h"
#include "ttlang/Dialect/TTKernel/Transforms/TTKernelCleanupPatterns.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsEnums.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Transforms/ComputeTarget.h"
#include "ttlang/Dialect/TTL/Transforms/PipeTransferAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/TransferProvenance.h"
#include "ttlang/Dialect/Utils/ConversionUtils.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <utility>
#include <variant>

namespace mlir::tt::ttl {
#define GEN_PASS_DEF_TTLCONVERTTTLTOTTKERNEL
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

using mlir::func::FuncOp;
namespace ttk = mlir::tt::ttkernel;

// Maps local args to global tensor indices for common runtime args (buffer
// addresses). CRTA is filtered per-thread, containing only addresses for
// tensors this thread uses.
constexpr llvm::StringLiteral kCRTAIndicesAttr = "ttl.crta_indices";
constexpr llvm::StringLiteral kExpandLinearizeIndexAttr =
    "ttlang.expand_linearize_index";
// Duplicating up to four callback bodies avoids table lookups for small nets.
// Larger nets use one loop so the transfer protocol body is not duplicated for
// every record.
constexpr size_t kPipeNetForeachDirectRecordLimit = 4;

// PipeGraph is defined in PipeGraph.h.

class TTLToTTKernelTypeConverter : public TypeConverter {
public:
  TTLToTTKernelTypeConverter() {
    // TypeConverter invokes the most recently registered applicable callback.
    addConversion([](Type type) { return type; });

    // Layout-encoded tensors remain available to CopyLowering until their
    // runtime TensorAccessor has been materialized. Other tensors recursively
    // convert their element type so tensors of pipe tokens remain legal.
    addConversion([this](RankedTensorType t) -> Type {
      if (t.getEncoding() && mlir::isa<tt::ttl::LayoutAttr>(t.getEncoding())) {
        return t;
      }
      Type convertedElementType = convertType(t.getElementType());
      if (!convertedElementType || convertedElementType == t.getElementType()) {
        return t;
      }
      return RankedTensorType::get(t.getShape(), convertedElementType,
                                   t.getEncoding());
    });

    // CB: lower to TTKernel CB type with flattened element count.
    addConversion([](CircularBufferType t) -> Type {
      return ttk::CBType::get(t.getContext(), t.getTotalElements(),
                              t.getElementType());
    });
    addConversion([](PipeTokenType type) -> Type {
      return IntegerType::get(type.getContext(), 32);
    });
    // Public pipe copies expose TransferHandleType and may preserve the dynamic
    // post sequence through SCF or tensor containers. DMA handles use the same
    // runtime representation, but their waits depend only on precomputed
    // provenance and lower to barriers without inspecting this value.
    addConversion([](TransferHandleType type) -> Type {
      return IntegerType::get(type.getContext(), 32);
    });

    auto castMaterialization = [](OpBuilder &builder, Type resultType,
                                  ValueRange inputs, Location loc) -> Value {
      return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                inputs)
          .getResult(0);
    };
    addSourceMaterialization(castMaterialization);
    addTargetMaterialization(castMaterialization);
  }
};

static Value createConvertedTensorOp(tensor::EmptyOp op,
                                     tensor::EmptyOp::Adaptor adaptor,
                                     Type convertedType,
                                     ConversionPatternRewriter &rewriter) {
  return tensor::EmptyOp::create(rewriter, op.getLoc(), convertedType,
                                 adaptor.getDynamicSizes());
}

static Value createConvertedTensorOp(tensor::InsertOp op,
                                     tensor::InsertOp::Adaptor adaptor,
                                     Type convertedType,
                                     ConversionPatternRewriter &rewriter) {
  return tensor::InsertOp::create(rewriter, op.getLoc(), convertedType,
                                  adaptor.getScalar(), adaptor.getDest(),
                                  adaptor.getIndices());
}

static Value createConvertedTensorOp(tensor::ExtractOp op,
                                     tensor::ExtractOp::Adaptor adaptor,
                                     Type convertedType,
                                     ConversionPatternRewriter &rewriter) {
  return tensor::ExtractOp::create(rewriter, op.getLoc(), convertedType,
                                   adaptor.getTensor(), adaptor.getIndices());
}

static Value createConvertedTensorOp(tensor::CastOp op,
                                     tensor::CastOp::Adaptor adaptor,
                                     Type convertedType,
                                     ConversionPatternRewriter &rewriter) {
  return tensor::CastOp::create(rewriter, op.getLoc(), convertedType,
                                adaptor.getSource());
}

template <typename TensorOp>
struct TensorOpTypeConversion : OpConversionPattern<TensorOp> {
  using OpConversionPattern<TensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TensorOp op, typename TensorOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType = this->getTypeConverter()->convertType(op.getType());
    if (!convertedType) {
      return rewriter.notifyMatchFailure(op, "failed to convert result type");
    }
    rewriter.replaceOp(
        op, createConvertedTensorOp(op, adaptor, convertedType, rewriter));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Helper utilities.
//===----------------------------------------------------------------------===//

/// Convert ttl.kernel_thread -> ttkernel.thread if present, returning the
/// resolved thread type from whichever attribute exists.
static std::optional<ttk::ThreadType> convertThreadAttr(Operation *op) {
  if (auto a =
          op->getAttrOfType<ttk::ThreadTypeAttr>(ttk::ThreadTypeAttr::name)) {
    return a.getValue();
  }
  if (auto a = op->getAttrOfType<ttk::ThreadTypeAttr>(kKernelThreadAttrName)) {
    op->removeAttr(kKernelThreadAttrName);
    op->setAttr(ttk::ThreadTypeAttr::name, a);
    return a.getValue();
  }
  return std::nullopt;
}

struct ExpandMarkedLinearizeIndex
    : OpRewritePattern<affine::AffineLinearizeIndexOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineLinearizeIndexOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasAttr(kExpandLinearizeIndexAttr)) {
      return failure();
    }
    return affine::lowerAffineLinearizeIndexOp(rewriter, op);
  }
};

/// Get the function argument index used to map a tensor to runtime arguments.
/// A region block argument is rejected because its position is unrelated to
/// the enclosing kernel function signature.
static FailureOr<unsigned> getTensorFuncArgIndex(Value tensor) {
  auto blockArg = llvm::dyn_cast<BlockArgument>(tensor);
  if (!blockArg) {
    return failure();
  }
  Block *block = blockArg.getParentBlock();
  auto func = block ? dyn_cast<func::FuncOp>(block->getParentOp()) : nullptr;
  if (!func || func.isDeclaration() || block != &func.getBody().front()) {
    return failure();
  }
  return blockArg.getArgNumber();
}

static FailureOr<int32_t> getValidatedDFBIndex(Value dfb, Operation *op) {
  std::optional<int64_t> dfbIndex = getCBIndex(dfb);
  if (!dfbIndex) {
    return op->emitError("cannot resolve finalized DFB index");
  }
  int32_t targetMaxDFBIndices = getTargetMaxDFBIndices(op);
  if (*dfbIndex < 0 || *dfbIndex >= targetMaxDFBIndices) {
    return op->emitError("finalized DFB index ")
           << *dfbIndex << " is outside [0, " << targetMaxDFBIndices - 1
           << "] for " << getTargetDFBIndexCapacityDescription(op);
  }
  return static_cast<int32_t>(*dfbIndex);
}

/// Read one L1 address from the function's common runtime arguments.
static Value getCommonRuntimeArg(unsigned argIdx, Location loc,
                                 ConversionPatternRewriter &rewriter) {
  auto idxConst = arith::ConstantIndexOp::create(rewriter, loc, argIdx);
  return ttk::GetCommonArgValOp::create(rewriter, loc, rewriter.getI32Type(),
                                        idxConst)
      .getResult();
}

/// Build a TensorAccessor using tt-metal's constexpr CTA offset chaining.
///
/// The CTA offset for tensor N is computed at device compile time via
/// get_tensor_accessor_args_cta_offset<N, baseCTA>(). This chains through
/// all preceding tensors' configs to find the correct offset, regardless of
/// whether each tensor is interleaved (2 CTAs) or sharded (variable CTAs).
static Value buildTensorAccessor(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 int32_t baseCTA, int32_t globalTensorIdx,
                                 int32_t crtaIndex, Value bankBase,
                                 Value pageSize) {
  std::string ctaExpr =
      "tensor_accessor::detail::get_tensor_accessor_args_cta_offset<" +
      std::to_string(globalTensorIdx) + ", " + std::to_string(baseCTA) + ">()";

  // Verifier requires cta_base even when cta_expr is set; EmitC ignores it.
  auto dummyCTA = arith::ConstantIntOp::create(rewriter, loc, 0, 32);
  auto crtaConst = arith::ConstantIntOp::create(rewriter, loc, crtaIndex, 32);
  auto args = ttk::TensorAccessorArgsOp::create(
      rewriter, loc, dummyCTA.getResult(), crtaConst.getResult(),
      /*prev_args=*/Value(), rewriter.getStringAttr(ctaExpr),
      /*crta_expr=*/nullptr);
  auto accessor = ttk::TensorAccessorOp::create(rewriter, loc, args.getResult(),
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
    auto cbType =
        ttk::CBType::get(ttlCbType.getContext(), ttlCbType.getTotalElements(),
                         ttlCbType.getElementType());

    // Get the CB index from the bind_cb op attribute.
    int64_t cbIndex = op.getCbIndex().getSExtValue();
    int32_t targetMaxDFBIndices = getTargetMaxDFBIndices(op);
    if (cbIndex < 0 || cbIndex >= targetMaxDFBIndices) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "cb_index " << cbIndex << " out of valid range [0, "
             << targetMaxDFBIndices - 1 << "] for "
             << getTargetDFBIndexCapacityDescription(op);
      });
    }

    // Create ttkernel.get_compile_time_arg_val to get the CB handle.
    auto getArgVal = ttk::GetCompileArgValOp::create(
        rewriter, op.getLoc(), cbType, static_cast<int32_t>(cbIndex));

    // Cast back to TTL CB type for downstream ops that still expect it.
    auto cast = UnrealizedConversionCastOp::create(
        rewriter, op.getLoc(), op.getResult().getType(), ValueRange{getArgVal});
    rewriter.replaceOp(op, cast.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CB synchronization operation lowering patterns
//===----------------------------------------------------------------------===//

// Tile count: use the `num_tiles` attribute if present (per-subblock
// reserve/push), otherwise derive from the DFB type shape (full block).
static Value computeNumTiles(Operation *sourceOp, CircularBufferType dfbType,
                             ConversionPatternRewriter &rewriter,
                             Location loc) {
  if (auto attr = sourceOp->getAttrOfType<IntegerAttr>("num_tiles")) {
    return arith::ConstantIntOp::create(rewriter, loc, attr.getInt(), 32);
  }
  return arith::ConstantIntOp::create(rewriter, loc,
                                      dfbType.getElementsPerBlock(), 32);
}

template <typename SourceOp, typename TargetOp, bool HasResult>
struct CBOpLowering : OpConversionPattern<SourceOp> {
  CBOpLowering(const TypeConverter &typeConverter, MLIRContext *context,
               const PipeTransportPlan &pipeTransportPlan)
      : OpConversionPattern<SourceOp>(typeConverter, context),
        pipeTransportPlan(pipeTransportPlan) {}

  LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (pipeTransportPlan.ownsDFBLifecycle(op.getOperation())) {
      if constexpr (HasResult) {
        auto convertedCb =
            utils::convertTTLCBToTTKernel(adaptor.getCb(), rewriter, loc);
        if (failed(convertedCb)) {
          return rewriter.notifyMatchFailure(op,
                                             "failed to convert DFB operand");
        }
        auto viewCast = UnrealizedConversionCastOp::create(
            rewriter, loc, op.getResult().getType(), *convertedCb);
        rewriter.replaceOp(op, viewCast.getResult(0));
      } else {
        rewriter.eraseOp(op);
      }
      return success();
    }

    Value originalCb = op.getCb();
    FailureOr<CircularBufferType> maybeDFBType =
        utils::getTTLCircularBufferType(originalCb);
    if (failed(maybeDFBType)) {
      return rewriter.notifyMatchFailure(op, "failed to get TTL CB type");
    }

    auto convertedCb =
        utils::convertTTLCBToTTKernel(adaptor.getCb(), rewriter, loc);
    if (failed(convertedCb)) {
      return rewriter.notifyMatchFailure(op, "failed to convert CB operand");
    }

    Value numTiles = computeNumTiles(op, *maybeDFBType, rewriter, loc);
    TargetOp::create(rewriter, loc, *convertedCb, numTiles);

    if constexpr (HasResult) {
      auto viewCast = UnrealizedConversionCastOp::create(
          rewriter, loc, op.getResult().getType(), *convertedCb);
      rewriter.replaceOp(op, viewCast.getResult(0));
    } else {
      rewriter.eraseOp(op);
    }
    return success();
  }

private:
  const PipeTransportPlan &pipeTransportPlan;
};

using CBReserveLowering =
    CBOpLowering<CBReserveOp, ttk::CBReserveBackOp, /*HasResult=*/true>;
using CBPushLowering =
    CBOpLowering<CBPushOp, ttk::CBPushBackOp, /*HasResult=*/false>;
using CBWaitLowering =
    CBOpLowering<CBWaitOp, ttk::CBWaitFrontOp, /*HasResult=*/true>;

struct CBPopLowering : OpConversionPattern<CBPopOp> {
  CBPopLowering(const TypeConverter &typeConverter, MLIRContext *context,
                const PipeCapacityPlan &pipeCapacityPlan,
                const PipeTransportPlan &pipeTransportPlan,
                const PipeTransportSlotCounterMap &slotCounters,
                const PipeResourcePlan &pipeResourcePlan)
      : OpConversionPattern(typeConverter, context),
        pipeCapacityPlan(pipeCapacityPlan),
        pipeTransportPlan(pipeTransportPlan), slotCounters(slotCounters),
        pipeResourcePlan(pipeResourcePlan) {}

  LogicalResult
  matchAndRewrite(CBPopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerCBPop(op, adaptor.getCb(), pipeCapacityPlan, pipeTransportPlan,
                      slotCounters, pipeResourcePlan, rewriter);
  }

private:
  const PipeCapacityPlan &pipeCapacityPlan;
  const PipeTransportPlan &pipeTransportPlan;
  const PipeTransportSlotCounterMap &slotCounters;
  const PipeResourcePlan &pipeResourcePlan;
};

/// Trace back from a view value to the underlying TTKernel CB.
/// Traverses ViewLikeOpInterface ops (CBReserveOp, CBWaitOp) and casts.
static FailureOr<Value> getCBFromView(Value v) {
  while (v) {
    if (llvm::isa<ttk::CBType>(v.getType())) {
      return v;
    }

    Operation *def = v.getDefiningOp();
    if (!def) {
      break;
    }

    if (auto viewLike = llvm::dyn_cast<ViewLikeOpInterface>(def)) {
      v = viewLike.getViewSource();
      continue;
    }

    if (auto cast = llvm::dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (cast.getInputs().size() == 1) {
        v = cast.getInputs()[0];
        continue;
      }
    }

    if (auto cast = llvm::dyn_cast<tensor::CastOp>(def)) {
      v = cast.getSource();
      continue;
    }

    // Trace through tensor.extract_slice (from compute subblocking).
    if (auto slice = llvm::dyn_cast<tensor::ExtractSliceOp>(def)) {
      v = slice.getSource();
      continue;
    }

    // Trace through ttl.attach_cb to get the DFB operand.
    if (auto attach = llvm::dyn_cast<AttachCBOp>(def)) {
      v = attach.getCb();
      continue;
    }

    break;
  }
  return failure();
}

/// Lower ttl.attach_cb to its input tensor.
/// After tile ops (including copy_tile) have been lowered and CB associations
/// have been used, attach_cb is purely metadata and can be erased. We replace
/// it with its input tensor to preserve SSA form.
struct AttachCBLowering : OpConversionPattern<AttachCBOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AttachCBOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the attach_cb result with its input tensor.
    // The CB association metadata has already been used by earlier lowerings.
    rewriter.replaceOp(op, adaptor.getTensor());
    return success();
  }
};

struct TileStoreLowering : OpConversionPattern<TileStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TileStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto cb = getCBFromView(adaptor.getView());
    if (failed(cb)) {
      // Adapted view may have lost the DFB chain (e.g., attach_cb already
      // converted). Trace the original (unconverted) view instead.
      Value origCB = getAttachedCB(op.getView());
      if (!origCB) {
        return rewriter.notifyMatchFailure(
            op, "view not associated with a dataflow buffer");
      }
      cb = utils::convertTTLCBToTTKernel(origCB, rewriter, loc,
                                         this->getTypeConverter());
      if (failed(cb)) {
        return rewriter.notifyMatchFailure(
            op, "could not convert dataflow buffer type");
      }
    }

    // Linearize multi-dimensional CB indices to a flat tile index.
    auto viewTy = mlir::cast<RankedTensorType>(op.getView().getType());
    ValueRange indices = adaptor.getIndices();
    Value cbTileIndex = affine::AffineLinearizeIndexOp::create(
        rewriter, loc, indices, viewTy.getShape());

    // If the view is a subblock slice, add the slice offset to produce
    // the global DFB tile index.
    cbTileIndex =
        utils::addSliceOffset(op.getView(), cbTileIndex, rewriter, loc);

    Value dstIndex = adaptor.getDstIndex();

    if (op.getStoreKind() == DFBTileStoreKind::ConsumerReplacement) {
      uint64_t acquiredTiles = static_cast<uint64_t>(
          cast<ttk::CBType>((*cb).getType()).getNumElements());
      ttk::PackWaitedTileOp::create(rewriter, loc, dstIndex, *cb, cbTileIndex,
                                    /*out_of_order=*/true, acquiredTiles);
    } else {
      ttk::PackTileOp::create(rewriter, loc, dstIndex, *cb, cbTileIndex,
                              /*out_of_order=*/true);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct DstIndexCleanup : OpConversionPattern<DstIndexOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DstIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getSource());
    return success();
  }
};

} // namespace

// PipeGraph implementation lives in PipeGraph.cpp.

namespace {

enum class CopyOperandKind {
  TensorSlice,
  CircularBuffer,
  Pipe,
  DFBAttachedTensor,
  Unknown
};

static CopyOperandKind classifyOperand(Value v) {
  if (llvm::isa<CircularBufferType>(v.getType())) {
    return CopyOperandKind::CircularBuffer;
  }
  if (llvm::isa<PipeType, SelectedPipeSrcType, SelectedPipeDstType>(
          v.getType())) {
    return CopyOperandKind::Pipe;
  }
  if (v.getDefiningOp<TensorSliceOp>()) {
    return CopyOperandKind::TensorSlice;
  }
  if (getAttachedCB(v)) {
    return CopyOperandKind::DFBAttachedTensor;
  }
  return CopyOperandKind::Unknown;
}

static Value makeZeroI32(Location loc, ConversionPatternRewriter &rewriter) {
  return arith::ConstantIntOp::create(rewriter, loc, 0, 32);
}

static std::optional<TransferKind> getTransferKindFromHandleType(Type t) {
  auto transferHandle = llvm::dyn_cast<TransferHandleType>(t);
  if (!transferHandle) {
    return std::nullopt;
  }
  return transferHandle.getKind();
}

static Value buildConstantTableLookup(OpBuilder &builder, Location loc,
                                      ArrayRef<int64_t> values,
                                      Value recordIndex) {
  assert(!values.empty() && "PipeNet foreach records must not be empty");
  return ttk::ConstantTableLookupOp::create(
      builder, loc, builder.getIndexType(), recordIndex,
      builder.getDenseI64ArrayAttr(values));
}

static bool shouldLowerPipeNetForeachDirect(PipeNetRecordsAttr records) {
  return records.getPipes().size() <= kPipeNetForeachDirectRecordLimit;
}

struct PipeForeachTables {
  SmallVector<int64_t> srcX;
  SmallVector<int64_t> srcY;
  SmallVector<int64_t> dstStartX;
  SmallVector<int64_t> dstStartY;
  SmallVector<int64_t> dstEndX;
  SmallVector<int64_t> dstEndY;
  SmallVector<int64_t> numDests;
  SmallVector<int64_t> srcInDstRange;
};

static PipeForeachTables buildPipeForeachTables(OpBuilder &builder,
                                                PipeNetRecordsAttr records) {
  SmallVector<int64_t> srcX;
  SmallVector<int64_t> srcY;
  SmallVector<int64_t> dstStartX;
  SmallVector<int64_t> dstStartY;
  SmallVector<int64_t> dstEndX;
  SmallVector<int64_t> dstEndY;
  SmallVector<int64_t> numDests;
  SmallVector<int64_t> srcInDstRange;
  MLIRContext *context = builder.getContext();
  for (PipeRecordAttr record : records.getPipes()) {
    PipeType pipeType =
        getPipeTypeFromRecord(context, record, records.getPipeNetId());
    srcX.push_back(pipeType.getSrcX());
    srcY.push_back(pipeType.getSrcY());
    dstStartX.push_back(pipeType.getDstStartX());
    dstStartY.push_back(pipeType.getDstStartY());
    dstEndX.push_back(pipeType.getDstEndX());
    dstEndY.push_back(pipeType.getDstEndY());
    numDests.push_back(pipeType.getNumDests());
    srcInDstRange.push_back(pipeType.srcInDstRange() ? 1 : 0);
  }
  return PipeForeachTables{std::move(srcX),      std::move(srcY),
                           std::move(dstStartX), std::move(dstStartY),
                           std::move(dstEndX),   std::move(dstEndY),
                           std::move(numDests),  std::move(srcInDstRange)};
}

template <typename SelectOp, typename SelectedType>
static SelectOp
buildSelectedPipe(OpBuilder &builder, Location loc, PipeNetRecordsAttr records,
                  const PipeForeachTables &tables, Value recordIndex) {
  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value srcInDstRangeIndex =
      buildConstantTableLookup(builder, loc, tables.srcInDstRange, recordIndex);
  Value srcInDstRange = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::ne, srcInDstRangeIndex, zero);
  return SelectOp::create(
      builder, loc, SelectedType::get(builder.getContext()), recordIndex,
      buildConstantTableLookup(builder, loc, tables.srcX, recordIndex),
      buildConstantTableLookup(builder, loc, tables.srcY, recordIndex),
      buildConstantTableLookup(builder, loc, tables.dstStartX, recordIndex),
      buildConstantTableLookup(builder, loc, tables.dstStartY, recordIndex),
      buildConstantTableLookup(builder, loc, tables.dstEndX, recordIndex),
      buildConstantTableLookup(builder, loc, tables.dstEndY, recordIndex),
      buildConstantTableLookup(builder, loc, tables.numDests, recordIndex),
      srcInDstRange, records);
}

template <typename ForeachOp>
static void clonePipeForeachBody(ForeachOp foreachOp, Value selectedPipe,
                                 OpBuilder &builder) {
  IRMapping mapping;
  Block &sourceBlock = foreachOp.getBody().front();
  mapping.map(sourceBlock.getArgument(0), selectedPipe);
  for (Operation &bodyOp : sourceBlock) {
    if (mlir::isa<YieldOp>(bodyOp)) {
      continue;
    }
    builder.clone(bodyOp, mapping);
  }
}

static Value buildIntegerMatch(RewriterBase &rewriter, Location loc, Value lhs,
                               int64_t rhs) {
  Value rhsValue = arith::ConstantIndexOp::create(rewriter, loc, rhs);
  return arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::eq, lhs,
                               rhsValue);
}

static Value buildIntegerRangeMatch(RewriterBase &rewriter, Location loc,
                                    Value value, int64_t start, int64_t end) {
  Value startValue = arith::ConstantIndexOp::create(rewriter, loc, start);
  Value endValue = arith::ConstantIndexOp::create(rewriter, loc, end);
  Value atStart = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::sge, value, startValue);
  Value atEnd = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sle,
                                      value, endValue);
  return arith::AndIOp::create(rewriter, loc, atStart, atEnd);
}

static Value buildRecordSrcMatch(RewriterBase &rewriter, Location loc,
                                 Value nodeX, Value nodeY,
                                 PipeRecordAttr record) {
  Value xMatches = buildIntegerMatch(rewriter, loc, nodeX, record.getSrcX());
  Value yMatches = buildIntegerMatch(rewriter, loc, nodeY, record.getSrcY());
  return arith::AndIOp::create(rewriter, loc, xMatches, yMatches);
}

static Value buildRecordDstMatch(RewriterBase &rewriter, Location loc,
                                 Value nodeX, Value nodeY,
                                 PipeRecordAttr record) {
  Value xMatches = buildIntegerRangeMatch(
      rewriter, loc, nodeX, record.getDstStartX(), record.getDstEndX());
  Value yMatches = buildIntegerRangeMatch(
      rewriter, loc, nodeY, record.getDstStartY(), record.getDstEndY());
  return arith::AndIOp::create(rewriter, loc, xMatches, yMatches);
}

static CreatePipeOp buildStaticPipeForRecord(RewriterBase &rewriter,
                                             Location loc,
                                             PipeNetRecordsAttr records,
                                             PipeRecordAttr record) {
  PipeType pipeType =
      getPipeTypeFromRecord(rewriter.getContext(), record,
                            static_cast<int64_t>(records.getPipeNetId()));
  BoolAttr isCollectiveAttr =
      record.getIsCollective() ? rewriter.getBoolAttr(true) : BoolAttr();
  return CreatePipeOp::create(
      rewriter, loc, pipeType, rewriter.getI64IntegerAttr(record.getSrcX()),
      rewriter.getI64IntegerAttr(record.getSrcY()),
      rewriter.getI64IntegerAttr(record.getDstStartX()),
      rewriter.getI64IntegerAttr(record.getDstStartY()),
      rewriter.getI64IntegerAttr(record.getDstEndX()),
      rewriter.getI64IntegerAttr(record.getDstEndY()),
      rewriter.getI64IntegerAttr(records.getPipeNetId()),
      records.getPipeNetName(), isCollectiveAttr);
}

template <typename ForeachOp>
static LogicalResult lowerPipeNetForeachDirect(
    ForeachOp op, RewriterBase &rewriter, PipeRole role,
    PipeForeachLoweringInfo &foreachLoweringInfo,
    llvm::function_ref<Value(RewriterBase &, Location, Value, Value,
                             PipeRecordAttr)>
        buildRecordMatch) {
  Location loc = op.getLoc();
  PipeNetRecordsAttr records = op.getRecords();
  rewriter.setInsertionPoint(op);
  Value nodeX =
      ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
  Value nodeY =
      ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());
  for (PipeRecordAttr record : records.getPipes()) {
    Value staticPipe =
        buildStaticPipeForRecord(rewriter, loc, records, record).getResult();
    Value isActiveRecord =
        buildRecordMatch(rewriter, loc, nodeX, nodeY, record);
    auto ifOp = scf::IfOp::create(rewriter, loc, isActiveRecord,
                                  /*withElseRegion=*/false);
    foreachLoweringInfo.controlOps.push_back(ifOp);
    foreachLoweringInfo.ifThenDomains[ifOp] =
        role == PipeRole::Source
            ? getPipeRecordSourceLaunchNodeDomain(record)
            : getPipeRecordDestinationLaunchNodeDomain(record);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    clonePipeForeachBody(op, staticPipe, rewriter);
    rewriter.setInsertionPointAfter(ifOp);
  }
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult
lowerPipeNetForeachSrc(PipeNetForeachSrcOp op, RewriterBase &rewriter,
                       PipeForeachLoweringInfo &foreachLoweringInfo) {
  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  PipeNetRecordsAttr records = op.getRecords();
  if (shouldLowerPipeNetForeachDirect(records)) {
    return lowerPipeNetForeachDirect(op, rewriter, PipeRole::Source,
                                     foreachLoweringInfo, buildRecordSrcMatch);
  }

  PipeForeachTables tables = buildPipeForeachTables(rewriter, records);
  Value lower = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value upper =
      arith::ConstantIndexOp::create(rewriter, loc, records.getPipes().size());
  Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto forOp = scf::ForOp::create(rewriter, loc, lower, upper, step);
  foreachLoweringInfo.recordLoops[forOp] =
      PipeNetRecordLoop{records, PipeNetRecordSelection::Source};

  rewriter.setInsertionPointToStart(forOp.getBody());
  Value recordIndex = forOp.getInductionVar();
  auto selectedPipe = buildSelectedPipe<SelectPipeSrcOp, SelectedPipeSrcType>(
      rewriter, loc, records, tables, recordIndex);
  Value nodeX =
      ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
  Value nodeY =
      ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());
  Value xMatches = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::eq, nodeX, selectedPipe.getSrcX());
  Value yMatches = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::eq, nodeY, selectedPipe.getSrcY());
  Value isSrc = arith::AndIOp::create(rewriter, loc, xMatches, yMatches);
  auto ifOp = scf::IfOp::create(rewriter, loc, isSrc,
                                /*withElseRegion=*/false);
  foreachLoweringInfo.controlOps.push_back(forOp);
  foreachLoweringInfo.controlOps.push_back(ifOp);
  foreachLoweringInfo.ifThenDomains[ifOp] =
      getPipeRecordsRoleLaunchNodeDomain(records, PipeRole::Source);
  rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  clonePipeForeachBody(op, selectedPipe.getPipe(), rewriter);
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult
lowerPipeNetForeachDst(PipeNetForeachDstOp op, RewriterBase &rewriter,
                       PipeForeachLoweringInfo &foreachLoweringInfo) {
  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  PipeNetRecordsAttr records = op.getRecords();
  if (shouldLowerPipeNetForeachDirect(records)) {
    return lowerPipeNetForeachDirect(op, rewriter, PipeRole::Destination,
                                     foreachLoweringInfo, buildRecordDstMatch);
  }

  PipeForeachTables tables = buildPipeForeachTables(rewriter, records);
  Value lower = arith::ConstantIndexOp::create(rewriter, loc, 0);
  Value upper =
      arith::ConstantIndexOp::create(rewriter, loc, records.getPipes().size());
  Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);
  auto forOp = scf::ForOp::create(rewriter, loc, lower, upper, step);
  foreachLoweringInfo.recordLoops[forOp] =
      PipeNetRecordLoop{records, PipeNetRecordSelection::Destination};

  rewriter.setInsertionPointToStart(forOp.getBody());
  Value recordIndex = forOp.getInductionVar();
  auto selectedPipe = buildSelectedPipe<SelectPipeDstOp, SelectedPipeDstType>(
      rewriter, loc, records, tables, recordIndex);
  Value nodeX =
      ttk::MyLogicalXOp::create(rewriter, loc, rewriter.getIndexType());
  Value nodeY =
      ttk::MyLogicalYOp::create(rewriter, loc, rewriter.getIndexType());
  Value xAtStart =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge, nodeX,
                            selectedPipe.getDstStartX());
  Value xAtEnd = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sle,
                                       nodeX, selectedPipe.getDstEndX());
  Value yAtStart =
      arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sge, nodeY,
                            selectedPipe.getDstStartY());
  Value yAtEnd = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sle,
                                       nodeY, selectedPipe.getDstEndY());
  Value xInRange = arith::AndIOp::create(rewriter, loc, xAtStart, xAtEnd);
  Value yInRange = arith::AndIOp::create(rewriter, loc, yAtStart, yAtEnd);
  Value isDst = arith::AndIOp::create(rewriter, loc, xInRange, yInRange);
  auto ifOp = scf::IfOp::create(rewriter, loc, isDst,
                                /*withElseRegion=*/false);
  foreachLoweringInfo.controlOps.push_back(forOp);
  foreachLoweringInfo.controlOps.push_back(ifOp);
  foreachLoweringInfo.ifThenDomains[ifOp] =
      getPipeRecordsRoleLaunchNodeDomain(records, PipeRole::Destination);
  rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
  clonePipeForeachBody(op, selectedPipe.getPipe(), rewriter);
  rewriter.eraseOp(op);
  return success();
}

static LogicalResult
lowerPipeNetForeachOps(ModuleOp mod,
                       PipeForeachLoweringInfo &foreachLoweringInfo) {
  // A module-wide greedy rewrite also deletes unrelated unused pure reads.
  // Rewrite only foreach operations so this expansion cannot change other IR.
  IRRewriter rewriter(mod.getContext());
  while (true) {
    Operation *foreachOp = nullptr;
    mod.walk<WalkOrder::PreOrder>([&](Operation *candidate) {
      if (!mlir::isa<PipeNetForeachSrcOp, PipeNetForeachDstOp>(candidate)) {
        return WalkResult::advance();
      }
      foreachOp = candidate;
      return WalkResult::interrupt();
    });
    if (!foreachOp) {
      return success();
    }

    // Lower an outer callback before its nested callbacks. The outer rewrite
    // clones its body, so any recorded control operations then remain in the
    // module and continue to identify the generated record selection.
    if (auto foreachSrcOp = mlir::dyn_cast<PipeNetForeachSrcOp>(foreachOp)) {
      if (failed(lowerPipeNetForeachSrc(foreachSrcOp, rewriter,
                                        foreachLoweringInfo))) {
        return failure();
      }
      continue;
    }
    if (failed(
            lowerPipeNetForeachDst(mlir::cast<PipeNetForeachDstOp>(foreachOp),
                                   rewriter, foreachLoweringInfo))) {
      return failure();
    }
  }
}

/// Compute CTA index for a tensor function argument.
/// Reads ttl.base_cta_index and ttl.crta_indices from parent function.
/// Returns the baseCTA (number of CBs) and global tensor index for a function
/// argument. These are used to build the constexpr CTA offset expression.
static FailureOr<std::pair<int32_t, int32_t>>
getBaseCTAAndGlobalTensorIdx(unsigned argIdx, Operation *op) {
  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc) {
    return op->emitError("operation must be inside a function");
  }

  auto baseCTAAttr =
      parentFunc->getAttrOfType<IntegerAttr>(kBaseCTAIndexAttrName);
  if (!baseCTAAttr) {
    return op->emitError("function missing ")
           << kBaseCTAIndexAttrName << " attribute";
  }

  auto crtaIndicesAttr = parentFunc->getAttrOfType<ArrayAttr>(kCRTAIndicesAttr);
  if (!crtaIndicesAttr) {
    return op->emitError("function missing ")
           << kCRTAIndicesAttr << " attribute";
  }

  if (argIdx >= crtaIndicesAttr.size()) {
    return op->emitError("argument index out of range for ")
           << kCRTAIndicesAttr;
  }

  int32_t baseCTA = static_cast<int32_t>(baseCTAAttr.getInt());
  int32_t globalTensorIdx = static_cast<int32_t>(
      mlir::cast<IntegerAttr>(crtaIndicesAttr[argIdx]).getInt());

  return std::make_pair(baseCTA, globalTensorIdx);
}

/// Validate TTLLayoutAttr encoding on a tensor and return the page size.
static FailureOr<int64_t> getValidatedPageSize(Value tensor, Operation *op) {
  auto tensorTy = llvm::dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorTy) {
    return op->emitError("expected RankedTensorType for tensor accessor");
  }

  auto layoutAttr =
      mlir::dyn_cast_or_null<tt::ttl::LayoutAttr>(tensorTy.getEncoding());
  if (!layoutAttr) {
    return op->emitError(
        "tensor must have ttl.layout encoding for accessor "
        "materialization; Python layer should reject tensors without layout");
  }

  // TTL layouts are always tiled. Compute page size from tile element type.
  auto tileType =
      mlir::dyn_cast<tt::ttcore::TileType>(layoutAttr.getElementType());
  if (!tileType) {
    return op->emitError("layout element type must be a TileType");
  }

  return tileType.getSizeBytes();
}

struct TensorAccessorInfo {
  unsigned argIdx = 0;
  int32_t baseCTA = 0;
  int32_t globalTensorIdx = 0;
  int64_t pageSizeBytes = 0;
};

static FailureOr<TensorAccessorInfo>
getTensorAccessorInfo(Value tensor, Operation *op,
                      ConversionPatternRewriter &rewriter) {
  FailureOr<int64_t> pageSizeBytes = getValidatedPageSize(tensor, op);
  if (failed(pageSizeBytes)) {
    return failure();
  }
  FailureOr<unsigned> argIdx = getTensorFuncArgIndex(tensor);
  if (failed(argIdx)) {
    return rewriter.notifyMatchFailure(
        op, "tensor must be a function argument for runtime arg mapping");
  }
  FailureOr<std::pair<int32_t, int32_t>> ctaInfo =
      getBaseCTAAndGlobalTensorIdx(*argIdx, op);
  if (failed(ctaInfo)) {
    return failure();
  }
  auto [baseCTA, globalTensorIdx] = *ctaInfo;
  return TensorAccessorInfo{*argIdx, baseCTA, globalTensorIdx, *pageSizeBytes};
}

/// Create a TensorAccessor after all validation checks that can fail have run.
static Value materializeTensorAccessor(Value tensor, Value bankBase,
                                       const TensorAccessorInfo &info,
                                       ConversionPatternRewriter &rewriter) {
  auto loc = tensor.getLoc();

  auto pageSize =
      arith::ConstantIntOp::create(rewriter, loc, info.pageSizeBytes, 32);

  return buildTensorAccessor(loc, rewriter, info.baseCTA, info.globalTensorIdx,
                             static_cast<int32_t>(info.argIdx), bankBase,
                             pageSize);
}

/// Extract tile grid shape from a Value with a static ranked tensor type.
/// Returns all dimensions of the tile grid for linearization.
static SmallVector<int64_t> getTileGridShapeFromValue(Value v) {
  auto tensorTy = llvm::dyn_cast<RankedTensorType>(v.getType());
  assert(tensorTy && "expected RankedTensorType");
  assert(tensorTy.hasStaticShape() && "expected static shape");
  assert(llvm::isa<ttcore::TileType>(tensorTy.getElementType()) &&
         "expected TileType element type");

  return SmallVector<int64_t>(tensorTy.getShape());
}

/// Emit a loop nest over the given dimension bounds (or invoke the body
/// directly when all bounds are 1). The callback receives the induction
/// variables as index-typed Values matching the rank of `tileBounds`.
static void emitTileLoop(
    OpBuilder &builder, Location loc, ArrayRef<int64_t> tileBounds,
    llvm::function_ref<void(OpBuilder &, Location, ValueRange)> emitBody) {
  auto zero = arith::ConstantIndexOp::create(builder, loc, 0);

  bool allOne = llvm::all_of(tileBounds,
                             [](int64_t dimension) { return dimension == 1; });
  if (allOne) {
    SmallVector<Value> zeros(tileBounds.size(), zero);
    emitBody(builder, loc, zeros);
    return;
  }

  auto one = arith::ConstantIndexOp::create(builder, loc, 1);
  SmallVector<Value> lbs(tileBounds.size(), zero);
  SmallVector<Value> ubs;
  SmallVector<Value> steps(tileBounds.size(), one);
  for (int64_t bound : tileBounds) {
    ubs.push_back(arith::ConstantIndexOp::create(builder, loc, bound));
  }

  scf::buildLoopNest(builder, loc, lbs, ubs, steps,
                     [&](OpBuilder &nestedBuilder, Location bodyLoc,
                         ValueRange inductionVars) {
                       emitBody(nestedBuilder, bodyLoc, inductionVars);
                     });
}

/// Direction of a tensor<->CB tile copy for NOC operations.
enum class NocCopyDirection { Read, Write };

/// Add the proven bounded-ring slot offset to a transport storage address.
static Value materializeTransportStorageAddress(
    CopyOp op, Value baseAddress, Value currentSlot,
    const PipeTransportStorageAccess &storageAccess,
    ConversionPatternRewriter &rewriter) {
  if (storageAccess.role == PipeTransportStorageRole::Source ||
      storageAccess.blockCount == 1) {
    return baseAddress;
  }

  assert(storageAccess.dynamicSlotCounterIndex &&
         storageAccess.blockCount > 1 && storageAccess.blockStrideBytes > 0 &&
         "invalid transport-owned destination storage calculation");

  Location loc = op.getLoc();
  Value blockStrideBytes = arith::ConstantIndexOp::create(
      rewriter, loc, storageAccess.blockStrideBytes);
  Value slotOffset =
      arith::MulIOp::create(rewriter, loc, currentSlot, blockStrideBytes);
  return arith::AddIOp::create(rewriter, loc, baseAddress, slotOffset);
}

/// Lower a tensor_slice<->CB copy in the given direction.
/// Read: tensor_slice -> CB (noc_async_read_tile, get_write_ptr)
/// Write: CB -> tensor_slice (noc_async_write_tile, get_read_ptr)
static LogicalResult lowerTensorCBCopy(
    CopyOp op, TensorSliceOp sliceOp, Value cb, NocCopyDirection direction,
    const PipeTransportStorageAccess *storageAccess,
    const PipeTransportSlotCounterMap &slotCounters,
    ConversionPatternRewriter &rewriter, const TypeConverter &typeConverter) {
  auto loc = op.getLoc();
  Value tensor = sliceOp.getTensor();
  auto startIndices = sliceOp.getIndices();

  FailureOr<TensorAccessorInfo> accessorInfo =
      getTensorAccessorInfo(tensor, op, rewriter);
  if (failed(accessorInfo)) {
    return failure();
  }

  FailureOr<CircularBufferType> maybeDFBType =
      utils::getTTLCircularBufferType(cb);
  if (failed(maybeDFBType)) {
    return rewriter.notifyMatchFailure(op, "failed to get CB type");
  }

  SmallVector<int64_t> tensorGridShape = getTileGridShapeFromValue(tensor);
  unsigned tensorRank = tensorGridShape.size();

  if (startIndices.size() != tensorRank) {
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << "tensor_slice index count (" << startIndices.size()
           << ") does not match tensor rank (" << tensorRank << ")";
    });
  }

  // cbRank <= tensorRank is guaranteed upstream: CopyOp enforces DFB rank ==
  // slice result rank, and TensorSliceOp enforces result rank <= tensor rank.
  auto transferTensorType = cast<RankedTensorType>(
      direction == NocCopyDirection::Read ? op.getSrc().getType()
                                          : op.getDst().getType());
  ArrayRef<int64_t> transferShape = transferTensorType.getShape();
  assert(transferShape.size() <= tensorRank &&
         "transfer tensor rank exceeds source tensor rank");

  Value bankBase = getCommonRuntimeArg(accessorInfo->argIdx, loc, rewriter);
  Value accessor =
      materializeTensorAccessor(tensor, bankBase, *accessorInfo, rewriter);

  bool isRead = direction == NocCopyDirection::Read;

  // Rank-reducing slice: the leading (tensorRank - cbRank) tensor dims are
  // squeezed via scalar indices (validated at slice creation). CB iteration
  // vars map to the trailing dims; squeezed dims contribute startIndices[d]
  // directly with no IV adder.
  unsigned cbRank = transferShape.size();
  unsigned rankDiff = tensorRank - cbRank;

  auto indexTy = rewriter.getIndexType();
  Value cbPtrIdx;
  if (storageAccess) {
    assert(((storageAccess->role == PipeTransportStorageRole::Source &&
             direction == NocCopyDirection::Read) ||
            (storageAccess->role == PipeTransportStorageRole::Destination &&
             direction == NocCopyDirection::Write)) &&
           "transport storage role does not match tensor copy direction");
    Value scratchAddress = buildPipeSramScratchAddress(
        op, storageAccess->scratchByteOffset, rewriter);
    Value scratchAddressIndex =
        arith::IndexCastOp::create(rewriter, loc, indexTy, scratchAddress);
    Value currentSlot;
    if (storageAccess->dynamicSlotCounterIndex) {
      Value slotCounter = lookupPipeTransportSlotCounter(
          op, *storageAccess->dynamicSlotCounterIndex, slotCounters);
      Value zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value currentSlotI32 = memref::LoadOp::create(rewriter, loc, slotCounter,
                                                    ValueRange{zeroIndex});
      currentSlot =
          arith::IndexCastOp::create(rewriter, loc, indexTy, currentSlotI32);
    }
    cbPtrIdx = materializeTransportStorageAddress(
        op, scratchAddressIndex, currentSlot, *storageAccess, rewriter);
  } else {
    auto cbConverted = utils::convertTTLCBToTTKernel(cb, rewriter, loc);
    assert(succeeded(cbConverted) && "preflight checked DFB type");
    Value cbPtr = isRead
                      ? ttk::GetWritePtrOp::create(rewriter, loc, *cbConverted)
                            .getResult()
                      : ttk::GetReadPtrOp::create(rewriter, loc, *cbConverted)
                            .getResult();
    cbPtrIdx = arith::IndexCastOp::create(rewriter, loc, indexTy, cbPtr);
  }
  auto pageSizeIdx = arith::ConstantIndexOp::create(
      rewriter, loc, accessorInfo->pageSizeBytes);
  auto i32Ty = rewriter.getI32Type();
  int64_t nocIndex = getNocIndex(op);
  Value nocVal = arith::ConstantOp::create(rewriter, loc, rewriter.getI8Type(),
                                           rewriter.getI8IntegerAttr(nocIndex));

  SmallVector<int64_t> cbBounds(transferShape.begin(), transferShape.end());

  emitTileLoop(
      rewriter, loc, cbBounds,
      [&](OpBuilder &loopBuilder, Location bodyLoc, ValueRange cbIVs) {
        // Tensor coordinates: for squeezed leading dims, use the scalar
        // startIndex directly. For range dims, add the CB loop IV.
        SmallVector<Value> tensorCoords;
        for (unsigned d = 0; d < tensorRank; ++d) {
          Value coord;
          if (d < rankDiff) {
            coord = startIndices[d];
          } else {
            coord = arith::AddIOp::create(loopBuilder, bodyLoc, startIndices[d],
                                          cbIVs[d - rankDiff]);
          }
          tensorCoords.push_back(coord);
        }

        auto tensorTileIdxOp = affine::AffineLinearizeIndexOp::create(
            loopBuilder, bodyLoc, tensorCoords, tensorGridShape);
        tensorTileIdxOp->setAttr(kExpandLinearizeIndexAttr,
                                 loopBuilder.getUnitAttr());
        Value tensorTileIdx = tensorTileIdxOp.getResult();

        auto cbTileIdxOp = affine::AffineLinearizeIndexOp::create(
            loopBuilder, bodyLoc, cbIVs, cbBounds);
        cbTileIdxOp->setAttr(kExpandLinearizeIndexAttr,
                             loopBuilder.getUnitAttr());
        Value cbTileIdx = cbTileIdxOp.getResult();

        // Compute CB address: cbPtr + cbTileIdx * pageSize
        Value byteOffset =
            arith::MulIOp::create(loopBuilder, bodyLoc, cbTileIdx, pageSizeIdx);
        Value cbAddrIdx =
            arith::AddIOp::create(loopBuilder, bodyLoc, cbPtrIdx, byteOffset);

        // Cast to i32 for NOC operation.
        Value tensorTileIdx32 = arith::IndexCastOp::create(
            loopBuilder, bodyLoc, i32Ty, tensorTileIdx);
        Value cbAddr =
            arith::IndexCastOp::create(loopBuilder, bodyLoc, i32Ty, cbAddrIdx);

        if (isRead) {
          ttk::NocAsyncReadTileOp::create(loopBuilder, bodyLoc, tensorTileIdx32,
                                          accessor, cbAddr, nocVal);
        } else {
          ttk::NocAsyncWriteTileOp::create(
              loopBuilder, bodyLoc, tensorTileIdx32, accessor, cbAddr, nocVal);
        }
      });

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

struct TensorSliceLowering : OpConversionPattern<TensorSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TensorSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TensorSliceOp is consumed by CopyLowering via getDefiningOp.
    // After copy lowering, the slice result has no users and can be erased.
    if (!op.getResult().use_empty()) {
      return rewriter.notifyMatchFailure(
          op, "tensor_slice has remaining uses after copy lowering");
    }
    rewriter.eraseOp(op);
    return success();
  }
};

struct CopyLowering : OpConversionPattern<CopyOp> {
  CopyLowering(const TypeConverter &typeConverter, MLIRContext *context,
               const PipeTransportPlan &pipeTransportPlan,
               const PipeTransportSlotCounterMap &slotCounters)
      : OpConversionPattern(typeConverter, context),
        pipeTransportPlan(pipeTransportPlan), slotCounters(slotCounters) {}

  LogicalResult
  matchAndRewrite(CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter = this->getTypeConverter();
    if (!typeConverter) {
      return rewriter.notifyMatchFailure(op, "no type converter");
    }

    Value src = op.getSrc();
    Value dst = op.getDst();
    auto srcKind = classifyOperand(src);
    auto dstKind = classifyOperand(dst);

    bool srcIsSlice = srcKind == CopyOperandKind::TensorSlice;
    bool srcIsCB = srcKind == CopyOperandKind::CircularBuffer;
    bool srcIsPipe = srcKind == CopyOperandKind::Pipe;
    bool dstIsSlice = dstKind == CopyOperandKind::TensorSlice;
    bool dstIsCB = dstKind == CopyOperandKind::CircularBuffer;
    bool dstIsPipe = dstKind == CopyOperandKind::Pipe;
    bool dstIsDFBAttachedTensor = dstKind == CopyOperandKind::DFBAttachedTensor;

    // Pipe transfers are expanded to ttl.pipe_transfer ops before conversion.
    if (srcIsCB && dstIsPipe) {
      return op.emitError("internal compiler error: pipe send copy "
                          "survived pipe transfer expansion");
    }
    if (srcIsPipe && dstIsDFBAttachedTensor) {
      return op.emitError("internal compiler error: pipe receive copy "
                          "survived pipe transfer expansion");
    }
    if (srcIsPipe || dstIsPipe) {
      return rewriter.notifyMatchFailure(
          op, "pipe copy requires CB <-> Pipe, got invalid combination");
    }

    // Non-pipe transfers: validate exactly one TensorSlice and one CB.
    if (!((srcIsSlice && dstIsCB) || (srcIsCB && dstIsSlice))) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "ttl.copy requires one tensor_slice and one circular_buffer, "
             << "got src=" << src.getType() << " dst=" << dst.getType();
      });
    }

    // TensorSlice -> CB: read tiles from tensor into circular buffer.
    if (srcIsSlice && dstIsCB) {
      auto sliceOp = src.getDefiningOp<TensorSliceOp>();
      if (!sliceOp) {
        return rewriter.notifyMatchFailure(
            op, "tensor_slice source must come from ttl.tensor_slice op");
      }
      return lowerTensorCBCopy(op, sliceOp, adaptor.getDst(),
                               NocCopyDirection::Read,
                               pipeTransportPlan.lookupStorageAccess(op),
                               slotCounters, rewriter, *typeConverter);
    }

    // CB -> TensorSlice: write tiles from circular buffer to tensor.
    auto sliceOp = dst.getDefiningOp<TensorSliceOp>();
    if (!sliceOp) {
      return rewriter.notifyMatchFailure(
          op, "tensor_slice destination must come from ttl.tensor_slice op");
    }
    return lowerTensorCBCopy(op, sliceOp, adaptor.getSrc(),
                             NocCopyDirection::Write,
                             pipeTransportPlan.lookupStorageAccess(op),
                             slotCounters, rewriter, *typeConverter);
  }

private:
  const PipeTransportPlan &pipeTransportPlan;
  const PipeTransportSlotCounterMap &slotCounters;
};

struct PipeTransferPostLowering : OpConversionPattern<PipeTransferPostOp> {
  PipeTransferPostLowering(
      const TypeConverter &typeConverter, MLIRContext *context,
      const PipeModulePlan &pipeModulePlan,
      const PipeCounterProgressMap &counters,
      const PipeSelectedPostSequenceMap &selectedPostSequenceCounters,
      const PipeResourcePlan &pipeResourcePlan)
      : OpConversionPattern(typeConverter, context),
        pipeModulePlan(pipeModulePlan), counters(counters),
        selectedPostSequenceCounters(selectedPostSequenceCounters),
        pipeResourcePlan(pipeResourcePlan) {}

  LogicalResult
  matchAndRewrite(PipeTransferPostOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (pipeResourcePlan.staticallyInactiveOps.contains(op.getOperation())) {
      lowerInactivePipeTransferPost(op, rewriter);
      return success();
    }
    // Slice-offset materialization requires the original destination tensor,
    // while the plan supplies the already-resolved receiver DFB.
    return lowerPipeTransferPost(
        op, op.getDst(), pipeModulePlan.getTransferPlan(op.getOperation()),
        counters, selectedPostSequenceCounters, pipeResourcePlan, rewriter);
  }

private:
  const PipeModulePlan &pipeModulePlan;
  const PipeCounterProgressMap &counters;
  const PipeSelectedPostSequenceMap &selectedPostSequenceCounters;
  const PipeResourcePlan &pipeResourcePlan;
};

struct PipeTransferSendLowering : OpConversionPattern<PipeTransferSendOp> {
  PipeTransferSendLowering(
      const TypeConverter &typeConverter, MLIRContext *context,
      const PipeModulePlan &pipeModulePlan,
      const PipeResourcePlan &pipeResourcePlan,
      const PipeCapacityPlan &pipeCapacityPlan,
      const PipeCounterProgressMap &senderCapacityCounters,
      const PipeComputedAddressCounterMap &computedAddressCounters)
      : OpConversionPattern(typeConverter, context),
        pipeModulePlan(pipeModulePlan), pipeResourcePlan(pipeResourcePlan),
        pipeCapacityPlan(pipeCapacityPlan),
        senderCapacityCounters(senderCapacityCounters),
        computedAddressCounters(computedAddressCounters) {}

  LogicalResult
  matchAndRewrite(PipeTransferSendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (pipeResourcePlan.staticallyInactiveOps.contains(op.getOperation())) {
      lowerInactivePipeTransferSend(op, rewriter);
      return success();
    }
    return lowerPipeTransferSend(
        op, adaptor.getSrc(), pipeModulePlan.getTransferPlan(op.getOperation()),
        pipeModulePlan.getTransportPlan(), pipeResourcePlan, pipeCapacityPlan,
        senderCapacityCounters, computedAddressCounters, rewriter);
  }

private:
  const PipeModulePlan &pipeModulePlan;
  const PipeResourcePlan &pipeResourcePlan;
  const PipeCapacityPlan &pipeCapacityPlan;
  const PipeCounterProgressMap &senderCapacityCounters;
  const PipeComputedAddressCounterMap &computedAddressCounters;
};

struct PipeTransferWaitLowering : OpConversionPattern<PipeTransferWaitOp> {
  PipeTransferWaitLowering(const TypeConverter &typeConverter,
                           MLIRContext *context,
                           const PipeModulePlan &pipeModulePlan,
                           const PipeResourcePlan &pipeResourcePlan)
      : OpConversionPattern(typeConverter, context),
        pipeModulePlan(pipeModulePlan), pipeResourcePlan(pipeResourcePlan) {}

  LogicalResult
  matchAndRewrite(PipeTransferWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (pipeResourcePlan.staticallyInactiveOps.contains(op.getOperation())) {
      rewriter.eraseOp(op);
      return success();
    }
    return lowerPipeTransferWait(
        op, adaptor.getToken(),
        pipeModulePlan.getTransferPlan(op.getOperation()), pipeResourcePlan,
        rewriter);
  }

private:
  const PipeModulePlan &pipeModulePlan;
  const PipeResourcePlan &pipeResourcePlan;
};

struct WaitLowering : OpConversionPattern<WaitOp> {
  WaitLowering(const TypeConverter &typeConverter, MLIRContext *context,
               const llvm::SmallPtrSetImpl<Operation *> &completedPipeSends)
      : OpConversionPattern(typeConverter, context),
        completedPipeSends(completedPipeSends) {}

  LogicalResult
  matchAndRewrite(WaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (completedPipeSends.contains(op)) {
      // Pipe sends wait for the payload write before signaling receiver
      // completion, so the send handle is complete when the send op returns.
      rewriter.eraseOp(op);
      return success();
    }

    // TODO(ttl): Lower ttl.wait to TRID-specific barriers keyed by the transfer
    // handle (read vs write barrier based on transfer direction). Issue: #87.
    //
    // MVP behavior: emit the corresponding global barrier based on transfer
    // direction. Pipe receive waits are expanded to ttl.pipe_transfer.wait
    // before this conversion.
    auto kind = getTransferKindFromHandleType(op.getXf().getType());
    if (!kind) {
      return op.emitError("untyped transfer handle survived pipe receive "
                          "expansion");
    }
    // Future-proofing: TransferKind is currently {read, write}, but fail
    // explicitly before mutating IR if it ever expands without updating the
    // lowering.
    if (*kind != TransferKind::read && *kind != TransferKind::write) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "unsupported TransferKind for ttl.wait lowering";
      });
    }
    int64_t nocIndex = getNocIndex(op);
    Value nocVal =
        arith::ConstantOp::create(rewriter, op.getLoc(), rewriter.getI8Type(),
                                  rewriter.getI8IntegerAttr(nocIndex));
    if (*kind == TransferKind::read) {
      ttk::NocAsyncReadBarrierOp::create(rewriter, op.getLoc(), nocVal);
    } else {
      ttk::NocAsyncWriteBarrierOp::create(rewriter, op.getLoc(), nocVal);
    }
    rewriter.eraseOp(op);
    return success();
  }

private:
  const llvm::SmallPtrSetImpl<Operation *> &completedPipeSends;
};

//===----------------------------------------------------------------------===//
// Core indexing operation lowering patterns
//===----------------------------------------------------------------------===//

struct CoreXLowering : OpConversionPattern<CoreXOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CoreXOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Use logical coordinates (grid position), not virtual NOC coordinates
    rewriter.replaceOpWithNewOp<ttk::MyLogicalXOp>(op, rewriter.getIndexType());
    return success();
  }
};

struct CoreYLowering : OpConversionPattern<CoreYOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CoreYOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Use logical coordinates (grid position), not virtual NOC coordinates
    rewriter.replaceOpWithNewOp<ttk::MyLogicalYOp>(op, rewriter.getIndexType());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// DFB index query lowering
//===----------------------------------------------------------------------===//

struct GetDfbIdLowering : OpConversionPattern<GetDfbIdOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(GetDfbIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<int32_t> dfbIndex = getValidatedDFBIndex(op.getDfb(), op);
    if (failed(dfbIndex)) {
      return failure();
    }
    auto convertedDfb =
        utils::convertTTLCBToTTKernel(adaptor.getDfb(), rewriter, op.getLoc());
    if (failed(convertedDfb)) {
      return rewriter.notifyMatchFailure(op, "failed to convert DFB type");
    }
    auto newOp = ttk::GetDfbIdOp::create(rewriter, op.getLoc(),
                                         rewriter.getI32Type(), *convertedDfb);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct RawAddrLowering : OpConversionPattern<RawAddrOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RawAddrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<unsigned> argIdx = getTensorFuncArgIndex(op.getTensor());
    if (failed(argIdx)) {
      return rewriter.notifyMatchFailure(
          op, "raw_addr operand must be a function tensor argument");
    }
    Value bankBase = getCommonRuntimeArg(*argIdx, op.getLoc(), rewriter);
    rewriter.replaceOp(op, bankBase);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Synchronized DFB reset lowering
//===----------------------------------------------------------------------===//

struct DFBResetLoweringPlan {
  DenseMap<SynchronizedDFBResetAttr, int64_t> stateOffsetByReset;
  int64_t scratchBaseOffset = 0;
  int64_t scratchBytes = 0;
  uint64_t allDFBMask = 0;
};

static FailureOr<DFBResetLoweringPlan>
buildDFBResetLoweringPlan(ModuleOp module) {
  SmallVector<SynchronizedDFBResetAttr> orderedResets;
  if (failed(collectSynchronizedDFBResets(module, orderedResets))) {
    return failure();
  }
  FailureOr<uint64_t> scratchBytes = getSynchronizedDFBResetStateBytes(module);
  if (failed(scratchBytes) ||
      *scratchBytes >
          static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    if (succeeded(scratchBytes)) {
      module.emitOpError(
          "DFB reset synchronization state is not representable");
    }
    return failure();
  }

  DFBResetLoweringPlan plan;
  for (auto [resetIndex, reset] : llvm::enumerate(orderedResets)) {
    plan.stateOffsetByReset.try_emplace(
        reset, static_cast<int64_t>(resetIndex) * kDFBResetStateBytes);
  }
  plan.scratchBytes = static_cast<int64_t>(*scratchBytes);

  WalkResult allocationResult = module.walk([&](BindCBOp bind) -> WalkResult {
    std::optional<int64_t> dfbIndex = getCBIndex(bind.getResult());
    if (!dfbIndex) {
      bind.emitOpError("requires a finalized DFB index before reset lowering");
      return WalkResult::interrupt();
    }
    int32_t targetMaxDFBIndices = getTargetMaxDFBIndices(bind);
    if (*dfbIndex < 0 || *dfbIndex >= targetMaxDFBIndices) {
      bind.emitOpError("finalized DFB index ")
          << *dfbIndex << " is outside [0, " << targetMaxDFBIndices - 1
          << "] for " << getTargetDFBIndexCapacityDescription(bind);
      return WalkResult::interrupt();
    }
    plan.allDFBMask |= uint64_t{1} << static_cast<unsigned>(*dfbIndex);
    return WalkResult::advance();
  });
  if (allocationResult.wasInterrupted()) {
    return failure();
  }

  if (!orderedResets.empty()) {
    Builder builder(module.getContext());
    module->setAttr(kDFBResetCountAttrName,
                    builder.getI64IntegerAttr(orderedResets.size()));
  }
  return plan;
}

static LogicalResult lowerDFBReset(Operation *operation,
                                   SynchronizedDFBResetAttr reset,
                                   uint64_t dfbMask,
                                   const DFBResetLoweringPlan &plan,
                                   ConversionPatternRewriter &rewriter) {
  auto stateOffsetIt = plan.stateOffsetByReset.find(reset);
  if (stateOffsetIt == plan.stateOffsetByReset.end()) {
    return operation->emitError("is absent from the DFB reset lowering plan");
  }
  Location location = operation->getLoc();
  Value synchronizationAddress = buildPipeSramScratchAddress(
      operation, plan.scratchBaseOffset + stateOffsetIt->second, rewriter);
  Value lowMask = arith::ConstantIntOp::create(
      rewriter, location, static_cast<uint32_t>(dfbMask), 32);
  Value highMask = arith::ConstantIntOp::create(
      rewriter, location, static_cast<uint32_t>(dfbMask >> 32), 32);
  ttk::OpaqueCallOp::create(
      rewriter, location, TypeRange{},
      rewriter.getStringAttr("experimental::reset_dfb_interfaces"),
      rewriter.getStringAttr("<cstdint>"),
      ValueRange{synchronizationAddress, lowMask, highMask}, ArrayAttr(),
      rewriter.getDenseI32ArrayAttr({0, 1, 2}));
  rewriter.eraseOp(operation);
  return success();
}

struct ResetDFBsLowering : OpConversionPattern<ResetDFBsOp> {
  ResetDFBsLowering(TypeConverter &typeConverter, MLIRContext *context,
                    const DFBResetLoweringPlan &plan)
      : OpConversionPattern(typeConverter, context), plan(plan) {}

  LogicalResult
  matchAndRewrite(ResetDFBsOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    uint64_t dfbMask = 0;
    for (Value dfb : op.getDfbs()) {
      FailureOr<int32_t> dfbIndex = getValidatedDFBIndex(dfb, op);
      if (failed(dfbIndex)) {
        return failure();
      }
      dfbMask |= uint64_t{1} << static_cast<unsigned>(*dfbIndex);
    }
    return lowerDFBReset(op, op.getReset(), dfbMask, plan, rewriter);
  }

private:
  const DFBResetLoweringPlan &plan;
};

struct ResetAllDFBsLowering : OpConversionPattern<ResetAllDFBsOp> {
  ResetAllDFBsLowering(TypeConverter &typeConverter, MLIRContext *context,
                       const DFBResetLoweringPlan &plan)
      : OpConversionPattern(typeConverter, context), plan(plan) {}

  LogicalResult
  matchAndRewrite(ResetAllDFBsOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerDFBReset(op, op.getReset(), plan.allDFBMask, plan, rewriter);
  }

private:
  const DFBResetLoweringPlan &plan;
};

//===----------------------------------------------------------------------===//
// DFB reconfiguration lowering
//===----------------------------------------------------------------------===//

struct DFBReconfigurationLowering : OpConversionPattern<DFBReconfigurationOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DFBReconfigurationOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    func::FuncOp function = op->getParentOfType<func::FuncOp>();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!function || !module) {
      return op.emitError("must be nested in a module kernel function");
    }
    auto plan =
        module->getAttrOfType<DictionaryAttr>(kDFBReconfigurationPlanAttrName);
    auto boundaryOrdinals =
        plan ? plan.getAs<DenseI64ArrayAttr>("boundary_ordinals")
             : DenseI64ArrayAttr();
    auto dfbEntries = plan ? plan.getAs<ArrayAttr>("dfbs") : ArrayAttr();
    if (!boundaryOrdinals || !dfbEntries) {
      return op.emitError("requires finalized DFB reconfiguration metadata");
    }
    int64_t ordinal = op.getBoundary().getOrdinal();
    auto ordinalIt = llvm::find(boundaryOrdinals.asArrayRef(), ordinal);
    if (ordinalIt == boundaryOrdinals.asArrayRef().end()) {
      return op.emitError("boundary ordinal is absent from finalized DFB "
                          "reconfiguration metadata");
    }

    size_t boundaryRuntimeArgOffset = static_cast<size_t>(
        std::distance(boundaryOrdinals.asArrayRef().begin(), ordinalIt));
    if (dfbEntries.size() > std::numeric_limits<int32_t>::max() ||
        boundaryRuntimeArgOffset > std::numeric_limits<int32_t>::max()) {
      return op.emitError("runtime argument index is out of range");
    }

    Value callerRuntimeArgCount = ttk::GetCompileArgValOp::create(
        rewriter, op.getLoc(), rewriter.getI32Type(),
        static_cast<int32_t>(dfbEntries.size()));
    Value boundaryOffset = arith::ConstantIntOp::create(
        rewriter, op.getLoc(), static_cast<int32_t>(boundaryRuntimeArgOffset),
        32);
    Value runtimeArgIndex = arith::AddIOp::create(
        rewriter, op.getLoc(), callerRuntimeArgCount, boundaryOffset);
    Value configurationAddress = ttk::GetArgValOp::create(
        rewriter, op.getLoc(),
        IntegerType::get(rewriter.getContext(), 32, IntegerType::Unsigned),
        runtimeArgIndex);
    ttk::OpaqueCallOp::create(
        rewriter, op.getLoc(), TypeRange{},
        rewriter.getStringAttr("experimental::reconfigure_dfb_interfaces"),
        rewriter.getStringAttr("<cstdint>"), ValueRange{configurationAddress},
        ArrayAttr(), rewriter.getDenseI32ArrayAttr({0}));
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Opaque call lowering
//===----------------------------------------------------------------------===//

struct OpaqueScalarArgument {
  Value value;
};

struct OpaqueDFBArgument {
  int32_t index;
};

struct OpaqueTensorArgument {
  Value tensor;
  TensorAccessorInfo accessorInfo;
};

using OpaqueArgumentPlan =
    std::variant<OpaqueScalarArgument, OpaqueDFBArgument, OpaqueTensorArgument>;

struct OpaqueCallLowering : OpConversionPattern<OpaqueCallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpaqueCallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location location = op.getLoc();

    SmallVector<Attribute> templateArgs;
    if (std::optional<ArrayAttr> sourceTemplateArgs = op.getTemplateArgs()) {
      for (Attribute attribute : *sourceTemplateArgs) {
        auto templateArg = cast<ExternalTemplateArgAttr>(attribute);
        FailureOr<Attribute> convertedTemplateArg = convertTemplateArg(
            templateArg, op.getTemplateDfbOperands(), op, rewriter);
        if (failed(convertedTemplateArg)) {
          return failure();
        }
        templateArgs.push_back(*convertedTemplateArg);
      }
    }

    SmallVector<Type> resultTypes;
    for (Type resultType : op.getResultTypes()) {
      Type convertedType = getTypeConverter()->convertType(resultType);
      if (!convertedType) {
        return rewriter.notifyMatchFailure(op, "failed to convert result type");
      }
      resultTypes.push_back(convertedType);
    }

    SmallVector<OpaqueArgumentPlan> argumentPlan;
    argumentPlan.reserve(op.getArgOperands().size());
    for (auto [originalArg, adaptedArg] :
         llvm::zip(op.getArgOperands(), adaptor.getArgOperands())) {
      Type originalType = originalArg.getType();

      if (mlir::isa<CircularBufferType>(originalType)) {
        FailureOr<int32_t> dfbIndex = getValidatedDFBIndex(originalArg, op);
        if (failed(dfbIndex)) {
          return failure();
        }
        argumentPlan.push_back(OpaqueDFBArgument{*dfbIndex});
        continue;
      }

      if (mlir::isa<RankedTensorType>(originalType)) {
        if (!isNocKernelThread(op)) {
          return op.emitError(
              "tensor operands require a data movement (noc) thread");
        }
        FailureOr<TensorAccessorInfo> accessorInfo =
            getTensorAccessorInfo(originalArg, op, rewriter);
        if (failed(accessorInfo)) {
          return failure();
        }
        argumentPlan.push_back(
            OpaqueTensorArgument{originalArg, *accessorInfo});
        continue;
      }

      argumentPlan.push_back(OpaqueScalarArgument{adaptedArg});
    }

    SmallVector<Value> convertedArgs;
    convertedArgs.reserve(argumentPlan.size());
    for (const OpaqueArgumentPlan &argument : argumentPlan) {
      if (const auto *scalar = std::get_if<OpaqueScalarArgument>(&argument)) {
        convertedArgs.push_back(scalar->value);
        continue;
      }
      if (const auto *dfb = std::get_if<OpaqueDFBArgument>(&argument)) {
        IntegerType unsignedI32 =
            IntegerType::get(rewriter.getContext(), 32, IntegerType::Unsigned);
        convertedArgs.push_back(ttk::GetCompileArgValOp::create(
            rewriter, location, unsignedI32, dfb->index));
        continue;
      }
      const auto &tensor = std::get<OpaqueTensorArgument>(argument);
      Value bankBase =
          getCommonRuntimeArg(tensor.accessorInfo.argIdx, location, rewriter);
      convertedArgs.push_back(materializeTensorAccessor(
          tensor.tensor, bankBase, tensor.accessorInfo, rewriter));
    }

    ArrayAttr templateArgsAttr;
    if (!templateArgs.empty()) {
      templateArgsAttr = rewriter.getArrayAttr(templateArgs);
    }
    auto newOp = ttk::OpaqueCallOp::create(
        rewriter, location, resultTypes, op.getCalleeAttr(), op.getHeaderAttr(),
        convertedArgs, templateArgsAttr, op.getUnsignedArgIndicesAttr());
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

private:
  /// Resolve DFB metadata before type conversion discards block geometry.
  static FailureOr<Attribute>
  convertTemplateArg(ExternalTemplateArgAttr templateArg,
                     ValueRange templateDFBs, OpaqueCallOp op,
                     ConversionPatternRewriter &rewriter) {
    ExternalTemplateArgKind kind = templateArg.getKind();
    int64_t payload = templateArg.getValue();
    if (kind == ExternalTemplateArgKind::SignedInteger) {
      IntegerType signedI32 =
          IntegerType::get(rewriter.getContext(), 32, IntegerType::Signed);
      return rewriter.getIntegerAttr(signedI32, payload);
    }
    if (kind == ExternalTemplateArgKind::Boolean) {
      return rewriter.getBoolAttr(payload != 0);
    }
    if (kind == ExternalTemplateArgKind::UnsignedInteger) {
      return rewriter.getUI32IntegerAttr(static_cast<uint32_t>(payload));
    }

    if (payload < 0 || static_cast<size_t>(payload) >= templateDFBs.size()) {
      return op.emitError("template DFB operand index ")
             << payload << " is out of range for " << templateDFBs.size()
             << " operands";
    }
    Value dfb = templateDFBs[static_cast<size_t>(payload)];
    FailureOr<int32_t> dfbIndex = getValidatedDFBIndex(dfb, op);
    if (failed(dfbIndex)) {
      return failure();
    }
    if (kind == ExternalTemplateArgKind::DFBIndex) {
      return rewriter.getUI32IntegerAttr(static_cast<uint32_t>(*dfbIndex));
    }
    if (kind == ExternalTemplateArgKind::DFBDescriptor) {
      auto dfbType = cast<CircularBufferType>(dfb.getType());
      FailureOr<uint64_t> pagesPerBlock = getDFBPagesPerBlock(dfbType);
      FailureOr<uint64_t> pageSizeBytes = getDFBPageSizeBytes(dfbType);
      int64_t blockCount = dfbType.getBlockCount();
      constexpr uint64_t maxDescriptorField =
          std::numeric_limits<uint32_t>::max();
      if (failed(pageSizeBytes)) {
        return op.emitError(
                   "DFB descriptor element type must occupy a positive whole "
                   "number of bytes, got ")
               << dfbType.getElementType();
      }
      if (failed(pagesPerBlock) || blockCount <= 0) {
        return op.emitError("DFB descriptor dimensions are not representable");
      }
      if (*pagesPerBlock > maxDescriptorField ||
          static_cast<uint64_t>(blockCount) > maxDescriptorField ||
          *pageSizeBytes > maxDescriptorField) {
        return op.emitError(
            "DFB descriptor dimensions or page size exceed uint32_t");
      }
      return ttk::DFBDescriptorAttr::get(rewriter.getContext(), *dfbIndex,
                                         *pagesPerBlock, blockCount,
                                         static_cast<int64_t>(*pageSizeBytes));
    }
    llvm_unreachable("unhandled external template argument kind");
  }
};

/// Tensor-level ttl.store ops must be lowered to tile_store by
/// convert-ttl-to-compute. Any surviving to this point is a miscompile.
struct StoreLowering : OpConversionPattern<StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return op.emitError("ttl.store survived to ttkernel lowering; "
                        "convert-ttl-to-compute should have lowered this to "
                        "ttl.tile_store");
  }
};

struct FuncKernelFinalize : OpRewritePattern<FuncOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(FuncOp op,
                                PatternRewriter &rewriter) const override {
    auto ttlAttr =
        op->getAttrOfType<ttk::ThreadTypeAttr>(kKernelThreadAttrName);
    if (!ttlAttr || ttlAttr.getValue() != ttk::ThreadType::Noc) {
      return failure();
    }
    op->removeAttr(kKernelThreadAttrName);
    op->setAttr(ttk::ThreadTypeAttr::name, ttlAttr);

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

//===----------------------------------------------------------------------===//
// Raw Element Access Lowering
//===----------------------------------------------------------------------===//

/// Return the same-width signless integer type used for raw float storage.
static IntegerType getIntegerStorageType(MLIRContext *context,
                                         FloatType floatType) {
  return IntegerType::get(context, floatType.getWidth());
}

/// Compute the flat element offset for a raw element access operation.
/// For tiled layouts, decomposes coordinates into tile index and intra-tile
/// face-order offset. For row-major layouts, linearizes coordinates directly.
/// Returns an i32 value.
static Value computeRawElementOffset(RankedTensorType blockType,
                                     ValueRange coords,
                                     ConversionPatternRewriter &rewriter,
                                     Location loc) {
  auto i32Ty = rewriter.getI32Type();

  auto toI32 = [&](Value v) -> Value {
    return arith::IndexCastOp::create(rewriter, loc, i32Ty, v);
  };
  auto cst = [&](int64_t v) -> Value {
    return arith::ConstantIntOp::create(rewriter, loc, v, 32);
  };

  Type elemTy = blockType.getElementType();
  auto tileType = mlir::dyn_cast<tt::ttcore::TileType>(elemTy);

  if (!tileType) {
    // Row-major: linearize coords into a flat element index.
    ArrayRef<int64_t> shape = blockType.getShape();
    int64_t rank = blockType.getRank();
    Value flat = toI32(coords[0]);
    for (int64_t i = 1; i < rank; ++i) {
      flat = arith::MulIOp::create(rewriter, loc, flat, cst(shape[i]));
      flat = arith::AddIOp::create(rewriter, loc, flat, toI32(coords[i]));
    }
    return flat;
  }

  // Tiled layout: decompose into tile index + face-order intra-tile offset.
  int64_t tileH = tileType.getHeight();
  int64_t tileW = tileType.getWidth();
  int64_t tileElems = tileH * tileW;
  constexpr int64_t kFaceH = 16;
  constexpr int64_t kFaceW = 16;
  constexpr int64_t kFaceElems = kFaceH * kFaceW;
  ArrayRef<int64_t> gridShape = blockType.getShape();
  int64_t rank = blockType.getRank();

  Value tileIdx, intraRow, intraCol;

  if (rank == 1) {
    Value coord = toI32(coords[0]);
    Value tileElemsC = cst(tileElems);
    tileIdx = arith::DivUIOp::create(rewriter, loc, coord, tileElemsC);
    Value intraFlat = arith::RemUIOp::create(rewriter, loc, coord, tileElemsC);
    Value tileWC = cst(tileW);
    intraRow = arith::DivUIOp::create(rewriter, loc, intraFlat, tileWC);
    intraCol = arith::RemUIOp::create(rewriter, loc, intraFlat, tileWC);
  } else {
    Value rowCoord = toI32(coords[rank - 2]);
    Value colCoord = toI32(coords[rank - 1]);
    Value tileHC = cst(tileH);
    Value tileWC = cst(tileW);

    Value tileRow = arith::DivUIOp::create(rewriter, loc, rowCoord, tileHC);
    Value tileCol = arith::DivUIOp::create(rewriter, loc, colCoord, tileWC);
    intraRow = arith::RemUIOp::create(rewriter, loc, rowCoord, tileHC);
    intraCol = arith::RemUIOp::create(rewriter, loc, colCoord, tileWC);

    int64_t gridCols = gridShape[rank - 1];
    tileIdx = arith::MulIOp::create(rewriter, loc, tileRow, cst(gridCols));
    tileIdx = arith::AddIOp::create(rewriter, loc, tileIdx, tileCol);

    for (int64_t i = rank - 3; i >= 0; --i) {
      int64_t stride = 1;
      for (int64_t j = i + 1; j < rank; ++j) {
        stride *= gridShape[j];
      }
      Value contrib =
          arith::MulIOp::create(rewriter, loc, toI32(coords[i]), cst(stride));
      tileIdx = arith::AddIOp::create(rewriter, loc, tileIdx, contrib);
    }
  }

  // Face decomposition: 4x(16x16) faces in row-major face order.
  Value faceHC = cst(kFaceH);
  Value faceWC = cst(kFaceW);
  Value faceRow = arith::DivUIOp::create(rewriter, loc, intraRow, faceHC);
  Value faceCol = arith::DivUIOp::create(rewriter, loc, intraCol, faceWC);
  Value faceIdx = arith::MulIOp::create(rewriter, loc, faceRow, cst(2));
  faceIdx = arith::AddIOp::create(rewriter, loc, faceIdx, faceCol);

  Value localRow = arith::RemUIOp::create(rewriter, loc, intraRow, faceHC);
  Value localCol = arith::RemUIOp::create(rewriter, loc, intraCol, faceWC);

  Value intraElem =
      arith::MulIOp::create(rewriter, loc, faceIdx, cst(kFaceElems));
  Value rowPart = arith::MulIOp::create(rewriter, loc, localRow, faceWC);
  intraElem = arith::AddIOp::create(rewriter, loc, intraElem, rowPart);
  intraElem = arith::AddIOp::create(rewriter, loc, intraElem, localCol);

  Value tileOffset =
      arith::MulIOp::create(rewriter, loc, tileIdx, cst(tileElems));
  return arith::AddIOp::create(rewriter, loc, tileOffset, intraElem);
}

/// Emit the common L1 pointer setup: get_read_ptr or get_write_ptr, then
/// reinterpret_cast to the appropriate L1 typed pointer.
static std::pair<Value, Value>
emitL1PtrAndOffset(Value cb, Value originalBlock, RankedTensorType blockType,
                   ValueRange coords, unsigned elemWidth,
                   ConversionPatternRewriter &rewriter, Location loc) {
  bool fromWait =
      llvm::isa_and_nonnull<CBWaitOp>(findCBAcquireOp(originalBlock));
  Value baseAddr =
      fromWait ? ttk::GetReadPtrOp::create(rewriter, loc, cb).getResult()
               : ttk::GetWritePtrOp::create(rewriter, loc, cb).getResult();

  auto l1PtrTy = ttk::L1AddrPtrType::get(rewriter.getContext(), elemWidth);
  Value l1Ptr = ttk::CastToL1PtrOp::create(rewriter, loc, l1PtrTy, baseAddr);

  Value offset = computeRawElementOffset(blockType, coords, rewriter, loc);
  return {l1Ptr, offset};
}

/// Resolve the TTKernel CB from a raw element op's block operand.
/// Tries getCBFromView on the adapted block first; falls back to
/// getAttachedCB on the original block and converts the !ttl.cb.
static FailureOr<Value>
resolveCBForRawElement(Value adaptedBlock, Value originalBlock,
                       ConversionPatternRewriter &rewriter, Location loc,
                       const TypeConverter *typeConverter) {
  auto cb = getCBFromView(adaptedBlock);
  if (succeeded(cb)) {
    return cb;
  }

  Value origCB = getAttachedCB(originalBlock);
  if (!origCB) {
    return failure();
  }

  return utils::convertTTLCBToTTKernel(origCB, rewriter, loc, typeConverter);
}

/// Convert the raw IEEE-754 representation of a finite, nonnegative float to
/// i32 with truncation toward zero. Both shift operands are clamped because
/// arith.select evaluates both candidate values.
static Value decodeNonnegativeFloatToI32(Value rawBits, FloatType floatType,
                                         ConversionPatternRewriter &rewriter,
                                         Location loc) {
  auto i32Type = rewriter.getI32Type();
  unsigned outputWidth = i32Type.getWidth();
  assert(floatType.getWidth() <= outputWidth &&
         "decode packs the significand into i32");
  unsigned mantissaWidth = floatType.getFPMantissaWidth() - 1;
  unsigned exponentWidth = floatType.getWidth() - mantissaWidth - 1;
  uint32_t exponentMask = (uint32_t{1} << exponentWidth) - 1;
  uint32_t exponentBias = (uint32_t{1} << (exponentWidth - 1)) - 1;

  auto constant = [&](int64_t value) -> Value {
    return arith::ConstantIntOp::create(rewriter, loc, value, outputWidth);
  };

  Value bits = rawBits;
  if (rawBits.getType().getIntOrFloatBitWidth() < outputWidth) {
    bits = arith::ExtUIOp::create(rewriter, loc, i32Type, rawBits);
  }

  Value zero = constant(0);
  Value maximumShift = constant(outputWidth - 1);
  auto clampShift = [&](Value shift) -> Value {
    Value isNegative = arith::CmpIOp::create(
        rewriter, loc, arith::CmpIPredicate::slt, shift, zero);
    Value nonnegativeShift =
        arith::SelectOp::create(rewriter, loc, isNegative, zero, shift);
    Value isTooLarge =
        arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::sgt,
                              nonnegativeShift, maximumShift);
    return arith::SelectOp::create(rewriter, loc, isTooLarge, maximumShift,
                                   nonnegativeShift);
  };

  Value mantissaWidthValue = constant(mantissaWidth);
  Value exponent =
      arith::ShRUIOp::create(rewriter, loc, bits, mantissaWidthValue);
  exponent =
      arith::AndIOp::create(rewriter, loc, exponent, constant(exponentMask));
  exponent =
      arith::SubIOp::create(rewriter, loc, exponent, constant(exponentBias));

  uint32_t mantissaMask = (uint32_t{1} << mantissaWidth) - 1;
  uint32_t hiddenBit = uint32_t{1} << mantissaWidth;
  Value significand =
      arith::AndIOp::create(rewriter, loc, bits, constant(mantissaMask));
  significand =
      arith::OrIOp::create(rewriter, loc, significand, constant(hiddenBit));

  Value leftShift =
      arith::SubIOp::create(rewriter, loc, exponent, mantissaWidthValue);
  Value rightShift =
      arith::SubIOp::create(rewriter, loc, mantissaWidthValue, exponent);
  leftShift = clampShift(leftShift);
  rightShift = clampShift(rightShift);

  Value shiftedLeft =
      arith::ShLIOp::create(rewriter, loc, significand, leftShift);
  Value shiftedRight =
      arith::ShRUIOp::create(rewriter, loc, significand, rightShift);
  Value usesLeftShift = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::sge, exponent, mantissaWidthValue);
  Value magnitude = arith::SelectOp::create(rewriter, loc, usesLeftShift,
                                            shiftedLeft, shiftedRight);

  Value isBelowOne = arith::CmpIOp::create(
      rewriter, loc, arith::CmpIPredicate::slt, exponent, zero);
  return arith::SelectOp::create(rewriter, loc, isBelowOne, zero, magnitude);
}

struct RawElementReadLowering : OpConversionPattern<RawElementReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RawElementReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto blockType = mlir::cast<RankedTensorType>(op.getBlock().getType());
    Type scalarTy = op.getResult().getType();
    IntegerType intTy = getIntegerStorageType(rewriter.getContext(),
                                              mlir::cast<FloatType>(scalarTy));
    unsigned elemWidth = intTy.getWidth();

    auto cb = resolveCBForRawElement(adaptor.getBlock(), op.getBlock(),
                                     rewriter, loc, this->getTypeConverter());
    if (failed(cb)) {
      return rewriter.notifyMatchFailure(op, "block does not trace to a CB");
    }

    auto [l1Ptr, offset] =
        emitL1PtrAndOffset(*cb, op.getBlock(), blockType, adaptor.getCoords(),
                           elemWidth, rewriter, loc);

    Value loaded =
        ttk::LoadFromL1Op::create(rewriter, loc, intTy, l1Ptr, offset);

    auto viewCast =
        UnrealizedConversionCastOp::create(rewriter, loc, scalarTy, loaded);
    rewriter.replaceOp(op, viewCast.getResult(0));
    return success();
  }
};

struct ReadIndexLowering : OpConversionPattern<ReadIndexOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReadIndexOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto blockType = mlir::cast<RankedTensorType>(op.getBlock().getType());
    Type elementType = blockType.getElementType();
    Type scalarType = getTileElementType(elementType).value_or(elementType);
    auto floatType = mlir::cast<FloatType>(scalarType);
    IntegerType integerType =
        getIntegerStorageType(rewriter.getContext(), floatType);
    unsigned elementWidth = integerType.getWidth();

    FailureOr<Value> cb =
        resolveCBForRawElement(adaptor.getBlock(), op.getBlock(), rewriter, loc,
                               this->getTypeConverter());
    if (failed(cb)) {
      return rewriter.notifyMatchFailure(
          op, "block does not trace to a dataflow buffer");
    }

    auto [l1Pointer, offset] =
        emitL1PtrAndOffset(*cb, op.getBlock(), blockType, adaptor.getCoords(),
                           elementWidth, rewriter, loc);
    Value rawBits = ttk::LoadFromL1Op::create(rewriter, loc, integerType,
                                              l1Pointer, offset);
    Value integerValue =
        decodeNonnegativeFloatToI32(rawBits, floatType, rewriter, loc);
    Value indexValue = arith::IndexCastOp::create(
        rewriter, loc, rewriter.getIndexType(), integerValue);
    rewriter.replaceOp(op, indexValue);
    return success();
  }
};

struct RawElementWriteLowering : OpConversionPattern<RawElementWriteOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RawElementWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto blockType = mlir::cast<RankedTensorType>(op.getBlock().getType());
    Type scalarTy = op.getValue().getType();
    IntegerType intTy = getIntegerStorageType(rewriter.getContext(),
                                              mlir::cast<FloatType>(scalarTy));
    unsigned elemWidth = intTy.getWidth();

    auto cb = resolveCBForRawElement(adaptor.getBlock(), op.getBlock(),
                                     rewriter, loc, this->getTypeConverter());
    if (failed(cb)) {
      return rewriter.notifyMatchFailure(op, "block does not trace to a CB");
    }

    Value floatVal = adaptor.getValue();
    Value intVal;
    if (auto cast = floatVal.getDefiningOp<UnrealizedConversionCastOp>();
        cast && cast.getInputs().size() == 1 &&
        cast.getInputs()[0].getType() == intTy) {
      intVal = cast.getInputs()[0];
    } else {
      intVal =
          UnrealizedConversionCastOp::create(rewriter, loc, intTy, floatVal)
              .getResult(0);
    }

    auto [l1Ptr, offset] =
        emitL1PtrAndOffset(*cb, op.getBlock(), blockType, adaptor.getCoords(),
                           elemWidth, rewriter, loc);

    ttk::StoreToL1Op::create(rewriter, loc, intVal, l1Ptr, offset);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TTLConvertTTLToTTKernelPass helper methods
//===----------------------------------------------------------------------===//

/// Phase 1: Lower TTL ops (bind_cb, copy, wait, cb ops, store) to TTKernel.
static LogicalResult lowerTTLOpsToTTKernel(
    ModuleOp mod, MLIRContext &ctx, TTLToTTKernelTypeConverter &typeConverter,
    StringRef passName, bool pipeComputedAddresses, bool pipeCapacitySync,
    bool pipeGlobalSemaphoresOnly, std::optional<uint64_t> l1BudgetOverride) {
  ConversionTarget target(ctx);
  target.addIllegalDialect<tt::ttl::TTLDialect>();
  target.addLegalDialect<affine::AffineDialect, arith::ArithDialect,
                         BuiltinDialect, memref::MemRefDialect, scf::SCFDialect,
                         func::FuncDialect, ttkernel::TTKernelDialect>();

  // Structural ops remain legal (converted elsewhere or kept as-is).
  target.addLegalOp<ComputeOp, YieldOp, AttachCBOp, DstIndexOp, SelectPipeSrcOp,
                    SelectPipeDstOp>();
  target.addLegalOp<PipeTransferCreateOp>();

  // DST lifecycle ops are not tile compute ops; keep them legal until the
  // tile ops lowering phase.
  target.addLegalOp<TileRegsAcquireOp, TileRegsCommitOp, TileRegsWaitOp,
                    TileRegsReleaseOp>();

  // SignpostOp and DPrintOp are lowered in separate EmitC passes.
  target.addLegalOp<SignpostOp, DPrintOp>();

  // Tile compute ops and data movement ops (copy_tile, copy_dst) remain legal
  // until the tile ops lowering phase. Raw element access ops are lowered here
  // despite carrying the DataMovement trait.
  target.addDynamicallyLegalDialect<tt::ttl::TTLDialect>([](Operation *op) {
    if (llvm::isa<RawElementReadOp, ReadIndexOp, RawElementWriteOp>(op)) {
      return false;
    }
    return tt::ttl::isTileComputeOp(op) ||
           op->hasTrait<TTLDataMovementOpTrait>();
  });

  // TensorSliceOp is legal while it has users (CopyLowering will consume them).
  // Once users are gone, TensorSliceLowering erases the op.
  target.addDynamicallyLegalOp<TensorSliceOp>(
      [](TensorSliceOp op) { return !op.getResult().use_empty(); });

  target.addDynamicallyLegalOp<ModuleOp>(
      [&](ModuleOp op) { return typeConverter.isLegal(&op.getBodyRegion()); });
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });

  // Preserve the generated record-selection regions so pipe graph ordering
  // does not mistake them for independent user control flow.
  PipeForeachLoweringInfo foreachLoweringInfo;
  if (failed(lowerPipeNetForeachOps(mod, foreachLoweringInfo))) {
    return failure();
  }

  // Validate explicit transfer IR and resolve every high-level pipe copy before
  // expansion mutates the values used by the analysis.
  {
    ValueOriginAnalysis preExpansionAnalysis(mod);
    if (failed(verifyTransferProvenance(mod, preExpansionAnalysis))) {
      return failure();
    }
    if (failed(expandPipeTransfers(mod, preExpansionAnalysis))) {
      return failure();
    }
  }

  // All remaining provenance consumers share this root-scoped cache.
  ValueOriginAnalysis transferAnalysis(mod);
  if (failed(verifyTransferProvenance(mod, transferAnalysis))) {
    return failure();
  }
  FailureOr<std::unique_ptr<PipeTransferIndex>> maybeTransferIndex =
      PipeTransferIndex::create(mod, transferAnalysis);
  if (failed(maybeTransferIndex)) {
    return failure();
  }
  const PipeTransferIndex &transferIndex = **maybeTransferIndex;

  // Validate receiver DFB consistency before lowering emits the pipe
  // synchronization protocol.
  auto pipeGraphOrErr =
      PipeGraph::build(mod, transferIndex, foreachLoweringInfo);
  if (failed(pipeGraphOrErr)) {
    return failure();
  }

  PipePlanningOptions pipePlanningOptions;
  FailureOr<DFBResetLoweringPlan> resetLoweringPlan =
      buildDFBResetLoweringPlan(mod);
  if (failed(resetLoweringPlan)) {
    return failure();
  }
  pipePlanningOptions.enableComputedAddresses = pipeComputedAddresses;
  pipePlanningOptions.enableCapacitySynchronization = pipeCapacitySync;
  pipePlanningOptions.counterAllocationPolicy =
      pipeGlobalSemaphoresOnly ? PipeCounterAllocationPolicy::GlobalOnly
                               : PipeCounterAllocationPolicy::LocalThenGlobal;
  pipePlanningOptions.trailingSramScratchBytes =
      resetLoweringPlan->scratchBytes;
  pipePlanningOptions.trailingSramScratchAlignment = 4;
  FailureOr<PipeModulePlan> maybePipeModulePlan =
      buildPipeModulePlan(mod, transferAnalysis, transferIndex, *pipeGraphOrErr,
                          pipePlanningOptions);
  if (failed(maybePipeModulePlan)) {
    return failure();
  }
  PipeModulePlan pipeModulePlan = std::move(*maybePipeModulePlan);
  resetLoweringPlan->scratchBaseOffset =
      pipeModulePlan.getTrailingSramScratchOffset();
  FailureOr<DFBAllocationFootprint> allocationFootprint =
      getDFBAllocationFootprint(mod);
  if (failed(allocationFootprint)) {
    mod.emitOpError("failed to compute finalized DFB allocation sizes");
    return failure();
  }
  const PipeResourceRequirements &resourceRequirements =
      pipeModulePlan.getResourceRequirements();
  if (resourceRequirements.sramScratchBytes < 0) {
    mod.emitOpError("PipeNet and reset scratch allocation is negative");
    return failure();
  }
  if (failed(validateCombinedDFBResourceL1Bytes(
          mod, *allocationFootprint,
          static_cast<uint64_t>(resourceRequirements.sramScratchBytes),
          resourceRequirements.globalSemaphoreCount, l1BudgetOverride))) {
    return failure();
  }
  mod->removeAttr(kPipeConservativeL1BytesAttrName);
  applyPipeModuleAttributes(mod, pipeModulePlan);
  const PipeResourcePlan &pipeResourcePlan = pipeModulePlan.getResourcePlan();
  const PipeCapacityPlan &pipeCapacityPlan = pipeModulePlan.getCapacityPlan();
  // [Device 2.0] The kPipeSyncSemaphoreCountAttrName,
  // kPipeGlobalSemaphoreCountAttrName, and kPipeSramScratchBytesAttrName attrs
  // are the current host/runtime ABI for pipe resource binding. Keep the
  // allocation decision in this compiler plan so future typed device APIs only
  // change runtime binding code.
  PipeCounterProgressMap senderCapacityCounters;
  initializePipeCapacityCounters(pipeCapacityPlan, pipeResourcePlan,
                                 senderCapacityCounters);
  PipeCounterProgressMap postSequenceCounters;
  PipeSelectedPostSequenceMap selectedPostSequenceCounters;
  initializePipePostSequenceCounters(pipeResourcePlan, postSequenceCounters,
                                     selectedPostSequenceCounters);
  PipeComputedAddressCounterMap computedAddressCounters;
  initializePipeComputedAddressCounters(pipeResourcePlan,
                                        computedAddressCounters);
  const PipeTransportPlan &pipeTransportPlan =
      pipeModulePlan.getTransportPlan();
  PipeTransportSlotCounterMap transportSlotCounters;
  initializePipeTransportSlotCounters(pipeTransportPlan, transportSlotCounters);
  materializePipeTransportCompletionBarriers(pipeTransportPlan);

  RewritePatternSet patterns(&ctx);
  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                       target);
  target.addDynamicallyLegalDialect<tensor::TensorDialect>(
      [&](Operation *op) { return typeConverter.isLegal(op); });
  patterns.add<TensorOpTypeConversion<tensor::EmptyOp>,
               TensorOpTypeConversion<tensor::InsertOp>,
               TensorOpTypeConversion<tensor::ExtractOp>,
               TensorOpTypeConversion<tensor::CastOp>>(typeConverter, &ctx);
  patterns.add<CopyLowering>(typeConverter, &ctx, pipeTransportPlan,
                             transportSlotCounters);
  patterns.add<PipeTransferPostLowering>(
      typeConverter, &ctx, pipeModulePlan, postSequenceCounters,
      selectedPostSequenceCounters, pipeResourcePlan);
  patterns.add<PipeTransferSendLowering>(
      typeConverter, &ctx, pipeModulePlan, pipeResourcePlan, pipeCapacityPlan,
      senderCapacityCounters, computedAddressCounters);
  patterns.add<PipeTransferWaitLowering>(typeConverter, &ctx, pipeModulePlan,
                                         pipeResourcePlan);
  patterns.add<WaitLowering>(typeConverter, &ctx,
                             pipeModulePlan.getCompletedPipeSendWaits());
  patterns.add<CBReserveLowering, CBPushLowering, CBWaitLowering>(
      typeConverter, &ctx, pipeTransportPlan);
  patterns.add<ResetDFBsLowering, ResetAllDFBsLowering>(typeConverter, &ctx,
                                                        *resetLoweringPlan);
  patterns
      .add<BindCBLowering, TensorSliceLowering, TileStoreLowering,
           StoreLowering, CoreXLowering, CoreYLowering, RawElementReadLowering,
           ReadIndexLowering, RawElementWriteLowering, RawAddrLowering,
           DFBReconfigurationLowering, OpaqueCallLowering, GetDfbIdLowering>(
          typeConverter, &ctx);
  patterns.add<CBPopLowering>(typeConverter, &ctx, pipeCapacityPlan,
                              pipeTransportPlan, transportSlotCounters,
                              pipeResourcePlan);
  populatePipeLoweringPatterns(patterns, typeConverter,
                               pipeModulePlan.getPipeNetIndex());
  populateFunctionOpInterfaceTypeConversionPattern(
      func::FuncOp::getOperationName(), patterns, typeConverter);

  FrozenRewritePatternSet frozen(std::move(patterns));
  std::string diagMessage;
  if (utils::applyPartialConversionWithDiag(mod, target, frozen, passName,
                                            diagMessage)) {
    mod.emitError() << diagMessage;
    return failure();
  }

  SmallVector<PipeTransferCreateOp> deadPipeTransfers;
  mod.walk([&](PipeTransferCreateOp op) {
    if (op->use_empty()) {
      deadPipeTransfers.push_back(op);
    }
  });
  for (PipeTransferCreateOp op : deadPipeTransfers) {
    op.erase();
  }

  SmallVector<Operation *> deadSelectedPipes;
  mod.walk([&](Operation *op) {
    if (mlir::isa<SelectPipeSrcOp, SelectPipeDstOp>(op) && op->use_empty()) {
      deadSelectedPipes.push_back(op);
    }
  });
  for (Operation *op : deadSelectedPipes) {
    op->erase();
  }

  // Greedy cleanup also erases dead unrealized casts used as temporary
  // transfer-token materializations.
  RewritePatternSet cleanupPatterns(&ctx);
  ttkernel::populateTTKernelCleanupPatterns(cleanupPatterns);
  cleanupPatterns.add<ExpandMarkedLinearizeIndex>(&ctx);
  if (failed(applyPatternsGreedily(mod, std::move(cleanupPatterns)))) {
    return failure();
  }

  return success();
}

/// Lower tile compute operations and DST lifecycle operations to TTKernel.
/// Tile compute ops are identified by TTLTileComputeOpTrait. ttl.compute is
/// kept legal here because it is lowered to loops in an earlier pass
/// (ttl-lower-to-loops).
static LogicalResult
lowerTileOpsToTTKernel(ModuleOp mod, MLIRContext &ctx,
                       TTLToTTKernelTypeConverter &typeConverter) {
  ConversionTarget computeTarget(ctx);
  computeTarget.addLegalDialect<ttkernel::TTKernelDialect>();
  computeTarget.addLegalDialect<affine::AffineDialect, arith::ArithDialect>();
  // Keep compute ops legal (tile-only lowering here).
  computeTarget.addLegalOp<ComputeOp, YieldOp, DstIndexOp>();

  // Other dialects are legal (func, tensor, etc.) EXCEPT tile ops.
  computeTarget.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  // Mark TTL ops that need lowering as illegal (tile compute ops, data movement
  // ops, DST lifecycle). All other TTL ops (ComputeOp, YieldOp, AttachCBOp)
  // were explicitly marked legal above.
  computeTarget.addDynamicallyLegalDialect<tt::ttl::TTLDialect>(
      [](Operation *op) {
        // Tile compute ops (add, mul, exp, etc.) are illegal.
        if (tt::ttl::isTileComputeOp(op)) {
          return false;
        }
        // Data movement ops (copy_tile, copy_dst) are illegal.
        if (op->hasTrait<TTLDataMovementOpTrait>()) {
          return false;
        }
        // DST lifecycle ops are illegal.
        if (mlir::isa<TileRegsAcquireOp, TileRegsCommitOp, TileRegsWaitOp,
                      TileRegsReleaseOp>(op)) {
          return false;
        }
        // All other TTL ops are legal (ComputeOp, YieldOp, AttachCBOp).
        return true;
      });

  RewritePatternSet computePatterns(&ctx);
  populateTTLTileOpsToTTKernelPatterns(&typeConverter, computePatterns);
  return applyPartialConversion(mod, computeTarget, std::move(computePatterns));
}

/// Phase 3: Remove structural TTL ops (AttachCBOp, ComputeOp, YieldOp).
/// These are now dead after tile ops have been lowered and CB associations
/// have been used by copy_tile lowering.
static LogicalResult
removeStructuralTTLOps(ModuleOp mod, MLIRContext &ctx,
                       TTLToTTKernelTypeConverter &typeConverter) {
  ConversionTarget cleanupTarget(ctx);
  cleanupTarget.addLegalDialect<ttkernel::TTKernelDialect, arith::ArithDialect,
                                BuiltinDialect, scf::SCFDialect,
                                func::FuncDialect, tensor::TensorDialect>();
  cleanupTarget.addIllegalOp<AttachCBOp>();
  // ComputeOp/YieldOp should be gone after loop lowering, but mark illegal
  // just in case.
  cleanupTarget.addIllegalOp<ComputeOp, YieldOp, DstIndexOp>();

  RewritePatternSet structuralPatterns(&ctx);
  structuralPatterns.add<AttachCBLowering, DstIndexCleanup>(typeConverter,
                                                            &ctx);
  if (failed(applyPartialConversion(mod, cleanupTarget,
                                    std::move(structuralPatterns)))) {
    return failure();
  }

  // Apply FuncKernelFinalize as a greedy rewrite after tile lowering.
  RewritePatternSet finalizePatterns(&ctx);
  finalizePatterns.add<FuncKernelFinalize>(&ctx);
  return applyPatternsGreedily(mod, std::move(finalizePatterns));
}

/// Remove dead tensor ops from a compute kernel function.
/// With side-effect-only loops, tensor.insert no longer exists. Clean up
/// remaining dead tensor.extract and tensor.empty ops.
static void removeTensorDataflowOps(func::FuncOp func) {
  SmallVector<Operation *> deadOps;
  func.walk([&](Operation *op) {
    if (mlir::isa<tensor::ExtractOp, tensor::ExtractSliceOp, tensor::EmptyOp>(
            op) &&
        op->use_empty()) {
      deadOps.push_back(op);
    }
  });
  // Erase innermost-first to avoid dangling uses.
  for (auto *op : llvm::reverse(deadOps)) {
    op->erase();
  }
}

/// Phase 4: Clean up tensor dataflow ops in compute kernels.
/// Remove tensor dataflow ops that were used only for SSA tracking.
/// After loops are lowered and tile ops are converted, tensor.extract/insert/
/// empty are dead code. The actual computation happens through circular
/// buffers and DST registers.
static void cleanupComputeKernels(ModuleOp mod, MLIRContext &ctx) {
  mod.walk([&](func::FuncOp func) {
    auto threadType = convertThreadAttr(func);
    if (!threadType || *threadType != ttk::ThreadType::Compute) {
      return;
    }

    removeTensorDataflowOps(func);

    // Erase unused function arguments. Compute kernels get data from CBs.
    // Only erase arguments that have no uses.
    if (func.getNumArguments() > 0) {
      llvm::BitVector argsToErase(func.getNumArguments());
      for (unsigned i = 0; i < func.getNumArguments(); ++i) {
        if (func.getArgument(i).use_empty()) {
          argsToErase.set(i);
        }
      }
      if (argsToErase.any()) {
        (void)func.eraseArguments(argsToErase);
      }
    }

    // For compute kernels, update function to return void.
    if (!func.getResultTypes().empty()) {
      func.walk([](func::ReturnOp returnOp) {
        if (returnOp.getNumOperands() > 0) {
          OpBuilder builder(returnOp);
          func::ReturnOp::create(builder, returnOp.getLoc());
          returnOp.erase();
        }
      });
      // Update function type to return void.
      auto newFuncType =
          FunctionType::get(&ctx, func.getArgumentTypes(), TypeRange{});
      func.setType(newFuncType);
    }
  });
}

//===----------------------------------------------------------------------===//
// DstSectionOp expansion
//===----------------------------------------------------------------------===//

/// Expand DstSectionOp: insert sync ops at the math/pack boundary (first
/// TileStoreOp), then inline the body. LowerToLoops ensures pack-phase ops
/// are already grouped at the end.
static void expandDstSection(DstSectionOp dstSection) {
  Block &body = dstSection.getBody().front();
  Block *parentBlock = dstSection->getBlock();
  Location loc = dstSection.getLoc();

  // Find the first TileStoreOp -- this is the math/pack boundary.
  Operation *firstStore = nullptr;
  for (Operation &op : body.without_terminator()) {
    if (mlir::isa<TileStoreOp>(&op)) {
      firstStore = &op;
      break;
    }
  }

  // Insert sync ops within the body at the correct positions.
  OpBuilder builder(dstSection->getContext());

  // Acquire at the start of the body.
  builder.setInsertionPointToStart(&body);
  TileRegsAcquireOp::create(builder, loc);

  // Commit + wait before the first store (or before yield if no stores).
  if (firstStore) {
    builder.setInsertionPoint(firstStore);
  } else {
    builder.setInsertionPoint(body.getTerminator());
  }
  TileRegsCommitOp::create(builder, loc);
  TileRegsWaitOp::create(builder, loc);

  // Release before the yield.
  builder.setInsertionPoint(body.getTerminator());
  TileRegsReleaseOp::create(builder, loc);

  // Erase the yield terminator -- the body will be inlined into the parent.
  body.getTerminator()->erase();

  // Inline the body into the parent block, replacing the DstSectionOp.
  parentBlock->getOperations().splice(Block::iterator(dstSection),
                                      body.getOperations());
  dstSection->erase();
}

/// Expand all DstSectionOps in the module to four TTL sync ops.
/// Runs as a pre-processing step before dialect conversion.
static void expandDstSections(ModuleOp mod) {
  SmallVector<DstSectionOp> sections;
  mod.walk([&](DstSectionOp op) { sections.push_back(op); });
  for (DstSectionOp section : sections) {
    expandDstSection(section);
  }
}

//===----------------------------------------------------------------------===//
// TTLConvertTTLToTTKernelPass
//===----------------------------------------------------------------------===//

static LogicalResult
validateTileOperationsForTarget(ModuleOp module,
                                const ComputeTargetEnvironment &target) {
  bool hasErrors = false;
  module.walk([&](func::FuncOp function) {
    function.walk([&](Operation *operation) {
      if (!getComputePrimitive(operation)) {
        return;
      }
      std::string failureReason;
      if (failed(target.validateOperation(operation, failureReason))) {
        operation->emitOpError(failureReason);
        hasErrors = true;
      }
    });
  });
  return failure(hasErrors);
}

struct TTLConvertTTLToTTKernelPass
    : impl::TTLConvertTTLToTTKernelBase<TTLConvertTTLToTTKernelPass> {
  using TTLConvertTTLToTTKernelBase::TTLConvertTTLToTTKernelBase;

  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    ModuleOp mod = getOperation();
    TTLToTTKernelTypeConverter typeConverter;

    if (failed(validateSynchronizedDFBResetTarget(mod))) {
      signalPassFailure();
      return;
    }
    if (failed(validateDFBReconfigurationTarget(mod))) {
      signalPassFailure();
      return;
    }
    std::string targetFailureReason;
    FailureOr<std::unique_ptr<ComputeTargetEnvironment>> target =
        ComputeTargetEnvironment::get(mod, targetFailureReason);
    if (failed(target)) {
      mod.emitOpError(targetFailureReason);
      signalPassFailure();
      return;
    }
    if (failed(validateTileOperationsForTarget(mod, **target))) {
      signalPassFailure();
      return;
    }
    if (failed(verifyTileExecutionSemantics(mod))) {
      signalPassFailure();
      return;
    }

    // Phase 0: Expand DstSectionOp into four TTL sync ops. This inlines the
    // DstSectionOp body and inserts acquire/commit/wait/release around it,
    // with stores reordered to the pack phase (after wait).
    expandDstSections(mod);

    // Phase 1: Lower TTL ops to TTKernel (bind_cb, copy, wait, cb ops, store)
    if (failed(lowerTTLOpsToTTKernel(
            mod, ctx, typeConverter, getName(), pipeComputedAddresses,
            pipeCapacitySync, pipeGlobalSemaphoresOnly,
            l1BudgetOverride == 0
                ? std::nullopt
                : std::optional<uint64_t>(l1BudgetOverride)))) {
      signalPassFailure();
      return;
    }

    // Phase 2: Lower tile compute ops to TTKernel (tile_add, tile_mul, ...)
    if (failed(lowerTileOpsToTTKernel(mod, ctx, typeConverter))) {
      signalPassFailure();
      return;
    }

    // Phase 3: Remove structural TTL ops (attach_cb, compute, yield)
    if (failed(removeStructuralTTLOps(mod, ctx, typeConverter))) {
      signalPassFailure();
      return;
    }

    // Phase 4: Clean up tensor dataflow ops in compute kernels.
    cleanupComputeKernels(mod, ctx);
  }
};

} // namespace

} // namespace mlir::tt::ttl
