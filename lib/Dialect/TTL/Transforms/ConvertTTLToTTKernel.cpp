// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h" // IWYU pragma: keep

#include "PipeGraph.h"
#include "PipeLowering.h"
#include "ttlang/Dialect/TTKernel/Transforms/TTKernelCleanupPatterns.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsEnums.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/Utils/ConversionUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include <cstdlib>
#include <utility>

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

// PipeGraph is defined in PipeGraph.h.

class TTLToTTKernelTypeConverter : public TypeConverter {
public:
  TTLToTTKernelTypeConverter() {
    // Specific conversions first; identity fallback last.
    // CB: lower to TTKernel CB type with flattened element count.
    addConversion([](CircularBufferType t) -> Type {
      return ttk::CBType::get(t.getContext(), t.getTotalElements(),
                              t.getElementType());
    });
    // Tensor -> TensorAccessor for TTKernel when TTL layout is present.
    addConversion([](RankedTensorType t) -> Type {
      if (t.getEncoding() && mlir::isa<tt::ttl::LayoutAttr>(t.getEncoding())) {
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
      return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                inputs)
          .getResult(0);
    };
    addSourceMaterialization(castMaterialization);
    addTargetMaterialization(castMaterialization);
  }
};

//===----------------------------------------------------------------------===//
// Helper utilities.
//===----------------------------------------------------------------------===//

/// Convert ttl.kernel_thread -> ttkernel.thread if present, returning the
/// resolved thread type from whichever attribute exists.
static std::optional<ttk::ThreadType> convertThreadAttr(Operation *op) {
  if (auto a = op->getAttrOfType<ttk::ThreadTypeAttr>("ttkernel.thread")) {
    return a.getValue();
  }
  if (auto a = op->getAttrOfType<ttk::ThreadTypeAttr>(kKernelThreadAttrName)) {
    op->removeAttr(kKernelThreadAttrName);
    op->setAttr("ttkernel.thread", a);
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
static Value
getBufferAddressFromRuntimeArg(unsigned argIdx, Location loc,
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
    if (cbIndex < 0 || cbIndex >= kMaxCircularBuffers) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "cb_index " << cbIndex << " out of valid range [0, "
             << kMaxCircularBuffers - 1 << "]";
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

// Tile count: use the `num_tiles` attribute if present (per-subblock
// reserve/push), otherwise derive from the DFB type shape (full block).
static Value computeNumTiles(Operation *sourceOp, Value cb,
                             ConversionPatternRewriter &rewriter,
                             Location loc) {
  if (auto attr = sourceOp->getAttrOfType<IntegerAttr>("num_tiles")) {
    return arith::ConstantIntOp::create(rewriter, loc, attr.getInt(), 32);
  }
  auto ttlCbTy = getTTLCBType(cb);
  int64_t numTiles = ttlCbTy ? ttlCbTy.getElementsPerBlock() : 1;
  return arith::ConstantIntOp::create(rewriter, loc, numTiles, 32);
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

    auto convertedCb =
        utils::convertTTLCBToTTKernel(adaptor.getCb(), rewriter, loc);
    if (failed(convertedCb)) {
      return rewriter.notifyMatchFailure(op, "failed to convert CB operand");
    }

    Value numTiles = computeNumTiles(op, originalCb, rewriter, loc);
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
};

using CBReserveLowering =
    CBOpLowering<CBReserveOp, ttk::CBReserveBackOp, /*HasResult=*/true>;
using CBPushLowering =
    CBOpLowering<CBPushOp, ttk::CBPushBackOp, /*HasResult=*/false>;
using CBWaitLowering =
    CBOpLowering<CBWaitOp, ttk::CBWaitFrontOp, /*HasResult=*/true>;
using CBPopLowering =
    CBOpLowering<CBPopOp, ttk::CBPopFrontOp, /*HasResult=*/false>;

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

    ttk::PackTileOp::create(rewriter, loc, dstIndex, *cb, cbTileIndex,
                            /*out_of_order=*/true);

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
  if (llvm::isa<PipeType>(v.getType())) {
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

static bool isPipeReceiveCopy(CopyOp op) {
  return llvm::isa<PipeType>(op.getSrc().getType()) &&
         getAttachedCB(op.getDst());
}

static bool isPipeSendCopy(CopyOp op) {
  return llvm::isa<CircularBufferType>(op.getSrc().getType()) &&
         llvm::isa<PipeType>(op.getDst().getType());
}

static CopyOp findPipeReceiveCopy(Value value) {
  llvm::SmallPtrSet<Value, 16> seen;
  return traceTransferHandleSource<CopyOp>(
      value,
      [](Value source) {
        auto copyOp = source.getDefiningOp<CopyOp>();
        if (!copyOp) {
          return CopyOp();
        }
        if (isPipeReceiveCopy(copyOp)) {
          return copyOp;
        }
        return CopyOp();
      },
      seen);
}

static PipeTransferSendOp findPipeTransferSend(Value value) {
  llvm::SmallPtrSet<Value, 16> seen;
  return traceTransferHandleSource<PipeTransferSendOp>(
      value,
      [](Value source) { return source.getDefiningOp<PipeTransferSendOp>(); },
      seen);
}

static PipeTransferKind getPipeTransferKind(PipeTransferContract contract) {
  return isCollectiveTransfer(contract) ? PipeTransferKind::Collective
                                        : PipeTransferKind::PointToPoint;
}

static CreatePipeOp findCreatePipeForPipeValue(Value pipe) {
  llvm::SmallPtrSet<Value, 16> seen;
  return traceTransferHandleSource<CreatePipeOp>(
      pipe, [](Value source) { return source.getDefiningOp<CreatePipeOp>(); },
      seen);
}

static PipeTransferContract getPipeTransferContractForPipeValue(Value pipe) {
  if (CreatePipeOp createPipe = findCreatePipeForPipeValue(pipe)) {
    return getPipeTransferContract(createPipe);
  }
  // Function and block arguments do not carry CreatePipeOp attrs; use the
  // PipeType-derived contract only when no defining pipe op can be traced.
  auto pipeType = mlir::cast<PipeType>(traceUnrealizedCasts(pipe).getType());
  return pipeType.hasMultipleReceivers() ? PipeTransferContract::Collective
                                         : PipeTransferContract::PointToPoint;
}

static PipeTransferCreateOp createPipeTransfer(OpBuilder &builder, Location loc,
                                               Value pipe) {
  auto pipeType = mlir::cast<PipeType>(traceUnrealizedCasts(pipe).getType());
  PipeTransferContract contract = getPipeTransferContractForPipeValue(pipe);
  auto kindAttr = PipeTransferKindAttr::get(builder.getContext(),
                                            getPipeTransferKind(contract));
  auto expectedReceiversAttr =
      builder.getI64IntegerAttr(pipeType.getNumDests());
  return PipeTransferCreateOp::create(
      builder, loc, PipeTransferType::get(builder.getContext()), pipe, kindAttr,
      expectedReceiversAttr);
}

static Value getOrCreatePipeTransfer(
    OpBuilder &builder, Location loc, Value pipe,
    llvm::MapVector<Value, Value> &transferByDirectCreatePipe) {
  Value key = traceUnrealizedCasts(pipe);
  if (auto createPipe = key.getDefiningOp<CreatePipeOp>()) {
    auto it = transferByDirectCreatePipe.find(key);
    if (it != transferByDirectCreatePipe.end()) {
      return it->second;
    }
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(createPipe);
    auto transferOp = createPipeTransfer(builder, createPipe.getLoc(), key);
    transferByDirectCreatePipe[key] = transferOp.getTransfer();
    return transferOp.getTransfer();
  }

  // Non-direct pipe values can be block arguments or region results. A shared
  // cached transfer for those values would need dominance analysis; creating it
  // at the use site keeps the transfer local to the post/send that consumes it.
  return createPipeTransfer(builder, loc, pipe).getTransfer();
}

static LogicalResult verifyPipeTransferWaits(ModuleOp mod) {
  LogicalResult result = success();
  mod.walk(
      [&](PipeTransferWaitOp waitOp) {
        PipeTransferPostOp postOp =
            findPipeTransferPostForToken(waitOp.getToken());
        if (!postOp) {
          waitOp.emitError()
              << "requires token derived from ttl.pipe_transfer.post";
          result = failure();
          return;
        }
        auto waitTokenType =
            mlir::cast<PipeTokenType>(waitOp.getToken().getType());
        auto postTokenType =
            mlir::cast<PipeTokenType>(postOp.getToken().getType());
        if (waitTokenType.getPipeNetId() != postTokenType.getPipeNetId()) {
          waitOp.emitError()
              << "token pipeNetId must match pipe transfer post pipeNetId";
          result = failure();
        }
      });
  return result;
}

static LogicalResult expandPipeTransferOps(ModuleOp mod) {
  SmallVector<CreatePipeOp> createPipes;
  mod.walk([&](CreatePipeOp op) { createPipes.push_back(op); });

  SmallVector<CopyOp> receiveCopies;
  SmallVector<CopyOp> sendCopies;
  mod.walk([&](CopyOp op) {
    if (isPipeReceiveCopy(op)) {
      receiveCopies.push_back(op);
      return;
    }
    if (isPipeSendCopy(op)) {
      sendCopies.push_back(op);
    }
  });

  struct ReceiveWaitExpansion {
    WaitOp waitOp;
    int64_t pipeNetId;
  };
  SmallVector<ReceiveWaitExpansion> receiveWaits;
  LogicalResult result = success();
  mod.walk(
      [&](WaitOp waitOp) {
        auto handleType =
            mlir::dyn_cast<TransferHandleType>(waitOp.getXf().getType());
        if (!handleType || handleType.getKind()) {
          return;
        }
        CopyOp copyOp = findPipeReceiveCopy(waitOp.getXf());
        if (!copyOp) {
          waitOp.emitError()
              << "untyped transfer handle wait must reference a pipe receive "
                 "ttl.copy";
          result = failure();
          return;
        }
        auto pipeType = mlir::cast<PipeType>(
            traceUnrealizedCasts(copyOp.getSrc()).getType());
        receiveWaits.push_back({waitOp, pipeType.getPipeNetId()});
      });
  if (failed(result)) {
    return failure();
  }

  OpBuilder builder(mod.getContext());
  llvm::MapVector<Value, Value> transferByDirectCreatePipe;
  for (CreatePipeOp createPipe : createPipes) {
    builder.setInsertionPointAfter(createPipe);
    auto transferOp = createPipeTransfer(builder, createPipe.getLoc(),
                                         createPipe.getResult());
    transferByDirectCreatePipe[createPipe.getResult()] =
        transferOp.getTransfer();
  }

  for (CopyOp copyOp : receiveCopies) {
    auto pipeType =
        mlir::cast<PipeType>(traceUnrealizedCasts(copyOp.getSrc()).getType());
    builder.setInsertionPoint(copyOp);
    Value transfer = getOrCreatePipeTransfer(
        builder, copyOp.getLoc(), copyOp.getSrc(), transferByDirectCreatePipe);
    auto postOp = PipeTransferPostOp::create(
        builder, copyOp.getLoc(),
        PipeTokenType::get(builder.getContext(), pipeType.getPipeNetId()),
        transfer, copyOp.getDst());
    auto handleCast = UnrealizedConversionCastOp::create(
        builder, copyOp.getLoc(), copyOp.getResult().getType(),
        ValueRange{postOp.getToken()});
    copyOp.getResult().replaceAllUsesWith(handleCast.getResult(0));
    copyOp->erase();
  }

  for (CopyOp copyOp : sendCopies) {
    builder.setInsertionPoint(copyOp);
    Value transfer = getOrCreatePipeTransfer(
        builder, copyOp.getLoc(), copyOp.getDst(), transferByDirectCreatePipe);
    auto sendOp = PipeTransferSendOp::create(builder, copyOp.getLoc(),
                                             copyOp.getResult().getType(),
                                             transfer, copyOp.getSrc());
    copyOp.getResult().replaceAllUsesWith(sendOp.getXf());
    copyOp->erase();
  }

  for (const ReceiveWaitExpansion &wait : receiveWaits) {
    WaitOp waitOp = wait.waitOp;
    builder.setInsertionPoint(waitOp);
    auto tokenCast = UnrealizedConversionCastOp::create(
        builder, waitOp.getLoc(),
        PipeTokenType::get(builder.getContext(), wait.pipeNetId),
        ValueRange{waitOp.getXf()});
    PipeTransferWaitOp::create(builder, waitOp.getLoc(),
                               tokenCast.getResult(0));
    waitOp->erase();
  }

  return success();
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

/// Lower a tensor_slice<->CB copy in the given direction.
/// Read: tensor_slice -> CB (noc_async_read_tile, get_write_ptr)
/// Write: CB -> tensor_slice (noc_async_write_tile, get_read_ptr)
static LogicalResult lowerTensorCBCopy(CopyOp op, TensorSliceOp sliceOp,
                                       Value cb, NocCopyDirection direction,
                                       ConversionPatternRewriter &rewriter,
                                       const TypeConverter &typeConverter) {
  auto loc = op.getLoc();
  Value tensor = sliceOp.getTensor();
  auto startIndices = sliceOp.getIndices();

  FailureOr<TensorAccessorInfo> accessorInfo =
      getTensorAccessorInfo(tensor, op, rewriter);
  if (failed(accessorInfo)) {
    return failure();
  }

  auto cbType = getTTLCBType(cb);
  if (!cbType) {
    return rewriter.notifyMatchFailure(op, "failed to get CB type");
  }

  SmallVector<int64_t> tensorGridShape = getTileGridShapeFromValue(tensor);
  unsigned tensorRank = tensorGridShape.size();

  auto cbShape = cbType.getShape();

  if (startIndices.size() != tensorRank) {
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << "tensor_slice index count (" << startIndices.size()
           << ") does not match tensor rank (" << tensorRank << ")";
    });
  }

  // cbRank <= tensorRank is guaranteed upstream: CopyOp enforces DFB rank ==
  // slice result rank, and TensorSliceOp enforces result rank <= tensor rank.
  assert(cbShape.size() <= tensorRank && "CB rank exceeds tensor rank");

  Value bankBase =
      getBufferAddressFromRuntimeArg(accessorInfo->argIdx, loc, rewriter);
  Value accessor =
      materializeTensorAccessor(tensor, bankBase, *accessorInfo, rewriter);

  auto cbConverted = utils::convertTTLCBToTTKernel(cb, rewriter, loc);
  assert(succeeded(cbConverted) && "preflight checked DFB type");

  bool isRead = direction == NocCopyDirection::Read;
  Value cbPtr =
      isRead
          ? ttk::GetWritePtrOp::create(rewriter, loc, *cbConverted).getResult()
          : ttk::GetReadPtrOp::create(rewriter, loc, *cbConverted).getResult();

  // Rank-reducing slice: the leading (tensorRank - cbRank) tensor dims are
  // squeezed via scalar indices (validated at slice creation). CB iteration
  // vars map to the trailing dims; squeezed dims contribute startIndices[d]
  // directly with no IV adder.
  unsigned cbRank = cbShape.size();
  unsigned rankDiff = tensorRank - cbRank;

  auto indexTy = rewriter.getIndexType();
  auto cbPtrIdx = arith::IndexCastOp::create(rewriter, loc, indexTy, cbPtr);
  auto pageSizeIdx = arith::ConstantIndexOp::create(
      rewriter, loc, accessorInfo->pageSizeBytes);
  auto i32Ty = rewriter.getI32Type();

  SmallVector<int64_t> cbBounds(cbShape.begin(), cbShape.end());

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
                                          accessor, cbAddr);
        } else {
          ttk::NocAsyncWriteTileOp::create(loopBuilder, bodyLoc,
                                           tensorTileIdx32, accessor, cbAddr);
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
  using OpConversionPattern::OpConversionPattern;

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
                               NocCopyDirection::Read, rewriter,
                               *typeConverter);
    }

    // CB -> TensorSlice: write tiles from circular buffer to tensor.
    auto sliceOp = dst.getDefiningOp<TensorSliceOp>();
    if (!sliceOp) {
      return rewriter.notifyMatchFailure(
          op, "tensor_slice destination must come from ttl.tensor_slice op");
    }
    return lowerTensorCBCopy(op, sliceOp, adaptor.getSrc(),
                             NocCopyDirection::Write, rewriter, *typeConverter);
  }
};

struct PipeTransferPostLowering : OpConversionPattern<PipeTransferPostOp> {
  PipeTransferPostLowering(const TypeConverter &typeConverter,
                           MLIRContext *context,
                           const PipeResourcePlan &pipeResourcePlan)
      : OpConversionPattern(typeConverter, context),
        pipeResourcePlan(pipeResourcePlan) {}

  LogicalResult
  matchAndRewrite(PipeTransferPostOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The receive destination is inspected for its TTL DFB provenance
    // (`ttl.cb_reserve`, `ttl.attach_cb`, and slice offset), so this lowering
    // must use the original SSA value rather than the converted adaptor value.
    return lowerPipeTransferPost(op, op.getDst(), pipeResourcePlan, rewriter);
  }

private:
  const PipeResourcePlan &pipeResourcePlan;
};

struct PipeTransferSendLowering : OpConversionPattern<PipeTransferSendOp> {
  PipeTransferSendLowering(const TypeConverter &typeConverter,
                           MLIRContext *context,
                           const PipeResourcePlan &pipeResourcePlan)
      : OpConversionPattern(typeConverter, context),
        pipeResourcePlan(pipeResourcePlan) {}

  LogicalResult
  matchAndRewrite(PipeTransferSendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // DFB -> Pipe: source core sends data to the pipe receivers.
    // Determine DFB access context: consumer (cb_wait/cb_pop) vs producer
    // (cb_reserve/cb_push). This controls whether the source address comes
    // from the DFB read pointer or write pointer.
    DominanceInfo domInfo(op->getParentOfType<func::FuncOp>());
    bool isConsumerCB =
        llvm::any_of(op.getSrc().getUsers(), [&](Operation *user) {
          return mlir::isa<CBWaitOp>(user) &&
                 user->getOperand(0) == op.getSrc() &&
                 domInfo.dominates(user, op);
        });
    return lowerPipeTransferSend(op, adaptor.getSrc(), isConsumerCB,
                                 pipeResourcePlan, rewriter);
  }

private:
  const PipeResourcePlan &pipeResourcePlan;
};

struct PipeTransferWaitLowering : OpConversionPattern<PipeTransferWaitOp> {
  PipeTransferWaitLowering(const TypeConverter &typeConverter,
                           MLIRContext *context,
                           const PipeNetCounterMap *pipeNetCounters,
                           const PipeResourcePlan &pipeResourcePlan)
      : OpConversionPattern(typeConverter, context),
        pipeNetCounters(pipeNetCounters), pipeResourcePlan(pipeResourcePlan) {}

  LogicalResult
  matchAndRewrite(PipeTransferWaitOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerPipeTransferWait(op, pipeNetCounters, pipeResourcePlan,
                                 rewriter);
  }

private:
  const PipeNetCounterMap *pipeNetCounters;
  const PipeResourcePlan &pipeResourcePlan;
};

struct WaitLowering : OpConversionPattern<WaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (findPipeTransferSend(op.getXf())) {
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
    auto kind = getTransferKindFromHandleType(adaptor.getXf().getType());
    if (!kind) {
      return op.emitError("untyped transfer handle survived pipe receive "
                          "expansion");
    }
    if (*kind == TransferKind::read) {
      ttk::NocAsyncReadBarrierOp::create(rewriter, op.getLoc(), Value());
    } else if (*kind == TransferKind::write) {
      ttk::NocAsyncWriteBarrierOp::create(rewriter, op.getLoc(), Value());
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
    op->removeAttr("ttl.noc_index");
    op->setAttr("ttkernel.thread", ttlAttr);

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

/// Return the scalar type and matching integer type for a raw element access.
/// f32 -> (i32, 32), bf16 -> (i16, 16).
static std::pair<Type, unsigned> getIntTypeForFloat(MLIRContext *ctx,
                                                    Type floatTy) {
  if (floatTy.isF32()) {
    return {IntegerType::get(ctx, 32), 32};
  }
  assert(floatTy.isBF16());
  return {IntegerType::get(ctx, 16), 16};
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

struct RawElementReadLowering : OpConversionPattern<RawElementReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RawElementReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto blockType = mlir::cast<RankedTensorType>(op.getBlock().getType());
    Type scalarTy = op.getResult().getType();
    auto [intTy, elemWidth] =
        getIntTypeForFloat(rewriter.getContext(), scalarTy);

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

struct RawElementWriteLowering : OpConversionPattern<RawElementWriteOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RawElementWriteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto blockType = mlir::cast<RankedTensorType>(op.getBlock().getType());
    Type scalarTy = op.getValue().getType();
    auto [intTy, elemWidth] =
        getIntTypeForFloat(rewriter.getContext(), scalarTy);

    auto cb = resolveCBForRawElement(adaptor.getBlock(), op.getBlock(),
                                     rewriter, loc, this->getTypeConverter());
    if (failed(cb)) {
      return rewriter.notifyMatchFailure(op, "block does not trace to a CB");
    }

    auto intVal =
        utils::materializeIntBits(adaptor.getValue(), intTy, rewriter, loc);
    if (failed(intVal)) {
      return rewriter.notifyMatchFailure(
          op, "could not materialize integer bits from float value");
    }

    auto [l1Ptr, offset] =
        emitL1PtrAndOffset(*cb, op.getBlock(), blockType, adaptor.getCoords(),
                           elemWidth, rewriter, loc);

    ttk::StoreToL1Op::create(rewriter, loc, *intVal, l1Ptr, offset);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TTLConvertTTLToTTKernelPass helper methods
//===----------------------------------------------------------------------===//

/// Phase 1: Lower TTL ops (bind_cb, copy, wait, cb ops, store) to TTKernel.
static LogicalResult
lowerTTLOpsToTTKernel(ModuleOp mod, MLIRContext &ctx,
                      TTLToTTKernelTypeConverter &typeConverter,
                      StringRef passName) {
  ConversionTarget target(ctx);
  target.addIllegalDialect<tt::ttl::TTLDialect>();
  target.addLegalDialect<affine::AffineDialect, arith::ArithDialect,
                         BuiltinDialect, memref::MemRefDialect, scf::SCFDialect,
                         func::FuncDialect, tensor::TensorDialect,
                         ttkernel::TTKernelDialect>();

  // Structural ops remain legal (converted elsewhere or kept as-is).
  target.addLegalOp<ComputeOp, YieldOp, AttachCBOp, DstIndexOp>();
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
    if (llvm::isa<RawElementReadOp, RawElementWriteOp>(op)) {
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

  // Validate explicit transfer IR before expansion mutates public pipe copies.
  if (failed(verifyPipeTransferWaits(mod))) {
    return failure();
  }
  if (failed(expandPipeTransferOps(mod))) {
    return failure();
  }
  // Expansion creates pipe_transfer.wait from public ttl.wait; validate those
  // token chains before graph and resource planning.
  if (failed(verifyPipeTransferWaits(mod))) {
    return failure();
  }

  // Validate receiver DFB consistency before lowering emits the pipe
  // synchronization protocol.
  auto pipeGraphOrErr = PipeGraph::build(mod);
  if (failed(pipeGraphOrErr)) {
    return failure();
  }

  // Per-PipeNet runtime counters for cumulative receive wait_min.
  PipeNetCounterMap pipeNetCounters;
  allocatePipeNetReceiveCounters(mod, pipeNetCounters);

  // Per-net-id pipe list, shared by IsSrc/IsDst/IsActive lowerings so they
  // don't walk the module per match.
  PipeNetIndex pipeNetIndex;
  buildPipeNetIndex(mod, pipeNetIndex);
  PipeResourcePlan pipeResourcePlan;
  if (failed(buildPipeResourcePlan(mod, pipeResourcePlan))) {
    return failure();
  }
  PipeResourceRequirements pipeResourceRequirements =
      getPipeResourceRequirements(pipeResourcePlan);
  if (failed(verifyPipeResourcePlanFitsHardware(mod, pipeResourcePlan,
                                                pipeResourceRequirements))) {
    return failure();
  }
  mod->setAttr(kPipeSyncSemaphoreCountAttrName,
               IntegerAttr::get(IntegerType::get(&ctx, 64),
                                pipeResourceRequirements.syncSemaphoreCount));
  if (pipeResourceRequirements.globalSemaphoreCount > 0) {
    mod->setAttr(
        kPipeGlobalSemaphoreCountAttrName,
        IntegerAttr::get(IntegerType::get(&ctx, 64),
                         pipeResourceRequirements.globalSemaphoreCount));
  }
  if (pipeResourceRequirements.sramScratchBytes > 0) {
    mod->setAttr(kPipeSramScratchBytesAttrName,
                 IntegerAttr::get(IntegerType::get(&ctx, 64),
                                  pipeResourceRequirements.sramScratchBytes));
  }
  // [Device 2.0] The kPipeSyncSemaphoreCountAttrName,
  // kPipeGlobalSemaphoreCountAttrName, and kPipeSramScratchBytesAttrName attrs
  // are the current host/runtime ABI for pipe resource binding. Keep the
  // allocation decision in this compiler plan so future typed device APIs only
  // change runtime binding code.

  RewritePatternSet patterns(&ctx);
  patterns.add<CopyLowering>(typeConverter, &ctx);
  patterns.add<PipeTransferPostLowering, PipeTransferSendLowering>(
      typeConverter, &ctx, pipeResourcePlan);
  patterns.add<PipeTransferWaitLowering>(typeConverter, &ctx, &pipeNetCounters,
                                         pipeResourcePlan);
  patterns.add<BindCBLowering, TensorSliceLowering, WaitLowering,
               CBReserveLowering, CBPushLowering, CBWaitLowering, CBPopLowering,
               TileStoreLowering, StoreLowering, CoreXLowering, CoreYLowering,
               RawElementReadLowering, RawElementWriteLowering>(typeConverter,
                                                                &ctx);
  populatePipeLoweringPatterns(patterns, typeConverter, pipeNetIndex);
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

/// Phase 2: Lower tile compute ops and DST lifecycle ops to TTKernel.
/// Tile compute ops are identified by TTLTileComputeOpTrait. ttl.compute is
/// kept legal here because it is lowered to loops in an earlier pass
/// (ttl-lower-to-loops).
static LogicalResult
lowerTileOpsToTTKernel(ModuleOp mod, MLIRContext &ctx,
                       TTLToTTKernelTypeConverter &typeConverter,
                       bool reduceFullFp32) {
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
  populateTTLTileOpsToTTKernelPatterns(&typeConverter, computePatterns,
                                       reduceFullFp32);
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

struct TTLConvertTTLToTTKernelPass
    : impl::TTLConvertTTLToTTKernelBase<TTLConvertTTLToTTKernelPass> {
  using TTLConvertTTLToTTKernelBase::TTLConvertTTLToTTKernelBase;

  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    ModuleOp mod = getOperation();
    TTLToTTKernelTypeConverter typeConverter;

    // Phase 0: Expand DstSectionOp into four TTL sync ops. This inlines the
    // DstSectionOp body and inserts acquire/commit/wait/release around it,
    // with stores reordered to the pack phase (after wait).
    expandDstSections(mod);

    // Phase 1: Lower TTL ops to TTKernel (bind_cb, copy, wait, cb ops, store)
    if (failed(lowerTTLOpsToTTKernel(mod, ctx, typeConverter, getName()))) {
      signalPassFailure();
      return;
    }

    // Phase 2: Lower tile compute ops to TTKernel (tile_add, tile_mul, ...)
    if (failed(
            lowerTileOpsToTTKernel(mod, ctx, typeConverter, reduceFullFp32))) {
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
