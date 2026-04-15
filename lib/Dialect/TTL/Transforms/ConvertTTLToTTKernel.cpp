// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h" // IWYU pragma: keep

#include "PipeGraph.h"
#include "PipeLowering.h"
#include "ttlang/Dialect/TTKernel/Transforms/TTKernelCleanupPatterns.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>

namespace mlir::tt::ttl {
#define GEN_PASS_DEF_TTLCONVERTTTLTOTTKERNEL
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

using mlir::func::FuncOp;
namespace ttk = mlir::tt::ttkernel;

// Start index in compile-time args for TA static metadata (is_sharded,
// is_dram). CTA layout is [CBs, TAs], so this is the number of CBs.
constexpr llvm::StringLiteral kBaseCTAIndexAttr = "ttl.base_cta_index";

// Maps local args to global tensor indices for common runtime args (buffer
// addresses). CRTA is filtered per-thread, containing only addresses for
// tensors this thread uses.
constexpr llvm::StringLiteral kCRTAIndicesAttr = "ttl.crta_indices";

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
  if (auto a = op->getAttrOfType<ttk::ThreadTypeAttr>("ttl.kernel_thread")) {
    op->removeAttr("ttl.kernel_thread");
    op->setAttr("ttkernel.thread", a);
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
  auto idxConst = arith::ConstantIndexOp::create(rewriter, loc, *argIdx);
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

} // namespace

//===----------------------------------------------------------------------===//
// PipeGraph implementation
//===----------------------------------------------------------------------===//

FailureOr<PipeGraph> PipeGraph::build(ModuleOp mod) {
  PipeGraph graph;

  // Find all Pipe->CB copies (receiver side) and extract CB index
  LogicalResult walkResult = success();
  mod.walk([&](CopyOp copyOp) {
    if (failed(walkResult)) {
      return;
    }
    auto srcPipeType = dyn_cast<PipeType>(copyOp.getSrc().getType());
    if (!srcPipeType) {
      return;
    }

    // Found Pipe->CB copy: this is the receiver side
    Value dstCB = copyOp.getDst();
    auto cbType = dyn_cast<CircularBufferType>(dstCB.getType());
    if (!cbType) {
      copyOp.emitWarning("pipe copy destination is not a circular buffer");
      return;
    }

    // Trace to the BindCBOp to get the CB index
    Value cbVal = traceUnrealizedCasts(dstCB);
    auto bindOp = cbVal.getDefiningOp<BindCBOp>();
    if (!bindOp) {
      copyOp.emitWarning("could not trace pipe receiver to a BindCBOp");
      return;
    }

    int64_t cbIndex = bindOp.getCbIndex().getSExtValue();
    walkResult = graph.addReceiverCB(
        srcPipeType.getSrcX(), srcPipeType.getSrcY(),
        srcPipeType.getDstStartX(), srcPipeType.getDstStartY(),
        srcPipeType.getDstEndX(), srcPipeType.getDstEndY(),
        srcPipeType.getPipeNetId(), cbIndex, cbType.getBlockCount(),
        copyOp.getLoc(), copyOp);
  });

  if (failed(walkResult)) {
    return failure();
  }

  graph.assignGatherSlotIndices();

  if (failed(graph.verifyGatherBlockCounts())) {
    return failure();
  }

  return graph;
}

namespace {

enum class CopyOperandKind { TensorSlice, CircularBuffer, Pipe, Unknown };

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

  auto baseCTAAttr = parentFunc->getAttrOfType<IntegerAttr>(kBaseCTAIndexAttr);
  if (!baseCTAAttr) {
    return op->emitError("function missing ")
           << kBaseCTAIndexAttr << " attribute";
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

/// Create a TensorAccessor from a tensor type, bank base address, and
/// pre-validated page size. The bankBase should come from runtime args via
/// getBufferAddressFromRuntimeArg; pageSizeBytes from getValidatedPageSize.
static FailureOr<Value>
materializeTensorAccessor(Value tensor, Value bankBase, int64_t pageSizeBytes,
                          Operation *op, ConversionPatternRewriter &rewriter) {
  auto argIdx = getTensorFuncArgIndex(tensor);
  if (failed(argIdx)) {
    // Callers (lowerTensorCBCopy) already guard this via
    // getBufferAddressFromRuntimeArg, so this is unreachable.
    llvm_unreachable("tensor must be a function argument");
  }

  auto loc = tensor.getLoc();

  auto ctaInfo = getBaseCTAAndGlobalTensorIdx(*argIdx, op);
  if (failed(ctaInfo)) {
    return failure();
  }
  auto [baseCTA, globalTensorIdx] = *ctaInfo;

  auto pageSize =
      arith::ConstantIntOp::create(rewriter, loc, pageSizeBytes, 32);

  return buildTensorAccessor(loc, rewriter, baseCTA, globalTensorIdx,
                             static_cast<int32_t>(*argIdx), bankBase, pageSize);
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

  bool allOne = llvm::all_of(tileBounds, [](int64_t d) { return d == 1; });
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
                     [&](OpBuilder &b, Location bodyLoc, ValueRange ivs) {
                       emitBody(b, bodyLoc, ivs);
                     });
}

/// Compute a linearized (row-major) index from ND coordinates and shape.
/// index = coords[0] * (shape[1]*...*shape[N-1]) + ... + coords[N-1]
static Value linearizeNDIndex(OpBuilder &builder, Location loc,
                              ValueRange coords, ArrayRef<int64_t> shape) {
  assert(coords.size() == shape.size() && "coords and shape rank mismatch");
  Value result = arith::ConstantIndexOp::create(builder, loc, 0);
  for (size_t i = 0; i < coords.size(); ++i) {
    // stride = product of shape[i+1..N-1]
    int64_t stride = 1;
    for (size_t j = i + 1; j < shape.size(); ++j) {
      stride *= shape[j];
    }
    Value strideVal = arith::ConstantIndexOp::create(builder, loc, stride);
    Value term = arith::MulIOp::create(builder, loc, coords[i], strideVal);
    result = arith::AddIOp::create(builder, loc, result, term);
  }
  return result;
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

  // Validate layout and get page size once.
  auto pageSizeBytes = getValidatedPageSize(tensor, op);
  if (failed(pageSizeBytes)) {
    return failure();
  }

  auto bankBase = getBufferAddressFromRuntimeArg(tensor, loc, rewriter);
  if (failed(bankBase)) {
    return rewriter.notifyMatchFailure(
        op, "tensor must be a function argument for runtime arg mapping");
  }

  auto accessor = materializeTensorAccessor(tensor, *bankBase, *pageSizeBytes,
                                            op, rewriter);
  if (failed(accessor)) {
    return failure();
  }

  auto cbConverted = utils::convertTTLCBToTTKernel(cb, rewriter, loc);
  if (failed(cbConverted)) {
    return rewriter.notifyMatchFailure(op, "failed to convert CB operand");
  }

  bool isRead = direction == NocCopyDirection::Read;
  Value cbPtr =
      isRead
          ? ttk::GetWritePtrOp::create(rewriter, loc, *cbConverted).getResult()
          : ttk::GetReadPtrOp::create(rewriter, loc, *cbConverted).getResult();

  // Get CB shape for loop bounds.
  auto cbType = getTTLCBType(cb);
  if (!cbType) {
    return rewriter.notifyMatchFailure(op, "failed to get CB type");
  }
  auto cbShape = cbType.getShape();

  // Tensor grid shape for linearization.
  auto tensorGridShape = getTileGridShapeFromValue(tensor);
  unsigned tensorRank = tensorGridShape.size();

  if (startIndices.size() != tensorRank) {
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << "tensor_slice index count (" << startIndices.size()
           << ") does not match tensor rank (" << tensorRank << ")";
    });
  }

  if (cbShape.size() != tensorRank) {
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << "CB shape rank (" << cbShape.size()
           << ") does not match tensor rank (" << tensorRank << ")";
    });
  }

  auto indexTy = rewriter.getIndexType();
  auto cbPtrIdx = arith::IndexCastOp::create(rewriter, loc, indexTy, cbPtr);
  auto pageSizeIdx =
      arith::ConstantIndexOp::create(rewriter, loc, *pageSizeBytes);
  auto i32Ty = rewriter.getI32Type();

  SmallVector<int64_t> cbBounds(cbShape.begin(), cbShape.end());

  emitTileLoop(
      rewriter, loc, cbBounds,
      [&](OpBuilder &b, Location bodyLoc, ValueRange cbIVs) {
        // Tensor coordinates: start index + CB loop IV for each dimension.
        SmallVector<Value> tensorCoords;
        for (unsigned d = 0; d < tensorRank; ++d) {
          Value coord =
              arith::AddIOp::create(b, bodyLoc, startIndices[d], cbIVs[d]);
          tensorCoords.push_back(coord);
        }

        Value tensorTileIdx =
            linearizeNDIndex(b, bodyLoc, tensorCoords, tensorGridShape);

        Value cbTileIdx = linearizeNDIndex(b, bodyLoc, cbIVs, cbBounds);

        // Compute CB address: cbPtr + cbTileIdx * pageSize
        Value byteOffset =
            arith::MulIOp::create(b, bodyLoc, cbTileIdx, pageSizeIdx);
        Value cbAddrIdx =
            arith::AddIOp::create(b, bodyLoc, cbPtrIdx, byteOffset);

        // Cast to i32 for NOC operation.
        Value tensorTileIdx32 =
            arith::IndexCastOp::create(b, bodyLoc, i32Ty, tensorTileIdx);
        Value cbAddr = arith::IndexCastOp::create(b, bodyLoc, i32Ty, cbAddrIdx);

        if (isRead) {
          ttk::NocAsyncReadTileOp::create(b, bodyLoc, tensorTileIdx32,
                                          *accessor, cbAddr);
        } else {
          ttk::NocAsyncWriteTileOp::create(b, bodyLoc, tensorTileIdx32,
                                           *accessor, cbAddr);
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
               const PipeGraph *pipeGraph)
      : OpConversionPattern(typeConverter, context), pipeGraph(pipeGraph) {}

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

    // Pipe transfers: CB <-> Pipe
    if (srcIsCB && dstIsPipe) {
      // CB -> Pipe: source core multicasts data to destination cores
      // Look up receiver CB info for gather patterns
      const ReceiverCBInfo *receiverInfo = nullptr;
      if (pipeGraph) {
        auto pipeType = llvm::cast<PipeType>(adaptor.getDst().getType());
        receiverInfo = pipeGraph->getReceiverInfo(
            pipeType.getSrcX(), pipeType.getSrcY(), pipeType.getDstStartX(),
            pipeType.getDstStartY(), pipeType.getDstEndX(),
            pipeType.getDstEndY(), pipeType.getPipeNetId());
      }
      // Determine CB access context: consumer (cb_wait/cb_pop) vs producer
      // (cb_reserve/cb_push). This controls whether we read from the CB's
      // read pointer or write pointer for the pipe source address.
      // Use dominance to correctly handle CBs in nested regions (e.g.,
      // inside scf.if from pipe callbacks) and CBs used in both roles.
      DominanceInfo domInfo(op->getParentOfType<func::FuncOp>());
      bool isConsumerCB = llvm::any_of(src.getUsers(), [&](Operation *user) {
        return isa<CBWaitOp>(user) && user->getOperand(0) == src &&
               domInfo.dominates(user, op);
      });
      return lowerCBToPipe(op, adaptor.getSrc(), adaptor.getDst(), receiverInfo,
                           isConsumerCB, rewriter);
    }
    if (srcIsPipe && dstIsCB) {
      // Pipe -> CB: destination receives data via multicast from source
      return lowerPipeToCB(op, adaptor.getSrc(), adaptor.getDst(), pipeGraph,
                           rewriter);
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

private:
  const PipeGraph *pipeGraph;
};

struct WaitLowering : OpConversionPattern<WaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO(ttl): Lower ttl.wait to TRID-specific barriers keyed by the transfer
    // handle (read vs write barrier based on transfer direction). Issue: #87.
    //
    // MVP behavior: emit the corresponding global barrier based on transfer
    // direction. Untyped handles (no kind) are no-ops - used for pipe receives
    // where data arrives via multicast and no local barrier is needed.
    auto kind = getTransferKindFromHandleType(adaptor.getXf().getType());
    if (!kind) {
      // No transfer kind means no barrier needed (e.g., pipe receive where
      // data arrives via multicast from source core).
      rewriter.eraseOp(op);
      return success();
    }
    if (*kind == TransferKind::read) {
      ttk::NocAsyncReadBarrierOp::create(rewriter, op.getLoc());
    } else if (*kind == TransferKind::write) {
      ttk::NocAsyncWriteBarrierOp::create(rewriter, op.getLoc());
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
    auto ttlAttr = op->getAttrOfType<ttk::ThreadTypeAttr>("ttl.kernel_thread");
    if (!ttlAttr || ttlAttr.getValue() != ttk::ThreadType::Noc) {
      return failure();
    }
    op->removeAttr("ttl.kernel_thread");
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
                         BuiltinDialect, scf::SCFDialect, func::FuncDialect,
                         tensor::TensorDialect, ttkernel::TTKernelDialect>();

  // Structural ops remain legal (converted elsewhere or kept as-is).
  target.addLegalOp<ComputeOp, YieldOp, AttachCBOp>();

  // DST lifecycle ops are not tile compute ops; keep them legal until the
  // tile ops lowering phase.
  target.addLegalOp<TileRegsAcquireOp, TileRegsCommitOp, TileRegsWaitOp,
                    TileRegsReleaseOp>();

  // SignpostOp and DPrintOp are lowered in separate EmitC passes.
  target.addLegalOp<SignpostOp, DPrintOp>();

  // Tile compute ops and data movement ops (copy_tile, copy_dst) remain legal
  // until the tile ops lowering phase.
  target.addDynamicallyLegalDialect<tt::ttl::TTLDialect>([](Operation *op) {
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

  // Build pipe graph to track receiver CB addresses for gather patterns.
  // This must happen before lowering so we can look up receiver info.
  auto pipeGraphOrErr = PipeGraph::build(mod);
  if (failed(pipeGraphOrErr)) {
    return failure();
  }
  PipeGraph pipeGraph = std::move(*pipeGraphOrErr);

  RewritePatternSet patterns(&ctx);
  // CopyLowering needs the pipe graph for gather pattern receiver CB lookup
  patterns.add<CopyLowering>(typeConverter, &ctx, &pipeGraph);
  patterns.add<BindCBLowering, TensorSliceLowering, WaitLowering,
               CBReserveLowering, CBPushLowering, CBWaitLowering, CBPopLowering,
               TileStoreLowering, StoreLowering, CoreXLowering, CoreYLowering>(
      typeConverter, &ctx);
  populatePipeLoweringPatterns(patterns, typeConverter);
  populateFunctionOpInterfaceTypeConversionPattern(
      func::FuncOp::getOperationName(), patterns, typeConverter);

  FrozenRewritePatternSet frozen(std::move(patterns));
  std::string diagMessage;
  if (utils::applyPartialConversionWithDiag(mod, target, frozen, passName,
                                            diagMessage)) {
    mod.emitError() << diagMessage;
    return failure();
  }

  // Apply post-conversion cleanup patterns (e.g., barrier deduplication).
  RewritePatternSet cleanupPatterns(&ctx);
  ttkernel::populateTTKernelCleanupPatterns(cleanupPatterns);
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
  computeTarget.addLegalOp<ComputeOp, YieldOp>();

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
        if (isa<TileRegsAcquireOp, TileRegsCommitOp, TileRegsWaitOp,
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
  cleanupTarget.addIllegalOp<ComputeOp, YieldOp>();

  RewritePatternSet structuralPatterns(&ctx);
  structuralPatterns.add<AttachCBLowering>(typeConverter, &ctx);
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
    if (isa<tensor::ExtractOp, tensor::ExtractSliceOp, tensor::EmptyOp>(op) &&
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
    if (isa<TileStoreOp>(&op)) {
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
