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
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/Utils/ConversionUtils.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"      // IWYU pragma: keep
#include "ttmlir/Dialect/TTNN/IR/TTNNOpsAttrs.h" // IWYU pragma: keep
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

// Start index in compile-time args for TA static metadata (is_sharded,
// is_dram). CTA layout is [CBs, TAs], so this is the number of CBs.
constexpr llvm::StringLiteral kBaseCTAIndexAttr = "ttl.base_cta_index";
// Maps local args to global tensor indices for common runtime args (buffer
// addresses). CRTA is filtered per-thread, containing only addresses for
// tensors this thread uses.
constexpr llvm::StringLiteral kCRTAIndicesAttr = "ttl.crta_indices";

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

/// Compute linearized CB tile index from enclosing scf.for loops.
/// For nested loops with IVs [iv0, iv1, ...] and bounds [ub0, ub1, ...],
/// computes: iv0 * (ub1 * ub2 * ...) + iv1 * (ub2 * ...) + ...
/// Returns constant 0 if not inside any loops (single-tile case).
static Value computeCBTileIndexFromLoops(Operation *op, OpBuilder &builder) {
  Location loc = op->getLoc();

  SmallVector<scf::ForOp> loops;
  Operation *parent = op->getParentOp();
  while (parent) {
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      loops.push_back(forOp);
    }
    parent = parent->getParentOp();
  }

  if (loops.empty()) {
    return builder.create<arith::ConstantIndexOp>(loc, 0);
  }

  // Validate assumptions: all loops have step=1 and lower bound=0.
  for (auto loop : loops) {
    auto lb = getConstantIntValue(loop.getLowerBound());
    assert(lb && *lb == 0 &&
           "computeCBTileIndexFromLoops: expected lower bound of 0");
    auto ub = getConstantIntValue(loop.getUpperBound());
    assert(ub && "computeCBTileIndexFromLoops: expected constant upper bound");
    auto step = getConstantIntValue(loop.getStep());
    assert(step && *step == 1 &&
           "computeCBTileIndexFromLoops: expected step of 1");
  }

  // Process in reverse order (innermost-first) without mutating the vector.
  Value linearIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
  for (auto [i, loop] : llvm::enumerate(llvm::reverse(loops))) {
    Value iv = loop.getInductionVar();

    // Stride = product of upper bounds of more-nested loops.
    Value stride = builder.create<arith::ConstantIndexOp>(loc, 1);
    for (auto innerLoop : llvm::drop_begin(llvm::reverse(loops), i + 1)) {
      stride =
          builder.create<arith::MulIOp>(loc, stride, innerLoop.getUpperBound());
    }

    Value term = builder.create<arith::MulIOp>(loc, iv, stride);
    linearIdx = builder.create<arith::AddIOp>(loc, linearIdx, term);
  }

  return linearIdx;
}

/// Build a TensorAccessor from CTA/CRTA indices, bank base, and page size.
/// ctaIndex: Index into compile-time args where tensor config starts.
/// crtaIndex: Index into compile-runtime args (typically 0).
static Value buildTensorAccessor(Location loc,
                                 ConversionPatternRewriter &rewriter,
                                 int32_t ctaIndex, int32_t crtaIndex,
                                 Value bankBase, Value pageSize) {
  auto ctaConst = rewriter.create<arith::ConstantIntOp>(loc, ctaIndex, 32);
  auto crtaConst = rewriter.create<arith::ConstantIntOp>(loc, crtaIndex, 32);
  auto args = rewriter.create<ttk::TensorAccessorArgsOp>(
      loc, ctaConst.getResult(), crtaConst.getResult(),
      /*prev_args=*/Value(), /*cta_expr=*/nullptr, /*crta_expr=*/nullptr);
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

// num_pages = product of CB shape dimensions (elements per block).
// Used by CBOpLowering template; [[maybe_unused]] silences linter warning.
[[maybe_unused]] static Value
computeNumPages(Value cb, ConversionPatternRewriter &rewriter, Location loc) {
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

    auto convertedCb =
        utils::convertTTLCBToTTKernel(adaptor.getCb(), rewriter, loc);
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

struct StoreLowering : OpConversionPattern<StoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Trace from the view back to the CB.
    auto cb = getCBFromView(adaptor.getView());
    if (failed(cb)) {
      return rewriter.notifyMatchFailure(
          op, "view must come from ttl.cb_reserve (unrealized cast from CB)");
    }

    // Get the DST index from the tile value's dst_idx attribute.
    // The DST assignment pass (ttl-tile-and-assign-dst) should run before this
    // pass and annotates tile-producing operations with DST register indices.
    // If the attribute is missing, we default to DST index 0.
    auto tileValue = adaptor.getTile();
    Value dstIndex;

    if (auto defOp = tileValue.getDefiningOp()) {
      if (auto dstIdxAttr =
              defOp->getAttrOfType<IntegerAttr>(kDstIdxAttrName)) {
        dstIndex =
            rewriter.create<arith::ConstantIndexOp>(loc, dstIdxAttr.getInt());
      }
    }

    // Default to DST index 0 if no attribute is found.
    // This can happen in unit tests or if DST assignment hasn't run.
    if (!dstIndex) {
      dstIndex = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    }

    // Compute CB tile index from enclosing loops for multi-tile cases.
    auto cbTileIndex = computeCBTileIndexFromLoops(op, rewriter);
    rewriter.create<ttk::PackTileOp>(loc, dstIndex, *cb, cbTileIndex,
                                     /*out_of_order=*/false);

    rewriter.eraseOp(op);
    return success();
  }
};

enum class CopySourceKind {
  TensorAccessor,
  TensorSlice,
  CircularBuffer,
  Pipe,
  Unknown
};
enum class CopyDestKind {
  TensorAccessor,
  TensorSlice,
  CircularBuffer,
  Pipe,
  Unknown
};

static bool isTensorAccessorLike(Type t) {
  return llvm::isa<ttk::TensorAccessorType>(t) ||
         llvm::isa<RankedTensorType>(t);
}

static CopySourceKind classifySrc(Value v) {
  if (llvm::isa<CircularBufferType>(v.getType())) {
    return CopySourceKind::CircularBuffer;
  }
  if (llvm::isa<TensorSliceType>(v.getType())) {
    return CopySourceKind::TensorSlice;
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
  if (llvm::isa<TensorSliceType>(v.getType())) {
    return CopyDestKind::TensorSlice;
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

/// Compute CTA index for a tensor function argument.
/// Reads ttl.base_cta_index and ttl.crta_indices from parent function.
/// Returns baseCTA + crtaIndices[localArgIdx].
static FailureOr<int32_t> computeCTAIndex(Value tensor, Operation *op) {
  auto argIdx = getTensorFuncArgIndex(tensor);
  if (failed(argIdx)) {
    return op->emitError("tensor must be a function argument");
  }

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

  if (*argIdx >= crtaIndicesAttr.size()) {
    return op->emitError("argument index out of range for ")
           << kCRTAIndicesAttr;
  }

  int64_t baseCTA = baseCTAAttr.getInt();
  int64_t globalTensorIdx =
      mlir::cast<IntegerAttr>(crtaIndicesAttr[*argIdx]).getInt();

  return static_cast<int32_t>(baseCTA + globalTensorIdx);
}

/// Create a TensorAccessor from a tensor type and bank base address.
/// The bankBase should come from runtime args via
/// getBufferAddressFromRuntimeArg.
///
/// This function derives page size from TTNNLayoutAttr encoding on the tensor.
/// Supported layouts:
///   - L1 interleaved (tiled)
///   - DRAM interleaved (tiled)
///
/// Unsupported layouts will emit errors referencing the appropriate GH issues:
///   - Sharded layouts: See GH issue #118
///   - Row-major (non-tiled): See GH issue #173
static FailureOr<Value>
materializeTensorAccessor(Value tensor, Value bankBase, Operation *op,
                          ConversionPatternRewriter &rewriter) {
  auto tensorTy = llvm::dyn_cast<RankedTensorType>(tensor.getType());
  if (!tensorTy) {
    return op->emitError("expected RankedTensorType for tensor accessor");
  }

  // Require TTNNLayoutAttr encoding - no fallback to contiguous layout.
  auto layoutAttr =
      mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(tensorTy.getEncoding());
  if (!layoutAttr) {
    return op->emitError(
        "tensor must have TTNNLayoutAttr encoding for accessor "
        "materialization; Python layer should reject tensors without TTNN "
        "layout");
  }

  // Reject sharded layouts - not yet supported (see GH issue #118).
  // Python error: "TTNN interop requires interleaved tensors"
  if (layoutAttr.hasShardedTensorMemoryLayout()) {
    return op->emitError("sharded memory layout not yet supported for tensor "
                         "accessor; see GH issue #118");
  }

  // Reject row-major (non-tiled) layouts - not yet supported (see GH #173).
  // Python error: "Only tiled CBs supported"
  if (!layoutAttr.isTiled()) {
    return op->emitError("row-major (non-tiled) layout not yet supported for "
                         "tensor accessor; see GH issue #173");
  }

  auto loc = tensor.getLoc();

  // Derive page size from the actual layout encoding.
  // For tiled interleaved layouts, page size = tile size in bytes.
  int64_t pageSizeBytes = layoutAttr.getElementSizeBytes();

  auto ctaIndex = computeCTAIndex(tensor, op);
  if (failed(ctaIndex)) {
    return failure();
  }

  auto argIdx = getTensorFuncArgIndex(tensor);
  if (failed(argIdx)) {
    return failure();
  }
  int32_t crtaIndex = static_cast<int32_t>(*argIdx);

  auto pageSize = rewriter.create<arith::ConstantIntOp>(loc, pageSizeBytes, 32);

  return buildTensorAccessor(loc, rewriter, *ctaIndex, crtaIndex, bankBase,
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

/// Extract tile grid shape from a Value if it's a static tensor.
/// Handles both rank-2 tensors (logical shape) and rank-4 tensors
/// (device shape: [grid_y, grid_x, shard_tiles_y, shard_tiles_x]).
static std::pair<int64_t, int64_t> getTileGridShapeFromValue(Value v) {
  auto tensorTy = llvm::dyn_cast<RankedTensorType>(v.getType());
  assert(tensorTy && "expected RankedTensorType");
  assert(tensorTy.hasStaticShape() && "expected static shape");

  auto dims = tensorTy.getShape();
  if (dims.size() == 2) {
    return getTileGridShape(tensorTy);
  } else if (dims.size() == 4) {
    // Rank-4 tensor: [grid_y, grid_x, shard_tiles_y, shard_tiles_x]
    // The last two dimensions are already tile counts, not element counts.
    return {dims[2], dims[3]};
  }

  llvm_unreachable("expected rank-2 or rank-4 tensor");
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

    scf::buildLoopNest(rewriter, loc, ValueRange{zero, zero},
                       ValueRange{yBound, xBound}, ValueRange{one, one},
                       [&](OpBuilder &b, Location bodyLoc, ValueRange ivs) {
                         // Compute linear tile offset: offset = iy * tilesX +
                         // ix
                         Value iy = ivs[0];
                         Value ix = ivs[1];
                         Value offsetY =
                             b.create<arith::MulIOp>(bodyLoc, iy, tilesXVal);
                         Value offset =
                             b.create<arith::AddIOp>(bodyLoc, offsetY, ix);

                         emitBody(b, bodyLoc, offset);
                       });
  } else {
    // Single tile: offset is always 0
    Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    emitBody(rewriter, loc, zeroIdx);
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
  // This derives page size from TTNNLayoutAttr encoding.
  auto srcAccessor =
      materializeTensorAccessor(srcTensor, *bankBase, op, rewriter);
  if (failed(srcAccessor)) {
    return failure(); // Error already emitted by materializeTensorAccessor
  }

  // Convert CB to TTKernel type and get write pointer.
  auto cbConverted = utils::convertTTLCBToTTKernel(dstCB, rewriter, loc);
  if (failed(cbConverted)) {
    return rewriter.notifyMatchFailure(op, "failed to convert CB operand");
  }
  auto cbWritePtr = rewriter.create<ttk::GetWritePtrOp>(loc, *cbConverted);

  auto [tilesY, tilesX] = getTileGridShapeFromValue(srcTensor);

  // Get page size for CB pointer arithmetic from TTNNLayoutAttr.
  auto tensorTy = mlir::cast<RankedTensorType>(srcTensor.getType());
  auto layoutAttr =
      mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(tensorTy.getEncoding());
  assert(layoutAttr &&
         "lowerTensorToCB: srcTensor must have TTNNLayoutAttr encoding");
  int64_t pageSizeBytes = layoutAttr.getElementSizeBytes();

  // Cast cbWritePtr to index for address arithmetic.
  auto indexTy = rewriter.getIndexType();
  auto cbWritePtrIdx =
      rewriter.create<arith::IndexCastOp>(loc, indexTy, cbWritePtr);

  // TODO(#138): Emit single block transfer for contiguous layouts instead of
  // tile loop.
  emitTileLoop(rewriter, loc, tilesY, tilesX,
               [&](OpBuilder &b, Location bodyLoc, Value tileOffset) {
                 // Compute CB address: cbWritePtr + tileOffset * pageSize
                 auto pageSizeIdx =
                     b.create<arith::ConstantIndexOp>(bodyLoc, pageSizeBytes);
                 Value byteOffset =
                     b.create<arith::MulIOp>(bodyLoc, tileOffset, pageSizeIdx);
                 Value cbAddrIdx = b.create<arith::AddIOp>(
                     bodyLoc, cbWritePtrIdx, byteOffset);
                 // Cast to i32 for NOC operation.
                 auto i32Ty = b.getI32Type();
                 Value tileOffset32 =
                     b.create<arith::IndexCastOp>(bodyLoc, i32Ty, tileOffset);
                 Value cbAddr =
                     b.create<arith::IndexCastOp>(bodyLoc, i32Ty, cbAddrIdx);
                 b.create<ttk::NocAsyncReadTileOp>(bodyLoc, tileOffset32,
                                                   *srcAccessor, cbAddr);
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
  // This derives page size from TTNNLayoutAttr encoding.
  auto dstAccessor =
      materializeTensorAccessor(dstTensor, *bankBase, op, rewriter);
  if (failed(dstAccessor)) {
    return failure(); // Error already emitted by materializeTensorAccessor
  }

  // Convert CB to TTKernel type and get read pointer.
  auto cbConverted = utils::convertTTLCBToTTKernel(srcCB, rewriter, loc);
  if (failed(cbConverted)) {
    return rewriter.notifyMatchFailure(op, "failed to convert CB operand");
  }
  auto cbReadPtr = rewriter.create<ttk::GetReadPtrOp>(loc, *cbConverted);

  auto [tilesY, tilesX] = getTileGridShapeFromValue(dstTensor);

  // Get page size for CB pointer arithmetic from TTNNLayoutAttr.
  auto tensorTy = mlir::cast<RankedTensorType>(dstTensor.getType());
  auto layoutAttr =
      mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(tensorTy.getEncoding());
  assert(layoutAttr &&
         "lowerCBToTensor: dstTensor must have TTNNLayoutAttr encoding");
  int64_t pageSizeBytes = layoutAttr.getElementSizeBytes();

  // Cast cbReadPtr to index for address arithmetic.
  auto indexTy = rewriter.getIndexType();
  auto cbReadPtrIdx =
      rewriter.create<arith::IndexCastOp>(loc, indexTy, cbReadPtr);

  // TODO(#138): Emit single block transfer for contiguous layouts instead of
  // tile loop.
  emitTileLoop(rewriter, loc, tilesY, tilesX,
               [&](OpBuilder &b, Location bodyLoc, Value tileOffset) {
                 // Compute CB address: cbReadPtr + tileOffset * pageSize
                 auto pageSizeIdx =
                     b.create<arith::ConstantIndexOp>(bodyLoc, pageSizeBytes);
                 Value byteOffset =
                     b.create<arith::MulIOp>(bodyLoc, tileOffset, pageSizeIdx);
                 Value cbAddrIdx =
                     b.create<arith::AddIOp>(bodyLoc, cbReadPtrIdx, byteOffset);
                 // Cast to i32 for NOC operation.
                 auto i32Ty = b.getI32Type();
                 Value tileOffset32 =
                     b.create<arith::IndexCastOp>(bodyLoc, i32Ty, tileOffset);
                 Value cbAddr =
                     b.create<arith::IndexCastOp>(bodyLoc, i32Ty, cbAddrIdx);
                 b.create<ttk::NocAsyncWriteTileOp>(bodyLoc, tileOffset32,
                                                    *dstAccessor, cbAddr);
               });

  auto handle = makeZeroI32(loc, rewriter);
  rewriter.replaceOp(op, handle);
  return success();
}

/// Compute linear tile offset from row and column indices.
/// offset = row * tilesX + col, converted to i32 for NOC ops.
static Value computeTileOffset(Value rowIdx, Value colIdx, int64_t tilesX,
                               Location loc,
                               ConversionPatternRewriter &rewriter) {
  auto tilesXVal = rewriter.create<arith::ConstantIndexOp>(loc, tilesX);
  Value offsetY = rewriter.create<arith::MulIOp>(loc, rowIdx, tilesXVal);
  Value offset = rewriter.create<arith::AddIOp>(loc, offsetY, colIdx);
  return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(),
                                             offset);
}

/// Lower tensor_slice->CB copy: read a single tile from tensor into CB.
static LogicalResult lowerSliceToCB(CopyOp op, TensorSliceOp sliceOp,
                                    Value dstCB,
                                    ConversionPatternRewriter &rewriter,
                                    const TypeConverter &typeConverter) {
  auto loc = op.getLoc();
  Value srcTensor = sliceOp.getTensor();
  Value tileRow = sliceOp.getTileRow();
  Value tileCol = sliceOp.getTileCol();

  auto bankBase = getBufferAddressFromRuntimeArg(srcTensor, loc, rewriter);
  if (failed(bankBase)) {
    return rewriter.notifyMatchFailure(
        op, "tensor must be a function argument for runtime arg mapping");
  }

  auto srcAccessor =
      materializeTensorAccessor(srcTensor, *bankBase, op, rewriter);
  if (failed(srcAccessor)) {
    return failure();
  }

  auto cbConverted = utils::convertTTLCBToTTKernel(dstCB, rewriter, loc);
  if (failed(cbConverted)) {
    return rewriter.notifyMatchFailure(op, "failed to convert CB operand");
  }
  auto cbWritePtr = rewriter.create<ttk::GetWritePtrOp>(loc, *cbConverted);

  auto [tilesY, tilesX] = getTileGridShapeFromValue(srcTensor);
  Value tileOffset = computeTileOffset(tileRow, tileCol, tilesX, loc, rewriter);

  rewriter.create<ttk::NocAsyncReadTileOp>(loc, tileOffset, *srcAccessor,
                                           cbWritePtr);

  auto handle = makeZeroI32(loc, rewriter);
  rewriter.replaceOp(op, handle);
  return success();
}

/// Lower CB->tensor_slice copy: write a single tile from CB to tensor.
static LogicalResult lowerCBToSlice(CopyOp op, Value srcCB,
                                    TensorSliceOp sliceOp,
                                    ConversionPatternRewriter &rewriter,
                                    const TypeConverter &typeConverter) {
  auto loc = op.getLoc();
  Value dstTensor = sliceOp.getTensor();
  Value tileRow = sliceOp.getTileRow();
  Value tileCol = sliceOp.getTileCol();

  auto bankBase = getBufferAddressFromRuntimeArg(dstTensor, loc, rewriter);
  if (failed(bankBase)) {
    return rewriter.notifyMatchFailure(
        op, "tensor must be a function argument for runtime arg mapping");
  }

  auto dstAccessor =
      materializeTensorAccessor(dstTensor, *bankBase, op, rewriter);
  if (failed(dstAccessor)) {
    return failure();
  }

  auto cbConverted = utils::convertTTLCBToTTKernel(srcCB, rewriter, loc);
  if (failed(cbConverted)) {
    return rewriter.notifyMatchFailure(op, "failed to convert CB operand");
  }
  auto cbReadPtr = rewriter.create<ttk::GetReadPtrOp>(loc, *cbConverted);

  auto [tilesY, tilesX] = getTileGridShapeFromValue(dstTensor);
  Value tileOffset = computeTileOffset(tileRow, tileCol, tilesX, loc, rewriter);

  rewriter.create<ttk::NocAsyncWriteTileOp>(loc, tileOffset, *dstAccessor,
                                            cbReadPtr);

  auto handle = makeZeroI32(loc, rewriter);
  rewriter.replaceOp(op, handle);
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

    // Use original operands for classification since lowering functions
    // handle type conversion internally.
    Value src = op.getSrc();
    Value dst = op.getDst();
    auto srcKind = classifySrc(src);
    auto dstKind = classifyDst(dst);

    // TensorSlice -> CB: read a single tile from tensor into circular buffer.
    if (srcKind == CopySourceKind::TensorSlice &&
        dstKind == CopyDestKind::CircularBuffer) {
      auto sliceOp = src.getDefiningOp<TensorSliceOp>();
      if (!sliceOp) {
        return rewriter.notifyMatchFailure(
            op, "tensor_slice source must come from ttl.tensor_slice op");
      }
      return lowerSliceToCB(op, sliceOp, adaptor.getDst(), rewriter,
                            *typeConverter);
    }

    // CB -> TensorSlice: write a single tile from circular buffer to tensor.
    if (srcKind == CopySourceKind::CircularBuffer &&
        dstKind == CopyDestKind::TensorSlice) {
      auto sliceOp = dst.getDefiningOp<TensorSliceOp>();
      if (!sliceOp) {
        return rewriter.notifyMatchFailure(
            op, "tensor_slice destination must come from ttl.tensor_slice op");
      }
      return lowerCBToSlice(op, adaptor.getSrc(), sliceOp, rewriter,
                            *typeConverter);
    }

    // Tensor -> CB: read all tiles from tensor into circular buffer (loop).
    if (srcKind == CopySourceKind::TensorAccessor &&
        dstKind == CopyDestKind::CircularBuffer) {
      return lowerTensorToCB(op, src, adaptor.getDst(), rewriter,
                             *typeConverter);
    }

    // CB -> Tensor: write all tiles from circular buffer to tensor (loop).
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

//===----------------------------------------------------------------------===//
// TTLConvertTTLToTTKernelPass helper methods
//===----------------------------------------------------------------------===//

// Forward declarations
static void removeTensorDataflowOps(func::FuncOp func);

/// Phase 1: Lower TTL ops (bind_cb, copy, wait, cb ops, store) to TTKernel.
static LogicalResult
lowerTTLOpsToTTKernel(ModuleOp mod, MLIRContext &ctx,
                      TTLToTTKernelTypeConverter &typeConverter,
                      StringRef passName) {
  ConversionTarget target(ctx);
  target.addIllegalDialect<tt::ttl::TTLDialect>();
  target.addLegalDialect<arith::ArithDialect, BuiltinDialect, scf::SCFDialect,
                         func::FuncDialect, tensor::TensorDialect,
                         ttkernel::TTKernelDialect>();

  // Structural ops remain legal (converted elsewhere or kept as-is).
  target.addLegalOp<ComputeOp, YieldOp, AttachCBOp>();

  // DST lifecycle ops are not tile compute ops; keep them legal until the
  // tile ops lowering phase.
  target.addLegalOp<InitSFPUOp, TileRegsAcquireOp, TileRegsCommitOp,
                    TileRegsWaitOp, TileRegsReleaseOp>();

  // CopyTileOp is a data movement op (CB -> DST), lowered in the tile ops
  // lowering phase.
  target.addLegalOp<CopyTileOp>();

  // Tile compute ops (identified by TTLTileComputeOpTrait) remain legal
  // until the tile ops lowering phase.
  target.addDynamicallyLegalDialect<tt::ttl::TTLDialect>(
      [](Operation *op) { return tt::ttl::isTileComputeOp(op); });

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

  RewritePatternSet patterns(&ctx);
  patterns.add<BindCBLowering, TensorSliceLowering, CopyLowering, WaitLowering,
               CBReserveLowering, CBPushLowering, CBWaitLowering, CBPopLowering,
               StoreLowering>(typeConverter, &ctx);
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
                       TTLToTTKernelTypeConverter &typeConverter) {
  ConversionTarget computeTarget(ctx);
  // TTKernel ops are legal (target dialect)
  computeTarget.addLegalDialect<ttkernel::TTKernelDialect>();
  // Arith ops are legal (used for index constants)
  computeTarget.addLegalDialect<arith::ArithDialect>();
  // Keep compute ops legal (tile-only lowering here).
  computeTarget.addLegalOp<ComputeOp, YieldOp>();

  // Other dialects are legal (func, tensor, etc.) EXCEPT tile ops.
  computeTarget.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

  // Mark TTL ops that need lowering as illegal (tile compute ops, CopyTileOp,
  // DST lifecycle). All other TTL ops (ComputeOp, YieldOp, AttachCBOp) were
  // explicitly marked legal above.
  computeTarget.addDynamicallyLegalDialect<tt::ttl::TTLDialect>(
      [](Operation *op) {
        // Tile compute ops (add, mul, exp, etc.) are illegal.
        if (tt::ttl::isTileComputeOp(op)) {
          return false;
        }
        // CopyTileOp (data movement) is illegal.
        if (isa<CopyTileOp>(op)) {
          return false;
        }
        // DST lifecycle ops are illegal.
        if (isa<InitSFPUOp, TileRegsAcquireOp, TileRegsCommitOp, TileRegsWaitOp,
                TileRegsReleaseOp>(op)) {
          return false;
        }
        // All other TTL ops are legal (ComputeOp, YieldOp, AttachCBOp).
        return true;
      });

  RewritePatternSet computePatterns(&ctx);
  populateTTLTileOpsToTTKernelPatterns(&typeConverter, computePatterns);
  if (failed(applyPartialConversion(mod, computeTarget,
                                    std::move(computePatterns)))) {
    return failure();
  }

  return success();
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
  if (failed(applyPatternsGreedily(mod, std::move(finalizePatterns)))) {
    return failure();
  }

  return success();
}

/// Phase 4: Clean up tensor dataflow ops in compute kernels.
/// Remove tensor dataflow ops that were used only for SSA tracking.
/// After loops are lowered and tile ops are converted, tensor.extract/insert/
/// empty are dead code. The actual computation happens through circular
/// buffers and DST registers.
static void cleanupComputeKernels(ModuleOp mod, MLIRContext &ctx) {
  mod.walk([&](func::FuncOp func) {
    // Check for compute kernel via either ttkernel.thread or
    // ttl.kernel_thread.
    auto threadAttr =
        func->getAttrOfType<ttk::ThreadTypeAttr>("ttkernel.thread");
    auto ttlThreadAttr =
        func->getAttrOfType<ttk::ThreadTypeAttr>("ttl.kernel_thread");

    bool isCompute = false;
    if (threadAttr && threadAttr.getValue() == ttk::ThreadType::Compute) {
      isCompute = true;
    } else if (ttlThreadAttr &&
               ttlThreadAttr.getValue() == ttk::ThreadType::Compute) {
      isCompute = true;
      // Convert ttl.kernel_thread to ttkernel.thread for compute kernels.
      func->removeAttr("ttl.kernel_thread");
      func->setAttr("ttkernel.thread", ttlThreadAttr);
    }

    if (!isCompute) {
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

    // Update return statements to return void if function has no results.
    // First check if there are any result uses.
    bool hasResultUses = false;
    func.walk([&](func::ReturnOp returnOp) {
      if (returnOp.getNumOperands() > 0) {
        // Check if the return value is actually used (it can't be for
        // func.return)
        hasResultUses = true;
      }
    });

    // For compute kernels, update function to return void.
    if (!func.getResultTypes().empty()) {
      func.walk([](func::ReturnOp returnOp) {
        if (returnOp.getNumOperands() > 0) {
          OpBuilder builder(returnOp);
          builder.create<func::ReturnOp>(returnOp.getLoc());
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

/// Helper: Remove dead tensor ops from a compute kernel function.
/// Tensor ops are removed in stages because each stage makes the next stage's
/// ops dead. This ensures use counts are updated correctly between stages.
static void removeTensorDataflowOps(func::FuncOp func) {

  // Stage 1: Replace tensor.insert results with dest tensor, then erase.
  // This makes tensor.extract results dead.
  SmallVector<tensor::InsertOp> insertOps;
  func.walk([&](tensor::InsertOp op) { insertOps.push_back(op); });
  for (auto op : insertOps) {
    op.getResult().replaceAllUsesWith(op.getDest());
    op.erase();
  }

  // Stage 2: Erase dead tensor.extract ops.
  // Must run after Stage 1 because replacing tensor.insert results makes
  // their corresponding extracts dead.
  SmallVector<tensor::ExtractOp> extractOps;
  func.walk([&](tensor::ExtractOp op) { extractOps.push_back(op); });
  for (auto op : extractOps) {
    if (op.getResult().use_empty()) {
      op.erase();
    }
  }

  // Stage 3: Erase dead tensor.empty ops.
  // Must run after Stage 2 because erasing extracts may make their source
  // tensor.empty ops dead.
  SmallVector<tensor::EmptyOp> emptyOps;
  func.walk([&](tensor::EmptyOp op) { emptyOps.push_back(op); });
  for (auto op : emptyOps) {
    if (op.getResult().use_empty()) {
      op.erase();
    }
  }

  // Simplify scf.for loops: remove unused iter_args and simplify yields.
  // After tensor dataflow removal, loops may have dead iter_args.
  func.walk([&](scf::ForOp forOp) {
    // Collect indices of iter_args that are still used outside the loop.
    SmallVector<unsigned> unusedArgIndices;
    for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
      if (forOp.getResult(i).use_empty()) {
        unusedArgIndices.push_back(i);
      }
    }

    // If all iter_args are unused, we can simplify but keep the loop
    // structure for the side effects (TTKernel ops).
    // The scf.yield will be updated in canonicalization.
  });
}

//===----------------------------------------------------------------------===//
// TTLConvertTTLToTTKernelPass
//===----------------------------------------------------------------------===//

struct TTLConvertTTLToTTKernelPass
    : impl::TTLConvertTTLToTTKernelBase<TTLConvertTTLToTTKernelPass> {
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    ModuleOp mod = getOperation();
    TTLToTTKernelTypeConverter typeConverter;

    // Phase 1: Lower TTL ops to TTKernel (bind_cb, copy, wait, cb ops, store)
    if (failed(lowerTTLOpsToTTKernel(mod, ctx, typeConverter, getName()))) {
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
