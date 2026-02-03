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

//===----------------------------------------------------------------------===//
// Pipe Graph: Tracks sender->receiver CB associations for pipe copies.
//
// For gather patterns, senders must write to the receiver's CB address, not
// their own. The PipeGraph identifies receiver CBs for each pipe and assigns
// runtime arg slots for passing receiver CB addresses to senders.
//===----------------------------------------------------------------------===//

/// Key for identifying a pipe by its source and destination coordinates.
struct PipeKey {
  int64_t srcX, srcY;
  int64_t dstStartX, dstStartY, dstEndX, dstEndY;

  bool operator==(const PipeKey &other) const {
    return srcX == other.srcX && srcY == other.srcY &&
           dstStartX == other.dstStartX && dstStartY == other.dstStartY &&
           dstEndX == other.dstEndX && dstEndY == other.dstEndY;
  }
};

struct PipeKeyHash {
  std::size_t operator()(const PipeKey &k) const {
    return llvm::hash_combine(k.srcX, k.srcY, k.dstStartX, k.dstStartY,
                              k.dstEndX, k.dstEndY);
  }
};

/// Receiver CB information for a pipe.
struct ReceiverCBInfo {
  int64_t cbIndex;       // CB index (0-31) used by receiver
  int64_t runtimeArgIdx; // Index in runtime args for receiver's CB address
};

/// Graph tracking pipe connections and receiver CB assignments.
/// Built before lowering by analyzing Pipe->CB copy operations.
class PipeGraph {
public:
  /// Analyze a module to find all pipe receivers and build the graph.
  static PipeGraph build(ModuleOp mod);

  /// Get receiver CB info for a pipe. Returns nullptr if not found.
  const ReceiverCBInfo *getReceiverInfo(int64_t srcX, int64_t srcY,
                                        int64_t dstStartX, int64_t dstStartY,
                                        int64_t dstEndX,
                                        int64_t dstEndY) const {
    PipeKey key{srcX, srcY, dstStartX, dstStartY, dstEndX, dstEndY};
    auto it = receiverCBs.find(key);
    if (it == receiverCBs.end()) {
      return nullptr;
    }
    return &it->second;
  }

  /// Get the number of runtime args needed for pipe receiver addresses.
  int64_t getNumPipeRuntimeArgs() const { return numPipeRuntimeArgs; }

  /// Check if any pipes were found.
  bool hasPipes() const { return !receiverCBs.empty(); }

  /// Add a receiver CB mapping.
  void addReceiverCB(int64_t srcX, int64_t srcY, int64_t dstStartX,
                     int64_t dstStartY, int64_t dstEndX, int64_t dstEndY,
                     int64_t cbIndex) {
    PipeKey key{srcX, srcY, dstStartX, dstStartY, dstEndX, dstEndY};
    receiverCBs[key] = {cbIndex, -1};
  }

  /// Assign runtime arg indices for all receiver CB addresses.
  void assignRuntimeArgIndices() {
    int64_t nextArgIdx = 0;
    for (auto &[key, info] : receiverCBs) {
      info.runtimeArgIdx = nextArgIdx++;
    }
    numPipeRuntimeArgs = nextArgIdx;
  }

  /// Emit pipe graph as JSON for Python to read and populate runtime args.
  /// Controlled by TTLANG_PIPE_GRAPH_JSON environment variable.
  void emitJSON() const {
    const char *path = std::getenv("TTLANG_PIPE_GRAPH_JSON");
    if (!path || receiverCBs.empty()) {
      return;
    }

    llvm::json::Object root;
    llvm::json::Array pipesArray;

    for (const auto &[key, info] : receiverCBs) {
      llvm::json::Object pipeObj;
      pipeObj["srcX"] = key.srcX;
      pipeObj["srcY"] = key.srcY;
      pipeObj["dstStartX"] = key.dstStartX;
      pipeObj["dstStartY"] = key.dstStartY;
      pipeObj["dstEndX"] = key.dstEndX;
      pipeObj["dstEndY"] = key.dstEndY;
      pipeObj["receiverCBIndex"] = info.cbIndex;
      pipeObj["runtimeArgSlot"] = info.runtimeArgIdx;
      pipesArray.push_back(std::move(pipeObj));
    }

    root["pipes"] = std::move(pipesArray);
    root["numPipeRuntimeArgs"] = numPipeRuntimeArgs;

    std::error_code ec;
    llvm::raw_fd_ostream os(path, ec);
    if (ec) {
      llvm::errs() << "Error writing pipe graph JSON to " << path << ": "
                   << ec.message() << "\n";
      return;
    }

    os << llvm::json::Value(std::move(root));
  }

private:
  std::unordered_map<PipeKey, ReceiverCBInfo, PipeKeyHash> receiverCBs;
  int64_t numPipeRuntimeArgs = 0;
};

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

    auto cb = getCBFromView(adaptor.getView());
    if (failed(cb)) {
      return rewriter.notifyMatchFailure(
          op, "view must come from ttl.cb_reserve (unrealized cast from CB)");
    }

    auto cbTileIndex =
        utils::computeCBTileIndexFromLoops(op, rewriter, /*cbShapeRank=*/2);

    // Determine DST index based on the source operation type:
    // - DST-to-DST ops (binary ops, copy_tile): have dst_idx attribute
    // - CB-reading ops (bcast, reduce): no dst_idx attribute, use loop index
    Value dstIndex;
    auto tileValue = adaptor.getTile();
    if (auto defOp = tileValue.getDefiningOp()) {
      if (auto dstIdxAttr =
              defOp->getAttrOfType<IntegerAttr>(kDstIdxAttrName)) {
        dstIndex =
            rewriter.create<arith::ConstantIndexOp>(loc, dstIdxAttr.getInt());
      } else if (auto copyTile = dyn_cast<CopyTileOp>(defOp)) {
        // Fallback: get dst_index directly from copy_tile operand
        dstIndex = copyTile.getDstIndex();
      } else {
        return op.emitError("ttl.store source op lacks dst_idx attribute: ")
               << defOp->getName();
      }
    } else {
      // Block argument (e.g., from bcast/reduce) - use CB tile index
      dstIndex = cbTileIndex;
    }

    rewriter.create<ttk::PackTileOp>(loc, dstIndex, *cb, cbTileIndex,
                                     /*out_of_order=*/false);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// PipeGraph implementation
//===----------------------------------------------------------------------===//

PipeGraph PipeGraph::build(ModuleOp mod) {
  PipeGraph graph;

  // Find all Pipe->CB copies (receiver side) and extract CB index
  mod.walk([&](CopyOp copyOp) {
    auto srcPipeType = dyn_cast<PipeType>(copyOp.getSrc().getType());
    if (!srcPipeType) {
      return;
    }

    // Found Pipe->CB copy: this is the receiver side
    Value dstCB = copyOp.getDst();
    if (!isa<CircularBufferType>(dstCB.getType())) {
      return;
    }

    // Trace to the BindCBOp to get the CB index
    Value cbVal = traceUnrealizedCasts(dstCB);
    auto bindOp = cbVal.getDefiningOp<BindCBOp>();
    if (!bindOp) {
      return;
    }

    int64_t cbIndex = bindOp.getCbIndex().getSExtValue();
    graph.addReceiverCB(srcPipeType.getSrcX(), srcPipeType.getSrcY(),
                        srcPipeType.getDstStartX(), srcPipeType.getDstStartY(),
                        srcPipeType.getDstEndX(), srcPipeType.getDstEndY(),
                        cbIndex);
  });

  graph.assignRuntimeArgIndices();
  return graph;
}

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

/// Extract tile grid shape from a Value if it's a static tensor.
/// Tensor shape must be [tiles_y, tiles_x] with TileType elements.
/// Returns the tile grid shape for linearization.
static std::pair<int64_t, int64_t> getTileGridShapeFromValue(Value v) {
  auto tensorTy = llvm::dyn_cast<RankedTensorType>(v.getType());
  assert(tensorTy && "expected RankedTensorType");
  assert(tensorTy.hasStaticShape() && "expected static shape");

  auto dims = tensorTy.getShape();
  assert(dims.size() == 2 && "expected rank-2 tensor [tiles_y, tiles_x]");
  assert(llvm::isa<ttcore::TileType>(tensorTy.getElementType()) &&
         "expected TileType element type");

  return {dims[0], dims[1]};
}

// Emit a tile loop (or single tile body). The callback receives (row, col)
// indices as index-typed Values.
static void emitTileLoop(
    OpBuilder &builder, Location loc, int64_t tilesY, int64_t tilesX,
    llvm::function_ref<void(OpBuilder &, Location, Value, Value)> emitBody) {
  auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  if (tilesY > 1 || tilesX > 1) {
    auto yBound = builder.create<arith::ConstantIndexOp>(loc, tilesY);
    auto xBound = builder.create<arith::ConstantIndexOp>(loc, tilesX);
    auto one = builder.create<arith::ConstantIndexOp>(loc, 1);

    scf::buildLoopNest(builder, loc, ValueRange{zero, zero},
                       ValueRange{yBound, xBound}, ValueRange{one, one},
                       [&](OpBuilder &b, Location bodyLoc, ValueRange ivs) {
                         emitBody(b, bodyLoc, ivs[0], ivs[1]);
                       });
  } else {
    emitBody(builder, loc, zero, zero);
  }
}

// Compute linear tile index from row/col: row * numCols + col.
static Value linearizeTileIndex(OpBuilder &builder, Location loc, Value row,
                                Value col, int64_t numCols) {
  auto numColsVal = builder.create<arith::ConstantIndexOp>(loc, numCols);
  Value rowOffset = builder.create<arith::MulIOp>(loc, row, numColsVal);
  return builder.create<arith::AddIOp>(loc, rowOffset, col);
}

/// Lower tensor_slice->CB copy: read tiles from tensor into CB.
/// Loops over CB shape, reading tiles starting at slice offset.
static LogicalResult lowerSliceToCB(CopyOp op, TensorSliceOp sliceOp,
                                    Value dstCB,
                                    ConversionPatternRewriter &rewriter,
                                    const TypeConverter &typeConverter) {
  auto loc = op.getLoc();
  Value srcTensor = sliceOp.getTensor();
  Value startRow = sliceOp.getTileRow();
  Value startCol = sliceOp.getTileCol();

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

  // Get CB shape for loop bounds.
  auto cbType = getTTLCBType(dstCB);
  if (!cbType) {
    return rewriter.notifyMatchFailure(op, "failed to get CB type");
  }
  auto cbShape = cbType.getShape();
  if (cbShape.size() != 2) {
    return rewriter.notifyMatchFailure(op, "CB shape must be 2D");
  }
  int64_t cbRows = cbShape[0];
  int64_t cbCols = cbShape[1];

  // Get tensor grid shape for computing tensor tile indices.
  auto tensorTileGridShape = getTileGridShapeFromValue(srcTensor);
  int64_t tensorTilesX = tensorTileGridShape.second;

  // Get page size for CB address arithmetic.
  auto tensorTy = mlir::cast<RankedTensorType>(srcTensor.getType());
  auto layoutAttr =
      mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(tensorTy.getEncoding());
  if (!layoutAttr) {
    return rewriter.notifyMatchFailure(
        op, "tensor must have TTNNLayoutAttr encoding");
  }
  int64_t pageSizeBytes = layoutAttr.getElementSizeBytes();

  auto indexTy = rewriter.getIndexType();
  auto cbWritePtrIdx =
      rewriter.create<arith::IndexCastOp>(loc, indexTy, cbWritePtr);
  auto pageSizeIdx =
      rewriter.create<arith::ConstantIndexOp>(loc, pageSizeBytes);
  auto i32Ty = rewriter.getI32Type();

  emitTileLoop(
      rewriter, loc, cbRows, cbCols,
      [&, tensorTilesX, cbCols](OpBuilder &b, Location bodyLoc, Value loopRow,
                                Value loopCol) {
        // Tensor tile index: (startRow + loopRow) * tensorCols + (startCol +
        // loopCol)
        Value tensorRow = b.create<arith::AddIOp>(bodyLoc, startRow, loopRow);
        Value tensorCol = b.create<arith::AddIOp>(bodyLoc, startCol, loopCol);
        Value tensorTileIdx =
            linearizeTileIndex(b, bodyLoc, tensorRow, tensorCol, tensorTilesX);

        // CB tile index within the CB buffer.
        Value cbTileIdx =
            linearizeTileIndex(b, bodyLoc, loopRow, loopCol, cbCols);

        // Compute CB address: cbWritePtr + cbTileIdx * pageSize
        Value byteOffset =
            b.create<arith::MulIOp>(bodyLoc, cbTileIdx, pageSizeIdx);
        Value cbAddrIdx =
            b.create<arith::AddIOp>(bodyLoc, cbWritePtrIdx, byteOffset);

        // Cast to i32 for NOC operation.
        Value tensorTileIdx32 =
            b.create<arith::IndexCastOp>(bodyLoc, i32Ty, tensorTileIdx);
        Value cbAddr = b.create<arith::IndexCastOp>(bodyLoc, i32Ty, cbAddrIdx);

        b.create<ttk::NocAsyncReadTileOp>(bodyLoc, tensorTileIdx32,
                                          *srcAccessor, cbAddr);
      });

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

/// Compute semaphore index for a pipe based on source coordinates.
/// For point-to-point pipes (gather pattern), each source uses a distinct
/// semaphore to avoid race conditions when multiple sources signal the same
/// destination. For multicast pipes, all destinations share semaphore 0.
static int64_t getPipeSemaphoreIndex(PipeType pipeType) {
  if (pipeType.isUnicast()) {
    // Point-to-point: use source X coordinate as semaphore index.
    // This ensures each source->dest pipe has a unique semaphore.
    // Note: For 2D grids, this should be srcX * gridHeight + srcY.
    return pipeType.getSrcX();
  }
  // Multicast: all destinations use semaphore 0.
  return 0;
}

/// Lower CB -> Pipe copy: multicast tiles from source CB to destination cores.
/// For gather patterns, uses receiver's CB address from PipeGraph.
/// After multicast, signals destinations via semaphore.
///
/// Parameters:
/// - receiverInfo: If non-null, contains the receiver's CB index and runtime
///   arg index for the gather pattern. The receiver's CB address is loaded from
///   runtime args to ensure data lands at the correct L1 address on the
///   destination core (which may differ from the sender's CB address).
static LogicalResult lowerCBToPipe(CopyOp op, Value srcCB, Value pipe,
                                   const ReceiverCBInfo *receiverInfo,
                                   ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto pipeType = llvm::cast<PipeType>(pipe.getType());

  auto cbConverted = utils::convertTTLCBToTTKernel(srcCB, rewriter, loc);
  if (failed(cbConverted)) {
    return rewriter.notifyMatchFailure(op, "failed to convert CB operand");
  }

  // Get CB info for address computation
  auto cbType = getTTLCBType(srcCB);
  if (!cbType) {
    return rewriter.notifyMatchFailure(op, "failed to get CB type");
  }
  auto cbShape = cbType.getShape();
  if (cbShape.size() != 2) {
    return rewriter.notifyMatchFailure(op, "CB shape must be 2D");
  }
  int64_t cbRows = cbShape[0];
  int64_t cbCols = cbShape[1];

  // Get page size from element type
  auto elementType = cbType.getElementType();
  auto tileType = llvm::dyn_cast<ttcore::TileType>(elementType);
  if (!tileType) {
    return rewriter.notifyMatchFailure(op, "CB element type must be tile");
  }
  int64_t pageSizeBytes = tileType.getSizeBytes();

  // Get destination range from pipe
  int64_t dstStartX = pipeType.getDstStartX();
  int64_t dstStartY = pipeType.getDstStartY();
  int64_t dstEndX = pipeType.getDstEndX();
  int64_t dstEndY = pipeType.getDstEndY();
  int64_t numDests = pipeType.getNumDests();

  auto indexTy = rewriter.getIndexType();
  auto i32Ty = rewriter.getI32Type();

  // Get CB read pointer (source L1 address for reading local data).
  // For gather patterns, the destination CB address on the receiver is assumed
  // to match the sender's CB address (uniform CB allocation across cores).
  // The receiverInfo is used for runtime arg passing (debugging/future use).
  (void)receiverInfo; // Used for runtime args, not address computation
  auto cbReadPtr = rewriter.create<ttk::GetReadPtrOp>(loc, *cbConverted);
  auto cbReadPtrIdx =
      rewriter.create<arith::IndexCastOp>(loc, indexTy, cbReadPtr);
  auto pageSizeIdx =
      rewriter.create<arith::ConstantIndexOp>(loc, pageSizeBytes);

  // Destination coordinates for multicast - convert logical to virtual coords
  auto dstStartXLogical =
      rewriter.create<arith::ConstantIndexOp>(loc, dstStartX);
  auto dstStartYLogical =
      rewriter.create<arith::ConstantIndexOp>(loc, dstStartY);
  auto dstEndXLogical = rewriter.create<arith::ConstantIndexOp>(loc, dstEndX);
  auto dstEndYLogical = rewriter.create<arith::ConstantIndexOp>(loc, dstEndY);

  // NOC operations require virtual/translated coordinates
  auto dstStartXVal = rewriter.create<ttk::ConvertLogicalXToTranslatedOp>(
      loc, indexTy, dstStartXLogical);
  auto dstStartYVal = rewriter.create<ttk::ConvertLogicalYToTranslatedOp>(
      loc, indexTy, dstStartYLogical);
  auto dstEndXVal = rewriter.create<ttk::ConvertLogicalXToTranslatedOp>(
      loc, indexTy, dstEndXLogical);
  auto dstEndYVal = rewriter.create<ttk::ConvertLogicalYToTranslatedOp>(
      loc, indexTy, dstEndYLogical);

  auto numDestsVal = rewriter.create<arith::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(numDests));
  auto pageSizeVal = rewriter.create<arith::ConstantOp>(
      loc, i32Ty, rewriter.getI32IntegerAttr(pageSizeBytes));

  // For unicast gather pipes (srcX > dstX), each source writes to a different
  // slot in the destination CB to avoid overwrites when multiple sources send
  // to the same destination. Slot index = srcX - dstX - 1 (0-based).
  // For forward pipes (srcX < dstX), slot offset is 0 since there's one source.
  int64_t slotIdx = 0;
  if (pipeType.isUnicast()) {
    int64_t srcX = pipeType.getSrcX();
    int64_t dstX = pipeType.getDstStartX();
    if (srcX > dstX) {
      // Gather pattern: offset by source position relative to destination
      slotIdx = srcX - dstX - 1;
    }
    // Forward pattern (srcX <= dstX): slotIdx stays 0
  }
  int64_t slotByteOffset = slotIdx * pageSizeBytes * cbRows * cbCols;

  emitTileLoop(
      rewriter, loc, cbRows, cbCols,
      [&](OpBuilder &b, Location bodyLoc, Value loopRow, Value loopCol) {
        // Compute CB tile index and address
        Value cbTileIdx =
            linearizeTileIndex(b, bodyLoc, loopRow, loopCol, cbCols);
        Value byteOffset =
            b.create<arith::MulIOp>(bodyLoc, cbTileIdx, pageSizeIdx);
        Value srcAddrIdx =
            b.create<arith::AddIOp>(bodyLoc, cbReadPtrIdx, byteOffset);
        Value srcAddr =
            b.create<arith::IndexCastOp>(bodyLoc, i32Ty, srcAddrIdx);

        // Compute destination address (same base as source, uniform CB layout).
        // Add slot offset for gather patterns (multiple sources to one dest).
        Value dstAddrIdx =
            b.create<arith::AddIOp>(bodyLoc, cbReadPtrIdx, byteOffset);
        if (slotByteOffset > 0) {
          auto slotOffsetIdx =
              b.create<arith::ConstantIndexOp>(bodyLoc, slotByteOffset);
          dstAddrIdx =
              b.create<arith::AddIOp>(bodyLoc, dstAddrIdx, slotOffsetIdx);
        }
        Value dstAddr =
            b.create<arith::IndexCastOp>(bodyLoc, i32Ty, dstAddrIdx);

        // Get multicast NOC address using destination address
        auto mcastAddr = b.create<ttk::GetNocMulticastAddrOp>(
            bodyLoc, dstStartXVal, dstStartYVal, dstEndXVal, dstEndYVal,
            dstAddr, /*noc=*/Value());

        // Perform multicast write
        // Optional attrs: linked=nullptr, multicast_path_reserve=nullptr,
        // noc=nullptr
        if (pipeType.srcInDstRange()) {
          // Source is in destination range - use loopback version
          b.create<ttk::NocAsyncWriteMulticastLoopbackSrcOp>(
              bodyLoc, srcAddr, mcastAddr.getResult(), pageSizeVal, numDestsVal,
              /*linked=*/nullptr, /*multicast_path_reserve=*/nullptr,
              /*noc=*/Value());
        } else {
          // Source not in destination range - normal multicast
          b.create<ttk::NocAsyncWriteMulticastOp>(
              bodyLoc, srcAddr, mcastAddr.getResult(), pageSizeVal, numDestsVal,
              /*linked=*/nullptr, /*multicast_path_reserve=*/nullptr,
              /*noc=*/Value());
        }
      });

  // Wait for all async writes to complete before signaling the semaphore.
  // Without this barrier, the receiver may wake up before all data arrives.
  rewriter.create<ttk::NocAsyncWriteBarrierOp>(loc);

  // Signal destinations that data has arrived.
  // For point-to-point pipes, use atomic increment to support gather pattern
  // (multiple sources to one destination). For multicast, use set+multicast.
  int64_t semIdxVal = getPipeSemaphoreIndex(pipeType);
  auto semIdx = rewriter.create<arith::ConstantIndexOp>(loc, semIdxVal);
  auto semAddr = rewriter.create<ttk::GetSemaphoreOp>(loc, semIdx);

  if (pipeType.isUnicast()) {
    // Point-to-point: atomically increment destination's semaphore.
    // This supports gather patterns where multiple sources signal one dest.
    auto incrVal = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Get NOC address of destination's semaphore for atomic increment.
    auto dstSemNocAddr = rewriter.create<ttk::GetNocAddrOp>(
        loc, dstStartXVal, dstStartYVal, semAddr);

    rewriter.create<ttk::NocSemaphoreIncOp>(loc, dstSemNocAddr.getResult(),
                                            incrVal, /*noc_id=*/Value());
  } else {
    // Multicast: set local semaphore and multicast to all destinations.
    auto semPtr = rewriter.create<ttk::CastToL1PtrOp>(loc, semAddr);
    auto validVal = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    rewriter.create<ttk::NocSemaphoreSetOp>(loc, semPtr, validVal);

    auto semMcastAddr = rewriter.create<ttk::GetNocMulticastAddrOp>(
        loc, dstStartXVal, dstStartYVal, dstEndXVal, dstEndYVal, semAddr,
        /*noc=*/Value());

    if (pipeType.srcInDstRange()) {
      auto falseBoolAttr = rewriter.getBoolAttr(false);
      rewriter.create<ttk::NocSemaphoreSetMulticastLoopbackOp>(
          loc, semAddr, semMcastAddr.getResult(), numDestsVal,
          /*linked=*/falseBoolAttr, /*multicast_path_reserve=*/falseBoolAttr);
    } else {
      rewriter.create<ttk::NocSemaphoreSetMulticastOp>(
          loc, semAddr, semMcastAddr.getResult(), numDestsVal,
          /*linked=*/nullptr, /*multicast_path_reserve=*/nullptr);
    }
  }

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

/// Lower Pipe -> CB copy: destination side of pipe transfer.
/// At the destination, data arrives via multicast/unicast from source core.
/// Waits for semaphore signal from source before proceeding.
static LogicalResult lowerPipeToCB(CopyOp op, Value pipe, Value dstCB,
                                   ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto pipeType = llvm::cast<PipeType>(pipe.getType());

  // Use semaphore index derived from pipe's source coordinates.
  // This ensures each source->dest pipe has a unique semaphore.
  int64_t semIdxVal = getPipeSemaphoreIndex(pipeType);
  auto semIdx = rewriter.create<arith::ConstantIndexOp>(loc, semIdxVal);
  auto semAddr = rewriter.create<ttk::GetSemaphoreOp>(loc, semIdx);
  auto semPtr = rewriter.create<ttk::CastToL1PtrOp>(loc, semAddr);

  // Wait for semaphore to reach at least 1.
  // For point-to-point, source uses atomic inc; for multicast, source uses set.
  auto oneVal = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI32Type(), rewriter.getI32IntegerAttr(1));
  rewriter.create<ttk::NocSemaphoreWaitMinOp>(loc, semPtr, oneVal);

  // Reset semaphore for next use
  auto zeroVal = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  rewriter.create<ttk::NocSemaphoreSetOp>(loc, semPtr, zeroVal);

  rewriter.replaceOp(op, makeZeroI32(loc, rewriter));
  return success();
}

/// Lower CB->tensor_slice copy: write tiles from CB to tensor.
/// Loops over CB shape, writing tiles starting at slice offset.
static LogicalResult lowerCBToSlice(CopyOp op, Value srcCB,
                                    TensorSliceOp sliceOp,
                                    ConversionPatternRewriter &rewriter,
                                    const TypeConverter &typeConverter) {
  auto loc = op.getLoc();
  Value dstTensor = sliceOp.getTensor();
  Value startRow = sliceOp.getTileRow();
  Value startCol = sliceOp.getTileCol();

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

  // Get CB shape for loop bounds.
  auto cbType = getTTLCBType(srcCB);
  if (!cbType) {
    return rewriter.notifyMatchFailure(op, "failed to get CB type");
  }
  auto cbShape = cbType.getShape();
  if (cbShape.size() != 2) {
    return rewriter.notifyMatchFailure(op, "CB shape must be 2D");
  }
  int64_t cbRows = cbShape[0];
  int64_t cbCols = cbShape[1];

  // Get tensor grid shape for computing tensor tile indices.
  auto tensorTileGridShape = getTileGridShapeFromValue(dstTensor);
  int64_t tensorTilesX = tensorTileGridShape.second;

  // Get page size for CB address arithmetic.
  auto tensorTy = mlir::cast<RankedTensorType>(dstTensor.getType());
  auto layoutAttr =
      mlir::dyn_cast_or_null<ttnn::TTNNLayoutAttr>(tensorTy.getEncoding());
  if (!layoutAttr) {
    return rewriter.notifyMatchFailure(
        op, "tensor must have TTNNLayoutAttr encoding");
  }
  int64_t pageSizeBytes = layoutAttr.getElementSizeBytes();

  auto indexTy = rewriter.getIndexType();
  auto cbReadPtrIdx =
      rewriter.create<arith::IndexCastOp>(loc, indexTy, cbReadPtr);
  auto pageSizeIdx =
      rewriter.create<arith::ConstantIndexOp>(loc, pageSizeBytes);
  auto i32Ty = rewriter.getI32Type();

  emitTileLoop(
      rewriter, loc, cbRows, cbCols,
      [&, tensorTilesX, cbCols](OpBuilder &b, Location bodyLoc, Value loopRow,
                                Value loopCol) {
        // Tensor tile index: (startRow + loopRow) * tensorCols + (startCol +
        // loopCol)
        Value tensorRow = b.create<arith::AddIOp>(bodyLoc, startRow, loopRow);
        Value tensorCol = b.create<arith::AddIOp>(bodyLoc, startCol, loopCol);
        Value tensorTileIdx =
            linearizeTileIndex(b, bodyLoc, tensorRow, tensorCol, tensorTilesX);

        // CB tile index within the CB buffer.
        Value cbTileIdx =
            linearizeTileIndex(b, bodyLoc, loopRow, loopCol, cbCols);

        // Compute CB address: cbReadPtr + cbTileIdx * pageSize
        Value byteOffset =
            b.create<arith::MulIOp>(bodyLoc, cbTileIdx, pageSizeIdx);
        Value cbAddrIdx =
            b.create<arith::AddIOp>(bodyLoc, cbReadPtrIdx, byteOffset);

        // Cast to i32 for NOC operation.
        Value tensorTileIdx32 =
            b.create<arith::IndexCastOp>(bodyLoc, i32Ty, tensorTileIdx);
        Value cbAddr = b.create<arith::IndexCastOp>(bodyLoc, i32Ty, cbAddrIdx);

        b.create<ttk::NocAsyncWriteTileOp>(bodyLoc, tensorTileIdx32,
                                           *dstAccessor, cbAddr);
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
            pipeType.getDstEndY());
      }
      return lowerCBToPipe(op, adaptor.getSrc(), adaptor.getDst(), receiverInfo,
                           rewriter);
    }
    if (srcIsPipe && dstIsCB) {
      // Pipe -> CB: destination receives data via multicast from source
      return lowerPipeToCB(op, adaptor.getSrc(), adaptor.getDst(), rewriter);
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
      return lowerSliceToCB(op, sliceOp, adaptor.getDst(), rewriter,
                            *typeConverter);
    }

    // CB -> TensorSlice: write tiles from circular buffer to tensor.
    auto sliceOp = dst.getDefiningOp<TensorSliceOp>();
    if (!sliceOp) {
      return rewriter.notifyMatchFailure(
          op, "tensor_slice destination must come from ttl.tensor_slice op");
    }
    return lowerCBToSlice(op, adaptor.getSrc(), sliceOp, rewriter,
                          *typeConverter);
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
// Pipe conditional operation lowering patterns
//===----------------------------------------------------------------------===//

struct IfSrcLowering : OpConversionPattern<IfSrcOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfSrcOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto pipeType = mlir::cast<PipeType>(op.getPipe().getType());

    // Get current core coordinates.
    auto coreX =
        rewriter.create<ttk::MyLogicalXOp>(loc, rewriter.getIndexType());
    auto coreY =
        rewriter.create<ttk::MyLogicalYOp>(loc, rewriter.getIndexType());

    // Get source coordinates from pipe type.
    auto srcXConst =
        rewriter.create<arith::ConstantIndexOp>(loc, pipeType.getSrcX());
    auto srcYConst =
        rewriter.create<arith::ConstantIndexOp>(loc, pipeType.getSrcY());

    // Check if current core matches source coordinates.
    auto matchX = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 coreX, srcXConst);
    auto matchY = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 coreY, srcYConst);
    auto isSrc = rewriter.create<arith::AndIOp>(loc, matchX, matchY);

    // Create scf.if with empty body (the builder adds a yield for us).
    auto ifOp =
        rewriter.create<scf::IfOp>(loc, isSrc, /*withElseRegion=*/false);

    // Move ops from the original body into the then block (before the yield).
    // Using inlineBlockBefore moves rather than clones, preserving SSA.
    Block &srcBlock = op.getBody().front();
    Block &thenBlock = ifOp.getThenRegion().front();
    rewriter.inlineBlockBefore(&srcBlock, thenBlock.getTerminator());

    rewriter.eraseOp(op);
    return success();
  }
};

struct IfDstLowering : OpConversionPattern<IfDstOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(IfDstOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto pipeType = mlir::cast<PipeType>(op.getPipe().getType());

    // Get current core coordinates.
    auto coreX =
        rewriter.create<ttk::MyLogicalXOp>(loc, rewriter.getIndexType());
    auto coreY =
        rewriter.create<ttk::MyLogicalYOp>(loc, rewriter.getIndexType());

    // Get destination range from pipe type.
    int64_t dstMinX = std::min(pipeType.getDstStartX(), pipeType.getDstEndX());
    int64_t dstMaxX = std::max(pipeType.getDstStartX(), pipeType.getDstEndX());
    int64_t dstMinY = std::min(pipeType.getDstStartY(), pipeType.getDstEndY());
    int64_t dstMaxY = std::max(pipeType.getDstStartY(), pipeType.getDstEndY());

    auto minXConst = rewriter.create<arith::ConstantIndexOp>(loc, dstMinX);
    auto maxXConst = rewriter.create<arith::ConstantIndexOp>(loc, dstMaxX);
    auto minYConst = rewriter.create<arith::ConstantIndexOp>(loc, dstMinY);
    auto maxYConst = rewriter.create<arith::ConstantIndexOp>(loc, dstMaxY);

    // Check if current core is within destination range.
    // coreX >= minX && coreX <= maxX && coreY >= minY && coreY <= maxY
    auto geMinX = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                                 coreX, minXConst);
    auto leMaxX = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle,
                                                 coreX, maxXConst);
    auto geMinY = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                                 coreY, minYConst);
    auto leMaxY = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sle,
                                                 coreY, maxYConst);

    auto inRangeX = rewriter.create<arith::AndIOp>(loc, geMinX, leMaxX);
    auto inRangeY = rewriter.create<arith::AndIOp>(loc, geMinY, leMaxY);
    auto isDst = rewriter.create<arith::AndIOp>(loc, inRangeX, inRangeY);

    // Create scf.if with empty body (the builder adds a yield for us).
    auto ifOp =
        rewriter.create<scf::IfOp>(loc, isDst, /*withElseRegion=*/false);

    // Move ops from the original body into the then block (before the yield).
    // Using inlineBlockBefore moves rather than clones, preserving SSA.
    Block &srcBlock = op.getBody().front();
    Block &thenBlock = ifOp.getThenRegion().front();
    rewriter.inlineBlockBefore(&srcBlock, thenBlock.getTerminator());

    rewriter.eraseOp(op);
    return success();
  }
};

struct CreatePipeLowering : OpConversionPattern<CreatePipeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CreatePipeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // CreatePipeOp is a Pure op that just produces a pipe type value.
    // The pipe type carries all the coordinate information as type parameters.
    // At runtime, pipes don't need any materialization - the coordinates are
    // baked into the generated code through if_src/if_dst lowering.
    //
    // Always replace with an unrealized cast to handle uses in nested regions
    // (like if_src/if_dst bodies) that may be processed in a different order.
    // The unrealized cast preserves the type for downstream patterns.
    auto cast = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), op.getResult().getType(), ValueRange{});
    rewriter.replaceOp(op, cast.getResult(0));
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

/// Lowering for tensor_store: handles cleanup after elementwise lowering.
/// For elementwise ops, the ComputeOp already writes to the output CB, so
/// tensor_store becomes a no-op. For passthrough (CB-attached input), we
/// would need to emit copy_tile + pack_tile, but that case should be handled
/// by LowerTensorStoreToCompute creating a passthrough ComputeOp.
struct TensorStoreLowering : OpConversionPattern<TensorStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TensorStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = op.getTensor();

    // If input is CB-attached, this is a passthrough case that should have
    // been handled by LowerTensorStoreToCompute. Emit error.
    if (getAttachedCB(input)) {
      return op.emitError(
          "passthrough tensor_store should be lowered to ComputeOp first; "
          "ensure convert-ttl-to-compute runs before this pass");
    }

    // For elementwise case: the ComputeOp already wrote to the output CB.
    // tensor_store is now a no-op - just erase it.
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

  // SignpostOp is lowered in a separate pass (ttl-lower-signpost-to-emitc).
  target.addLegalOp<SignpostOp>();

  // CopyTileOp is a data movement op (CB -> DST), lowered in the tile ops
  // lowering phase.
  target.addLegalOp<CopyTileOp>();

  // Tile compute ops (identified by TTLTileComputeOpTrait) remain legal
  // until the tile ops lowering phase.
  target.addDynamicallyLegalDialect<tt::ttl::TTLDialect>([](Operation *op) {
    // Tile compute ops stay legal until tile ops lowering phase.
    return tt::ttl::isTileComputeOp(op);
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
  PipeGraph pipeGraph = PipeGraph::build(mod);

  // Emit pipe graph JSON for Python to read (controlled by env var).
  pipeGraph.emitJSON();

  RewritePatternSet patterns(&ctx);
  patterns.add<CopyLowering>(typeConverter, &ctx, &pipeGraph);
  patterns
      .add<BindCBLowering, TensorSliceLowering, WaitLowering,
           CBReserveLowering, CBPushLowering, CBWaitLowering, CBPopLowering,
           StoreLowering, CoreXLowering, CoreYLowering, TensorStoreLowering,
           IfSrcLowering, IfDstLowering, CreatePipeLowering>(
          typeConverter, &ctx);
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
