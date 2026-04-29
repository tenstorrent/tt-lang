// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"

namespace mlir::tt::ttl {
#define GEN_PASS_DEF_TTLLOWERDPRINTTOEMITC
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

template <typename BuilderT>
static void emitVerbatim(Location loc, StringRef value, BuilderT &builder) {
  OperationState state(loc, "emitc.verbatim");
  state.addAttribute("value", builder.getStringAttr(value));
  builder.create(state);
}

/// Resolve the CB index from a CB value. Handles:
///   - Direct BindCBOp result
///   - Block argument of ComputeOp (via ttl.cb_index.N annotation)
static FailureOr<int64_t> resolveCBIndex(Value cbValue, Operation *dprintOp) {
  cbValue = traceUnrealizedCasts(cbValue);

  if (auto bindOp = cbValue.getDefiningOp<BindCBOp>()) {
    return bindOp.getCbIndex().getSExtValue();
  }

  if (auto blockArg = dyn_cast<BlockArgument>(cbValue)) {
    auto *parentOp = blockArg.getOwner()->getParentOp();
    if (auto computeOp = dyn_cast<ComputeOp>(parentOp)) {
      unsigned argIdx = blockArg.getArgNumber();
      auto cbIdx = getCBIndexAttr(computeOp, argIdx);
      if (cbIdx) {
        return *cbIdx;
      }
      return dprintOp->emitError("CB index annotation missing for compute "
                                 "input ")
             << argIdx;
    }
  }

  return dprintOp->emitError(
      "cannot resolve CB index: value must trace to ttl.bind_cb "
      "or be a compute block argument with CB annotation");
}

/// Get the function arg index for a tensor value, if it is a direct
/// function argument (tensor accessor).
static std::optional<unsigned> getTensorFuncArgIndex(Value tensor) {
  tensor = traceUnrealizedCasts(tensor);
  auto blockArg = dyn_cast<BlockArgument>(tensor);
  if (!blockArg || !blockArg.getParentBlock() ||
      !blockArg.getParentBlock()->isEntryBlock()) {
    return std::nullopt;
  }
  return blockArg.getArgNumber();
}

/// Resolve a scalar variable to its C++ expression string.
static FailureOr<std::string> resolveScalarExpr(Value val, Operation *op) {
  if (auto constOp = val.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue())) {
      return std::to_string(intAttr.getInt());
    }
    if (auto floatAttr = dyn_cast<FloatAttr>(constOp.getValue())) {
      // DPRINT supports float but not double; emit float literal.
      return std::to_string(floatAttr.getValueAsDouble()) + "f";
    }
    return op->emitError("unsupported arith.constant type in dprint");
  }

  if (auto coreX = val.getDefiningOp<CoreXOp>()) {
    return std::string("get_absolute_logical_x()");
  }
  if (auto coreY = val.getDefiningOp<CoreYOp>()) {
    return std::string("get_absolute_logical_y()");
  }

  return op->emitError("unsupported variable in scalar dprint: only "
                       "arith constants and core coordinates are supported");
}

/// Build the DPRINT streaming statement for scalar mode.
/// Format string uses {} as placeholder for each argv operand.
static FailureOr<std::string> buildScalarDPrintStmt(DPrintOp op) {
  std::string stmt = "DPRINT << ";
  StringRef fmt = op.getFmt();
  auto argv = op.getArgv();
  unsigned argIdx = 0;

  size_t pos = 0;
  while (pos < fmt.size()) {
    size_t placeholderPos = fmt.find("{}", pos);

    // Emit the string literal before the placeholder (or remaining string).
    StringRef strPart = fmt.slice(
        pos, placeholderPos == StringRef::npos ? fmt.size() : placeholderPos);
    if (!strPart.empty()) {
      stmt += "\"";
      // Escape quotes and backslashes in the string literal.
      for (char c : strPart) {
        if (c == '"' || c == '\\') {
          stmt += '\\';
        }
        stmt += c;
      }
      stmt += "\" << ";
    }

    if (placeholderPos == StringRef::npos) {
      break;
    }

    // Emit the variable expression for this placeholder.
    if (argIdx >= argv.size()) {
      return op.emitError("format string has more {} placeholders than argv "
                          "operands");
    }
    auto expr = resolveScalarExpr(argv[argIdx], op);
    if (failed(expr)) {
      return failure();
    }
    stmt += *expr + " << ";
    argIdx++;
    pos = placeholderPos + 2; // skip "{}"
  }

  stmt += "ENDL()";
  return stmt;
}

/// Resolve tensor element type info for page printing.
/// Returns {dprint_formatter, c_ptr_type, elements_per_page, page_size_bytes}.
struct TensorPrintInfo {
  std::string formatter; // e.g. "BF16" or "F32"
  std::string cPtrType;  // e.g. "uint16_t" or "uint32_t"
  int64_t eltsPerPage;
  int64_t pageSizeBytes;
};

static FailureOr<TensorPrintInfo> getTensorPrintInfo(Type elementType,
                                                     Operation *op) {
  auto tileType = dyn_cast<ttcore::TileType>(elementType);
  if (!tileType) {
    return op->emitError("tensor element type is not a tile type");
  }

  int64_t tileH = tileType.getHeight();
  int64_t tileW = tileType.getWidth();
  Type dataType = tileType.getElementType();

  if (dataType.isBF16()) {
    return TensorPrintInfo{"BF16", "uint16_t", tileH * tileW,
                           tileH * tileW * 2};
  }
  if (dataType.isF32()) {
    return TensorPrintInfo{"F32", "uint32_t", tileH * tileW, tileH * tileW * 4};
  }
  return op->emitError("unsupported data type for tensor print: only bf16 "
                       "and f32 are supported");
}

struct DSTSlotInfo {
  int64_t slot;
  std::string opName;
  bool isFloat32;
};

/// Find all live DST register slots at a given program point.
///
/// Walks all ops in the parent function that precede the dprint op in
/// program order, including ops inside nested regions (scf.for loops
/// generated by ttl-lower-to-loops). TileStoreOp clears a slot;
/// tile compute ops with dst_index produce slots.
static SmallVector<DSTSlotInfo> findLiveDSTSlots(Operation *dprintOp) {
  SmallVector<DSTSlotInfo> liveSlots;
  llvm::DenseMap<int64_t, std::pair<std::string, bool>> slotToOp;

  auto funcOp = dprintOp->getParentOfType<func::FuncOp>();
  if (!funcOp) {
    return liveSlots;
  }

  bool reachedDprint = false;
  funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (reachedDprint) {
      return WalkResult::interrupt();
    }
    if (op == dprintOp) {
      reachedDprint = true;
      return WalkResult::interrupt();
    }

    if (isa<TileStoreOp>(op)) {
      if (auto dstVal = getTileOpDstIndex(op)) {
        if (auto slot = foldIndexToConstant(*dstVal)) {
          slotToOp.erase(*slot);
        }
      }
    } else if (isTileComputeOp(op)) {
      if (auto dstVal = getTileOpDstIndex(op)) {
        if (auto slot = foldIndexToConstant(*dstVal)) {
          bool isF32 = false;
          if (op->getNumResults() > 0) {
            if (auto tileType =
                    dyn_cast<ttcore::TileType>(op->getResult(0).getType())) {
              isF32 = tileType.getElementType().isF32();
            }
          }
          slotToOp[*slot] = {op->getName().getStringRef().str(), isF32};
        }
      }
    }
    return WalkResult::advance();
  });

  for (auto &[slot, info] : slotToOp) {
    liveSlots.push_back({slot, info.first, info.second});
  }
  llvm::sort(liveSlots,
             [](const auto &a, const auto &b) { return a.slot < b.slot; });
  return liveSlots;
}

//===----------------------------------------------------------------------===//
// DPrint lowering pattern
//===----------------------------------------------------------------------===//

struct DPrintLowering : OpConversionPattern<DPrintOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DPrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    StringRef mode = op.getMode();
    auto thread = op.getThread();

    // Thread conditioning: open DPRINT_THREAD( wrapper.
    if (thread) {
      std::string macro;
      if (*thread == "math") {
        macro = "DPRINT_MATH(";
      } else if (*thread == "pack") {
        macro = "DPRINT_PACK(";
      } else if (*thread == "unpack") {
        macro = "DPRINT_UNPACK(";
      } else {
        return op.emitError("unsupported thread type: ") << *thread;
      }
      emitVerbatim(loc, macro, rewriter);
    }

    if (mode == "scalar") {
      auto stmt = buildScalarDPrintStmt(op);
      if (failed(stmt)) {
        return failure();
      }
      emitVerbatim(loc, *stmt + ";", rewriter);

    } else if (mode == "cb") {
      if (op.getArgv().size() != 1) {
        return op.emitError("cb mode requires exactly one operand");
      }
      auto cbIdx = resolveCBIndex(op.getArgv()[0], op);
      if (failed(cbIdx)) {
        return failure();
      }
      emitVerbatim(loc,
                   "DPRINT << ttmlir::CBPrinter(get_compile_time_arg_val(" +
                       std::to_string(*cbIdx) + ")) << ENDL();",
                   rewriter);

    } else if (mode == "tile") {
      if (op.getArgv().size() != 1) {
        return op.emitError("tile mode requires exactly one operand");
      }
      Value tileVal = op.getArgv()[0];

      // Trace tile back to its CB.
      Value cb = getAttachedCB(tileVal);
      if (!cb) {
        return op.emitError("cannot trace tile value back to a CB; "
                            "tile must come from cb_wait or cb_reserve");
      }
      auto cbIdx = resolveCBIndex(cb, op);
      if (failed(cbIdx)) {
        return failure();
      }
      // Inline tile print using TileSlice/SliceRange from dprint.h.
      std::string cbArg =
          "get_compile_time_arg_val(" + std::to_string(*cbIdx) + ")";
      emitVerbatim(loc, "{", rewriter);
      emitVerbatim(loc, "DPRINT << \"======\" << ENDL();", rewriter);
      emitVerbatim(loc, "for (uint16_t r = 0; r < 32; ++r) {", rewriter);
      emitVerbatim(loc,
                   "DPRINT << (uint)r << \" : \" << TileSlice(" + cbArg +
                       ", 0, SliceRange{.h0=(uint8_t)r, .h1=(uint8_t)(r+1), "
                       ".hs=1, .w0=0, .w1=32, .ws=1}, true, false) << ENDL();",
                   rewriter);
      emitVerbatim(loc, "}", rewriter);
      emitVerbatim(loc, "DPRINT << \"++++++\" << ENDL();", rewriter);
      emitVerbatim(loc, "}", rewriter);

    } else if (mode == "tensor") {
      if (op.getArgv().size() != 1) {
        return op.emitError("tensor mode requires exactly one operand");
      }
      Value tensorVal = op.getArgv()[0];
      auto tensorType = dyn_cast<RankedTensorType>(tensorVal.getType());
      if (!tensorType) {
        return op.emitError("tensor mode operand must be a RankedTensorType");
      }

      auto info = getTensorPrintInfo(tensorType.getElementType(), op);
      if (failed(info)) {
        return failure();
      }

      int64_t numPages = op.getNumPages().value_or(1);

      Value cb = getAttachedCB(tensorVal);
      if (cb) {
        // CB-backed tensor: data is already in L1 at the CB read pointer.
        auto cbIdx = resolveCBIndex(cb, op);
        if (failed(cbIdx)) {
          return failure();
        }
        std::string l1Addr = "get_read_ptr(get_compile_time_arg_val(" +
                             std::to_string(*cbIdx) + "))";
        emitVerbatim(loc, "{", rewriter);
        emitVerbatim(loc,
                     "volatile tt_l1_ptr " + info->cPtrType +
                         "* ptr = reinterpret_cast<volatile tt_l1_ptr " +
                         info->cPtrType + "*>(" + l1Addr + ");",
                     rewriter);
        emitVerbatim(loc,
                     "for (uint32_t page = 0; page < " +
                         std::to_string(numPages) + "; ++page) {",
                     rewriter);
        emitVerbatim(loc, "DPRINT << page << \": \";", rewriter);
        emitVerbatim(loc,
                     "for (uint32_t j = 0; j < " +
                         std::to_string(info->eltsPerPage) + "; ++j, ++ptr) {",
                     rewriter);
        emitVerbatim(loc, "DPRINT << " + info->formatter + "(*ptr) << \" \";",
                     rewriter);
        emitVerbatim(loc, "}", rewriter);
        emitVerbatim(loc, "DPRINT << ENDL();", rewriter);
        emitVerbatim(loc, "}", rewriter);
        emitVerbatim(loc, "}", rewriter);
      } else {
        // Tensor accessor: buffer_address() is a bank-relative address,
        // not a directly dereferenceable L1 pointer. Use TensorAccessor
        // + noc_async_read_tile to fetch each page into a scratch CB
        // buffer before printing.
        auto argIdx = getTensorFuncArgIndex(tensorVal);
        if (!argIdx) {
          return op.emitError(
              "cannot resolve tensor for page print: value must trace to "
              "attach_cb or be a function argument (tensor accessor)");
        }

        auto parentFunc = op->getParentOfType<func::FuncOp>();
        auto baseCTAAttr =
            parentFunc->getAttrOfType<IntegerAttr>("ttl.base_cta_index");
        auto crtaIndicesAttr =
            parentFunc->getAttrOfType<ArrayAttr>("ttl.crta_indices");
        if (!baseCTAAttr || !crtaIndicesAttr) {
          return op.emitError(
              "tensor accessor print requires ttl.base_cta_index and "
              "ttl.crta_indices attributes on parent function");
        }
        if (*argIdx >= crtaIndicesAttr.size()) {
          return op.emitError("tensor argument index out of range");
        }

        int64_t baseCTA = baseCTAAttr.getInt();
        if (baseCTA == 0) {
          return op.emitError(
              "tensor accessor page print requires at least one circular "
              "buffer for scratch space");
        }

        int64_t globalIdx =
            mlir::cast<IntegerAttr>(crtaIndicesAttr[*argIdx]).getInt();
        int32_t ctaIdx = static_cast<int32_t>(baseCTA + globalIdx);
        int32_t crtaIdx = static_cast<int32_t>(globalIdx);

        std::string ctaStr = std::to_string(ctaIdx);
        std::string crtaStr = std::to_string(crtaIdx);
        std::string pageSizeStr = std::to_string(info->pageSizeBytes);

        emitVerbatim(loc, "{", rewriter);
        emitVerbatim(loc,
                     "auto dprint_ta_args = TensorAccessorArgs<" + ctaStr +
                         ", " + crtaStr + ">();",
                     rewriter);
        emitVerbatim(loc,
                     "TensorAccessor dprint_ta(dprint_ta_args, "
                     "get_common_arg_val<uint32_t>(" +
                         crtaStr + "), " + pageSizeStr + ");",
                     rewriter);
        emitVerbatim(loc, "cb_reserve_back(get_compile_time_arg_val(0), 1);",
                     rewriter);
        emitVerbatim(loc,
                     "uint32_t dprint_scratch = "
                     "get_write_ptr(get_compile_time_arg_val(0));",
                     rewriter);
        emitVerbatim(loc,
                     "for (uint32_t page = 0; page < " +
                         std::to_string(numPages) + "; ++page) {",
                     rewriter);
        emitVerbatim(loc,
                     "noc_async_read_tile(page, dprint_ta, dprint_scratch);",
                     rewriter);
        emitVerbatim(loc, "noc_async_read_barrier();", rewriter);
        emitVerbatim(loc,
                     "volatile tt_l1_ptr " + info->cPtrType +
                         "* ptr = reinterpret_cast<volatile tt_l1_ptr " +
                         info->cPtrType + "*>(dprint_scratch);",
                     rewriter);
        emitVerbatim(loc, "DPRINT << page << \": \";", rewriter);
        emitVerbatim(loc,
                     "for (uint32_t j = 0; j < " +
                         std::to_string(info->eltsPerPage) + "; ++j) {",
                     rewriter);
        emitVerbatim(loc, "DPRINT << " + info->formatter + "(ptr[j]) << \" \";",
                     rewriter);
        emitVerbatim(loc, "}", rewriter);
        emitVerbatim(loc, "DPRINT << ENDL();", rewriter);
        emitVerbatim(loc, "}", rewriter);
        emitVerbatim(loc, "}", rewriter);
      }

    } else if (mode == "dst") {
      auto liveSlots = findLiveDSTSlots(op);
      StringRef label = op.getFmt();

      // Only bf16 dest register layout is supported. f32 uses a split
      // float16+mantissa16 layout that requires config register reads
      // and architecture-specific reconstruction.
      for (auto &info : liveSlots) {
        if (info.isFloat32) {
          return op.emitError("DST register print is not supported for f32 "
                              "dest format; only bf16 is currently supported");
        }
      }

      emitVerbatim(loc, "{", rewriter);
      if (!label.empty()) {
        emitVerbatim(loc,
                     "DPRINT << \"=== " + label.str() + " ===\" << ENDL();",
                     rewriter);
      }
      for (auto &info : liveSlots) {
        std::string slotStr = std::to_string(info.slot);
        emitVerbatim(loc,
                     "DPRINT << \"DST[" + slotStr + "] (" + info.opName +
                         ")\" << ENDL();",
                     rewriter);
        // Inline dest register read. dbg_read_dest_acc_row is available
        // from compute_kernel_api.h (no extra include needed). Reads
        // one row (8 x uint32) from the dest register file. Each uint32
        // holds two packed bf16 values. The read only executes on the
        // math thread (MATH wrapper). Prints first row of each face
        // (4 faces per tile, 16 rows per face).
        emitVerbatim(loc, "dbg_halt();", rewriter);
        emitVerbatim(loc, "MATH({", rewriter);
        emitVerbatim(loc, "  uint32_t rd_data[8];", rewriter);
        emitVerbatim(loc, "  for (uint16_t f = 0; f < 4; ++f) {", rewriter);
        emitVerbatim(loc,
                     "    dbg_read_dest_acc_row(" + slotStr +
                         " * 64 + f * 16, rd_data);",
                     rewriter);
        emitVerbatim(loc, "    DPRINT << \"  f\" << f << \": \";", rewriter);
        emitVerbatim(loc,
                     "    for (int i = 0; i < 8; ++i) { DPRINT << HEX() << "
                     "rd_data[i] << \" \"; }",
                     rewriter);
        emitVerbatim(loc, "    DPRINT << ENDL();", rewriter);
        emitVerbatim(loc, "  }", rewriter);
        emitVerbatim(loc, "})", rewriter);
        emitVerbatim(loc, "dbg_unhalt();", rewriter);
      }
      emitVerbatim(loc, "}", rewriter);

    } else {
      return op.emitError("unsupported dprint mode: ") << mode;
    }

    // Thread conditioning: close wrapper.
    if (thread) {
      emitVerbatim(loc, ");", rewriter);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

/// Add a hidden emitc.call_opaque "ttmlir::dprint" trigger to cause
/// ScopedModuleHelper (in tt-mlir's TTKernelToCpp) to emit the dprint
/// include and helpers at file scope. Wrapped in #if 0/#endif so it
/// compiles away. Created via OperationState to avoid EmitC C++ type
/// dependencies (different binary from tt-mlir).
static void addDPrintIncludeTrigger(func::FuncOp func, OpBuilder &builder) {
  builder.setInsertionPointToStart(&func.getBody().front());
  auto loc = func.getLoc();

  emitVerbatim(loc, "#if 0", builder);

  OperationState callState(loc, "emitc.call_opaque");
  callState.addAttribute("callee", builder.getStringAttr("ttmlir::dprint"));
  builder.create(callState);

  emitVerbatim(loc, "#endif", builder);
}

struct TTLLowerDPrintToEmitCPass
    : impl::TTLLowerDPrintToEmitCBase<TTLLowerDPrintToEmitCPass> {
  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    ModuleOp mod = getOperation();

    // Record which functions have dprint ops.
    llvm::DenseSet<func::FuncOp> funcsWithDPrint;
    mod.walk([&](DPrintOp op) {
      auto func = op->getParentOfType<func::FuncOp>();
      if (func) {
        funcsWithDPrint.insert(func);
      }
    });

    ConversionTarget target(ctx);
    target.addIllegalOp<DPrintOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(&ctx);
    patterns.add<DPrintLowering>(&ctx);

    if (failed(applyPartialConversion(mod, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Add include trigger for each function that had dprints.
    OpBuilder builder(&ctx);
    for (auto func : funcsWithDPrint) {
      addDPrintIncludeTrigger(func, builder);
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
