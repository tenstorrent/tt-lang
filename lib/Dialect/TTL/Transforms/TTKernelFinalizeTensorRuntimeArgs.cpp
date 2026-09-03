// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>
#include <optional>

namespace ttk = mlir::tt::ttkernel;

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTKERNELFINALIZETENSORRUNTIMEARGS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

struct CommonArgIndexUse {
  ttk::GetCommonArgValOp get;
  ttk::ConstantTableLookupOp table;
  SmallVector<int64_t> originalIndices;
};

struct TensorAccessorArgsIndexUse {
  ttk::TensorAccessorArgsOp accessorArgs;
  int64_t originalIndex;
};

// Verbatim C++ may read runtime arguments without an analyzable MLIR use.
static bool containsHiddenCommonArgAccess(func::FuncOp function) {
  bool hiddenAccess = false;
  function.walk([&](emitc::VerbatimOp verbatim) {
    if (verbatim.getValue().contains("get_common_arg_val")) {
      hiddenAccess = true;
    }
  });
  return hiddenAccess;
}

static LogicalResult
readGlobalTensorIndices(func::FuncOp function, ArrayAttr indicesAttr,
                        SmallVectorImpl<int64_t> &indices) {
  for (Attribute attribute : indicesAttr) {
    auto integer = dyn_cast<IntegerAttr>(attribute);
    if (!integer || integer.getInt() < 0) {
      function.emitOpError() << kCRTAIndicesAttrName
                             << " must contain non-negative integer values";
      return failure();
    }
    indices.push_back(integer.getInt());
  }
  return success();
}

// Record every resolvable index for rewriting. An unresolved index requires the
// complete tensor prefix to remain stable.
static LogicalResult classifyCommonArgIndices(
    func::FuncOp function, int64_t tensorCount, BitVector &liveTensorSlots,
    SmallVectorImpl<CommonArgIndexUse> &uses, bool &hasUnresolvedIndex) {
  WalkResult walkResult = function.walk([&](ttk::GetCommonArgValOp get) {
    Value index = traceUnrealizedCasts(get.getArgIndex());
    APInt constantIndex;
    if (matchPattern(index, m_ConstantInt(&constantIndex))) {
      int64_t value = constantIndex.getSExtValue();
      if (value < 0) {
        get.emitOpError("common runtime argument index must be non-negative");
        return WalkResult::interrupt();
      }
      if (value < tensorCount) {
        liveTensorSlots.set(value);
      }
      uses.push_back({get, nullptr, {value}});
      return WalkResult::advance();
    }

    auto table = index.getDefiningOp<ttk::ConstantTableLookupOp>();
    if (!table) {
      hasUnresolvedIndex = true;
      return WalkResult::advance();
    }
    SmallVector<int64_t> values(table.getValues().begin(),
                                table.getValues().end());
    for (int64_t value : values) {
      if (value < tensorCount) {
        liveTensorSlots.set(value);
      }
    }
    uses.push_back({get, table, std::move(values)});
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted() ? failure() : success();
}

// LocalTensorAccessor lowering derives its bank base directly from one tensor
// common runtime argument.
static FailureOr<int64_t>
getLocalTensorSlot(ttk::LocalTensorAccessorOp accessor, int64_t tensorCount) {
  Value bankBase = traceUnrealizedCasts(accessor.getBankBaseAddressIn());
  auto get = bankBase.getDefiningOp<ttk::GetCommonArgValOp>();
  if (!get) {
    return accessor.emitOpError(
        "requires a structurally visible common runtime argument");
  }
  APInt constantIndex;
  if (!matchPattern(traceUnrealizedCasts(get.getArgIndex()),
                    m_ConstantInt(&constantIndex))) {
    return accessor.emitOpError("requires a constant tensor-address index");
  }
  int64_t slot = constantIndex.getSExtValue();
  if (slot < 0 || slot >= tensorCount) {
    return accessor.emitOpError("tensor-address index ")
           << slot << " is outside [0, " << tensorCount << ")";
  }
  return slot;
}

// TensorAccessorArgs stores its common-argument base separately from direct
// common-argument reads, so it participates in the same compaction analysis.
static LogicalResult classifyTensorAccessorArgsIndices(
    func::FuncOp function, int64_t tensorCount, BitVector &liveTensorSlots,
    SmallVectorImpl<TensorAccessorArgsIndexUse> &uses,
    bool &hasUnresolvedIndex) {
  WalkResult walkResult = function.walk([&](ttk::TensorAccessorArgsOp op) {
    Value crtaBase = op.getCrtaBase();
    if (!crtaBase) {
      hasUnresolvedIndex = true;
      return WalkResult::advance();
    }
    APInt constantIndex;
    if (!matchPattern(traceUnrealizedCasts(crtaBase),
                      m_ConstantInt(&constantIndex))) {
      hasUnresolvedIndex = true;
      return WalkResult::advance();
    }
    int64_t value = constantIndex.getSExtValue();
    if (value < 0) {
      op.emitOpError("common runtime argument index must be non-negative");
      return WalkResult::interrupt();
    }
    if (value < tensorCount) {
      liveTensorSlots.set(value);
    }
    uses.push_back({op, value});
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted() ? failure() : success();
}

// Tensor addresses form a prefix. Removing prefix entries shifts every
// following compiler-managed argument by the same count.
static int64_t
remapCommonArgIndex(int64_t originalIndex,
                    ArrayRef<std::optional<int64_t>> tensorSlotMap,
                    int64_t tensorCount, int64_t removedCount) {
  if (originalIndex >= tensorCount) {
    return originalIndex - removedCount;
  }
  assert(tensorSlotMap[originalIndex] &&
         "referenced tensor runtime argument must be retained");
  return *tensorSlotMap[originalIndex];
}

static LogicalResult finalizeFunction(func::FuncOp function) {
  auto crtaIndices = function->getAttrOfType<ArrayAttr>(kCRTAIndicesAttrName);
  if (!crtaIndices) {
    return success();
  }

  SmallVector<int64_t> globalTensorIndices;
  if (failed(readGlobalTensorIndices(function, crtaIndices,
                                     globalTensorIndices))) {
    return failure();
  }
  int64_t tensorCount = globalTensorIndices.size();
  BitVector liveTensorSlots(tensorCount);
  SmallVector<CommonArgIndexUse> commonArgUses;
  SmallVector<TensorAccessorArgsIndexUse> tensorAccessorArgsUses;
  bool hasUnresolvedIndex = false;
  if (failed(classifyCommonArgIndices(function, tensorCount, liveTensorSlots,
                                      commonArgUses, hasUnresolvedIndex))) {
    return failure();
  }
  if (failed(classifyTensorAccessorArgsIndices(
          function, tensorCount, liveTensorSlots, tensorAccessorArgsUses,
          hasUnresolvedIndex))) {
    return failure();
  }

  BitVector localTensorSlots(tensorCount);
  WalkResult localWalk =
      function.walk([&](ttk::LocalTensorAccessorOp accessor) {
        FailureOr<int64_t> slot = getLocalTensorSlot(accessor, tensorCount);
        if (failed(slot)) {
          return WalkResult::interrupt();
        }
        localTensorSlots.set(*slot);
        return WalkResult::advance();
      });
  if (localWalk.wasInterrupted()) {
    return failure();
  }

  bool preserveTensorPrefix =
      hasUnresolvedIndex || containsHiddenCommonArgAccess(function);
  if (preserveTensorPrefix) {
    liveTensorSlots.set();
  }

  SmallVector<std::optional<int64_t>> tensorSlotMap(tensorCount);
  SmallVector<Attribute> retainedGlobalIndices;
  SmallVector<Attribute> localGlobalIndices;
  OpBuilder builder(function.getContext());
  for (int64_t slot = 0; slot < tensorCount; ++slot) {
    if (liveTensorSlots.test(slot)) {
      tensorSlotMap[slot] = retainedGlobalIndices.size();
      retainedGlobalIndices.push_back(
          builder.getI32IntegerAttr(globalTensorIndices[slot]));
    }
    if (localTensorSlots.test(slot)) {
      localGlobalIndices.push_back(
          builder.getI32IntegerAttr(globalTensorIndices[slot]));
    }
  }
  function->setAttr(kCRTAIndicesAttrName,
                    builder.getArrayAttr(retainedGlobalIndices));
  if (localGlobalIndices.empty()) {
    function->removeAttr(kLocalTensorIndicesAttrName);
  } else {
    function->setAttr(kLocalTensorIndicesAttrName,
                      builder.getArrayAttr(localGlobalIndices));
  }

  if (preserveTensorPrefix) {
    return success();
  }

  int64_t removedCount = tensorCount - retainedGlobalIndices.size();
  llvm::DenseMap<Operation *, Value> remappedTables;
  for (CommonArgIndexUse &use : commonArgUses) {
    if (use.table) {
      auto [iterator, inserted] =
          remappedTables.try_emplace(use.table.getOperation());
      if (inserted) {
        SmallVector<int64_t> remappedValues;
        for (int64_t index : use.originalIndices) {
          remappedValues.push_back(remapCommonArgIndex(
              index, tensorSlotMap, tensorCount, removedCount));
        }
        builder.setInsertionPointAfter(use.table);
        iterator->second = ttk::ConstantTableLookupOp::create(
            builder, use.table.getLoc(), use.table.getType(),
            use.table.getIndex(), builder.getDenseI64ArrayAttr(remappedValues));
      }
      use.get.getArgIndexMutable().assign(iterator->second);
      continue;
    }

    int64_t remappedIndex = remapCommonArgIndex(
        use.originalIndices.front(), tensorSlotMap, tensorCount, removedCount);
    builder.setInsertionPoint(use.get);
    Type indexType = use.get.getArgIndex().getType();
    auto constant =
        arith::ConstantOp::create(builder, use.get.getLoc(), indexType,
                                  IntegerAttr::get(indexType, remappedIndex));
    use.get.getArgIndexMutable().assign(constant.getResult());
  }
  for (TensorAccessorArgsIndexUse &use : tensorAccessorArgsUses) {
    int64_t remappedIndex = remapCommonArgIndex(
        use.originalIndex, tensorSlotMap, tensorCount, removedCount);
    builder.setInsertionPoint(use.accessorArgs);
    Type indexType = use.accessorArgs.getCrtaBase().getType();
    auto constant =
        arith::ConstantOp::create(builder, use.accessorArgs.getLoc(), indexType,
                                  IntegerAttr::get(indexType, remappedIndex));
    use.accessorArgs.getCrtaBaseMutable().assign(constant.getResult());
  }
  for (auto &entry : remappedTables) {
    if (entry.first->use_empty()) {
      entry.first->erase();
    }
  }
  return success();
}

struct TTKernelFinalizeTensorRuntimeArgsPass
    : public impl::TTKernelFinalizeTensorRuntimeArgsBase<
          TTKernelFinalizeTensorRuntimeArgsPass> {
  using Base::Base;

  void runOnOperation() override {
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      if (failed(finalizeFunction(function))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
