// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "CommonRuntimeArgLayout.h"

#include "mlir/IR/BuiltinTypes.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "llvm/ADT/STLExtras.h"

#include <cassert>

namespace mlir::tt::ttl {

static int64_t getTensorArgumentCount(func::FuncOp function) {
  return llvm::count_if(function.getArguments(), [](BlockArgument argument) {
    return mlir::isa<RankedTensorType>(argument.getType());
  });
}

static int64_t getComputedReceiverDFBBaseCount(func::FuncOp function) {
  auto dfbIndices = function->getAttrOfType<DenseI32ArrayAttr>(
      kPipeComputedAddressDFBIndicesAttrName);
  return dfbIndices ? static_cast<int64_t>(dfbIndices.size()) : 0;
}

static int64_t getPipeResourceCount(ModuleOp module) {
  int64_t count = 0;
  if (auto scratchBytes =
          module->getAttrOfType<IntegerAttr>(kPipeSramScratchBytesAttrName)) {
    count += scratchBytes.getInt() > 0 ? 1 : 0;
  }
  if (auto globalSemaphoreCount = module->getAttrOfType<IntegerAttr>(
          kPipeGlobalSemaphoreCountAttrName)) {
    count += globalSemaphoreCount.getInt();
  }
  return count;
}

static bool functionHasFabricRoutes(func::FuncOp function) {
  auto routes = function->getAttrOfType<ArrayAttr>(kFabricRoutesAttrName);
  return routes && !routes.empty();
}

CommonRuntimeArgLayout::CommonRuntimeArgLayout(func::FuncOp function)
    : CommonRuntimeArgLayout(function,
                             getComputedReceiverDFBBaseCount(function)) {}

CommonRuntimeArgLayout::CommonRuntimeArgLayout(
    func::FuncOp function, int64_t computedReceiverDFBBaseCount)
    : computedReceiverDFBBaseCount(computedReceiverDFBBaseCount) {
  ModuleOp module = function->getParentOfType<ModuleOp>();
  assert(module && "kernel function must be nested in a module");
  assert(computedReceiverDFBBaseCount >= 0 &&
         "computed receiver DFB base count must be nonnegative");

  computedReceiverDFBBaseArgIndex = getTensorArgumentCount(function);
  pipeResourceBaseArgIndex =
      computedReceiverDFBBaseArgIndex + computedReceiverDFBBaseCount;
  pipeResourceCount = getPipeResourceCount(module);
  fabricRuntimeArgBaseIndex = pipeResourceBaseArgIndex + pipeResourceCount;
  hasFabricRuntimeArgBase = functionHasFabricRoutes(function);
  deviceCoordinateBaseArgIndex =
      fabricRuntimeArgBaseIndex + (hasFabricRuntimeArgBase ? 1 : 0);
}

int64_t
CommonRuntimeArgLayout::getComputedReceiverDFBBaseIndex(int64_t ordinal) const {
  assert(ordinal >= 0 && "runtime argument ordinal must be nonnegative");
  assert(ordinal < computedReceiverDFBBaseCount &&
         "computed receiver DFB ordinal is out of range");
  return computedReceiverDFBBaseArgIndex + ordinal;
}

int64_t CommonRuntimeArgLayout::getPipeResourceIndex(int64_t ordinal) const {
  assert(ordinal >= 0 && "runtime argument ordinal must be nonnegative");
  assert(ordinal < pipeResourceCount &&
         "PipeNet resource ordinal is out of range");
  return pipeResourceBaseArgIndex + ordinal;
}

int64_t CommonRuntimeArgLayout::getFabricRuntimeArgBaseIndex() const {
  assert(hasFabricRuntimeArgBase &&
         "kernel function has no compiler-managed fabric routes");
  return fabricRuntimeArgBaseIndex;
}

int64_t
CommonRuntimeArgLayout::getDeviceCoordinateIndex(int64_t ordinal) const {
  assert(ordinal >= 0 && "runtime argument ordinal must be nonnegative");
  return deviceCoordinateBaseArgIndex + ordinal;
}

} // namespace mlir::tt::ttl
