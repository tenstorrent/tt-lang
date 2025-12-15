// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "TTLOpsVerifyUtils.h"
#include "mlir/IR/DialectImplementation.h" // IWYU pragma: keep
#include "mlir/Support/LogicalResult.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"       // IWYU pragma: keep
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h" // IWYU pragma: keep
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"          // IWYU pragma: keep
#include "llvm/ADT/TypeSwitch.h"                     // IWYU pragma: keep

#define GET_OP_CLASSES
#include "ttlang/Dialect/TTL/IR/TTLOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrDefs.cpp.inc"

namespace mlir::tt::ttl {

void TTLDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrDefs.cpp.inc"
      >();
}

llvm::LogicalResult
SliceAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                  int64_t start, int64_t stop, int64_t step) {
  if (step == 0) {
    return emitError() << "slice step cannot be zero";
  }
  if (step > 0 && stop < start) {
    return emitError() << "slice stop (" << stop << ") must be >= start ("
                       << start << ") when step is positive";
  }
  if (step < 0 && stop > start) {
    return emitError() << "slice stop (" << stop << ") must be <= start ("
                       << start << ") when step is negative";
  }
  return llvm::success();
}

} // namespace mlir::tt::ttl

mlir::LogicalResult mlir::tt::ttl::CopyOp::verify() {
  auto srcTy = getSrc().getType();
  auto dstTy = getDst().getType();

  const bool srcIsCb = mlir::isa<CircularBufferType>(srcTy);
  const bool dstIsCb = mlir::isa<CircularBufferType>(dstTy);

  // MVP (no pipes): copy is between a TTNN tensor slice and a circular buffer
  // block. Exactly one side must be a CB.
  if (srcIsCb == dstIsCb) {
    return emitOpError()
           << "expects exactly one operand to be !ttl.cb; got src=" << srcTy
           << " dst=" << dstTy;
  }

  // TODO(ttl): Add support for pipes and blocks as ttl.copy operands once those
  // IR types/ops land.
  // Issue: #88.

  Type tensorTy = srcIsCb ? dstTy : srcTy;
  auto rankedTensorTy = mlir::dyn_cast<RankedTensorType>(tensorTy);
  if (!rankedTensorTy) {
    return emitOpError()
           << "expects the non-CB operand to be a ranked tensor; got "
           << tensorTy;
  }

  // TT-Lang programs operate on TTNN tensors. Require a TTNN layout encoding so
  // lowering can derive tile/addressing information.
  auto enc = rankedTensorTy.getEncoding();
  if (!enc || !mlir::isa<tt::ttnn::TTNNLayoutAttr>(enc)) {
    return emitOpError()
           << "expects tensor operand to carry TTNNLayout encoding; got "
           << rankedTensorTy;
  }

  // TODO(ttl): Verify that the tensor tile/block shape and element type match
  // the CB element_type and shape/buffer_factor semantics.
  // Issue: #89.

  // MVP: every transfer must be synchronized explicitly. Requiring a `ttl.wait`
  // use ensures we do not silently drop transfers.
  if (failed(mlir::tt::ttl::verify::isEventuallyWaitedOn(getOperation(),
                                                         getXf()))) {
    return failure();
  }

  return success();
}

mlir::LogicalResult mlir::tt::ttl::WaitOp::verify() {
  if (failed(
          mlir::tt::ttl::verify::isValidWaitOperand(getOperation(), getXf()))) {
    return failure();
  }
  return success();
}

namespace {

// Verify CB ops with tensor results (cb_reserve, cb_wait).
// Checks that result tensor shape and element type match the CB configuration.
mlir::LogicalResult verifyCBOpWithResult(mlir::Operation *op,
                                         mlir::tt::ttl::CircularBufferType cbTy,
                                         mlir::RankedTensorType resultTy) {
  auto cbShape = cbTy.getShape();
  auto resultShape = resultTy.getShape();

  if (cbShape.size() != resultShape.size()) {
    return op->emitOpError()
           << "result tensor rank (" << resultShape.size()
           << ") must match CB shape rank (" << cbShape.size() << ")";
  }

  for (size_t i = 0; i < cbShape.size(); ++i) {
    if (cbShape[i] != resultShape[i]) {
      return op->emitOpError()
             << "result tensor shape dimension " << i << " (" << resultShape[i]
             << ") must match CB shape dimension (" << cbShape[i] << ")";
    }
  }

  auto cbElemTy = cbTy.getElementType();
  auto resultElemTy = resultTy.getElementType();
  if (cbElemTy != resultElemTy) {
    return op->emitOpError()
           << "result tensor element type (" << resultElemTy
           << ") must match CB element type (" << cbElemTy << ")";
  }

  return mlir::success();
}

} // namespace

mlir::LogicalResult mlir::tt::ttl::CBReserveOp::verify() {
  auto cbTy = mlir::cast<CircularBufferType>(getCb().getType());
  auto resultTy = mlir::cast<RankedTensorType>(getResult().getType());
  return verifyCBOpWithResult(getOperation(), cbTy, resultTy);
}

mlir::LogicalResult mlir::tt::ttl::CBPushOp::verify() {
  // cb_push has no result to verify; the CB type is already enforced by
  // tablegen constraints.
  return success();
}

mlir::LogicalResult mlir::tt::ttl::CBWaitOp::verify() {
  auto cbTy = mlir::cast<CircularBufferType>(getCb().getType());
  auto resultTy = mlir::cast<RankedTensorType>(getResult().getType());
  return verifyCBOpWithResult(getOperation(), cbTy, resultTy);
}

mlir::LogicalResult mlir::tt::ttl::CBPopOp::verify() {
  // cb_pop has no result to verify; the CB type is already enforced by
  // tablegen constraints.
  return success();
}

mlir::LogicalResult mlir::tt::ttl::TileAddOp::verify() {
  auto lhsTy = getLhs().getType();
  auto rhsTy = getRhs().getType();

  // Operand types must match exactly (same shape and datatype).
  if (lhsTy != rhsTy) {
    return emitOpError() << "operand types must match; got lhs=" << lhsTy
                         << " rhs=" << rhsTy;
  }

  // Result type must match operand type.
  if (getResult().getType() != lhsTy) {
    return emitOpError() << "result type must match operand type; got result="
                         << getResult().getType() << " operand=" << lhsTy;
  }

  return success();
}

mlir::LogicalResult mlir::tt::ttl::StoreOp::verify() {
  auto valueTy = getValue().getType();
  auto destTy = mlir::cast<RankedTensorType>(getDest().getType());
  auto destElemTy = destTy.getElementType();

  // Destination tensor element type must be a tile type.
  if (!mlir::isa<tt::ttcore::TileType>(destElemTy)) {
    return emitOpError()
           << "destination tensor element type must be a tile type; got "
           << destElemTy;
  }

  // Value tile type must match destination element type.
  if (valueTy != destElemTy) {
    return emitOpError() << "value type must match destination element type; "
                         << "got value=" << valueTy
                         << " dest element=" << destElemTy;
  }

  return success();
}

mlir::LogicalResult mlir::tt::ttl::KernelOp::verify() {
  size_t numThreadAttrs = getThreads().size();
  size_t numRegions = getThreadRegions().size();

  if (numThreadAttrs != numRegions) {
    return emitOpError() << "threads attribute count (" << numThreadAttrs
                         << ") must match number of regions (" << numRegions
                         << ")";
  }

  if (numRegions == 0) {
    return emitOpError() << "requires at least one thread region";
  }

  // Check if regions have been extracted (all thread attrs have symbol refs).
  bool allExtracted = true;
  for (Attribute attr : getThreads()) {
    auto threadAttr = mlir::cast<ThreadAttr>(attr);
    if (!threadAttr.getSymbolRef()) {
      allExtracted = false;
      break;
    }
  }

  // After extraction, regions are empty - skip region verification.
  if (allExtracted) {
    // Verify number of results matches number of outputs.
    if (getResults().size() != getOutputs().size()) {
      return emitOpError() << "number of results (" << getResults().size()
                           << ") must match number of outputs ("
                           << getOutputs().size() << ")";
    }
    return success();
  }

  // Verify each region has a single entry block.
  for (unsigned i = 0; i < numRegions; ++i) {
    Region &region = getThreadRegions()[i];
    if (region.empty()) {
      return emitOpError() << "thread region " << i
                           << " must have an entry block";
    }
    if (!region.hasOneBlock()) {
      return emitOpError() << "thread region " << i
                           << " must have exactly one block";
    }
  }

  // Verify all regions have identical block argument types.
  Block &firstBlock = getThreadRegions().front().front();
  TypeRange firstArgTypes = firstBlock.getArgumentTypes();

  for (unsigned i = 1; i < numRegions; ++i) {
    Block &block = getThreadRegions()[i].front();
    TypeRange argTypes = block.getArgumentTypes();

    if (argTypes.size() != firstArgTypes.size()) {
      return emitOpError() << "thread region " << i << " has "
                           << argTypes.size() << " block arguments, expected "
                           << firstArgTypes.size()
                           << " (must match first region)";
    }

    for (unsigned j = 0; j < argTypes.size(); ++j) {
      if (argTypes[j] != firstArgTypes[j]) {
        return emitOpError()
               << "thread region " << i << " block argument " << j
               << " has type " << argTypes[j] << ", expected "
               << firstArgTypes[j] << " (must match first region)";
      }
    }
  }

  // Verify number of results matches number of outputs.
  if (getResults().size() != getOutputs().size()) {
    return emitOpError() << "number of results (" << getResults().size()
                         << ") must match number of outputs ("
                         << getOutputs().size() << ")";
  }

  return success();
}
