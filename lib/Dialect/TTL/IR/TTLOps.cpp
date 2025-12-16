// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "TTLOpsVerifyUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/DialectImplementation.h" // IWYU pragma: keep
#include "mlir/Support/LogicalResult.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h" // IWYU pragma: keep
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"    // IWYU pragma: keep
#include "llvm/ADT/TypeSwitch.h"               // IWYU pragma: keep

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

//===----------------------------------------------------------------------===//
// ComputeOp - TilingInterface implementations
//===----------------------------------------------------------------------===//

mlir::SmallVector<mlir::utils::IteratorType>
mlir::tt::ttl::ComputeOp::getLoopIteratorTypes() {
  mlir::SmallVector<mlir::utils::IteratorType> result;
  for (mlir::Attribute attr : getIteratorTypes()) {
    auto strAttr = mlir::cast<mlir::StringAttr>(attr);
    if (strAttr.getValue() == "parallel") {
      result.push_back(mlir::utils::IteratorType::parallel);
    } else if (strAttr.getValue() == "reduction") {
      result.push_back(mlir::utils::IteratorType::reduction);
    }
  }
  return result;
}

mlir::SmallVector<mlir::Range>
mlir::tt::ttl::ComputeOp::getIterationDomain(mlir::OpBuilder &b) {
  mlir::SmallVector<mlir::Range> domain;
  // Use the first output tensor shape to define the iteration domain.
  if (getOutputs().empty()) {
    return domain;
  }
  auto outTy =
      mlir::cast<mlir::RankedTensorType>(getOutputs().front().getType());
  mlir::Location loc = getLoc();
  for (int64_t i = 0; i < outTy.getRank(); ++i) {
    mlir::OpFoldResult offset = b.getIndexAttr(0);
    mlir::OpFoldResult stride = b.getIndexAttr(1);
    mlir::OpFoldResult size;
    if (outTy.isDynamicDim(i)) {
      size = b.create<mlir::tensor::DimOp>(loc, getOutputs().front(), i)
                 .getResult();
    } else {
      size = b.getIndexAttr(outTy.getDimSize(i));
    }
    domain.push_back(mlir::Range{offset, size, stride});
  }
  return domain;
}

mlir::FailureOr<mlir::TilingResult>
mlir::tt::ttl::ComputeOp::getTiledImplementation(
    mlir::OpBuilder &b, mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes) {
  // For now, return failure to indicate tiling is not yet fully implemented.
  // A full implementation would slice the inputs/outputs and clone this op.
  return mlir::failure();
}

mlir::LogicalResult mlir::tt::ttl::ComputeOp::getResultTilePosition(
    mlir::OpBuilder &b, unsigned resultNumber,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::SmallVector<mlir::OpFoldResult> &resultOffsets,
    mlir::SmallVector<mlir::OpFoldResult> &resultSizes) {
  // For identity indexing maps, result tile position equals the iteration tile.
  resultOffsets.assign(offsets.begin(), offsets.end());
  resultSizes.assign(sizes.begin(), sizes.end());
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ComputeOp - DestinationStyleOpInterface implementations
//===----------------------------------------------------------------------===//

mlir::MutableOperandRange mlir::tt::ttl::ComputeOp::getDpsInitsMutable() {
  return getOutputsMutable();
}

//===----------------------------------------------------------------------===//
// ComputeOp - Custom assembly format and verifier
//===----------------------------------------------------------------------===//

mlir::ParseResult
mlir::tt::ttl::ComputeOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  // Parse: ins(operands : types) outs(operands : types) attrs region -> results
  mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand> inputOperands;
  mlir::SmallVector<mlir::Type> inputTypes;
  mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand> outputOperands;
  mlir::SmallVector<mlir::Type> outputTypes;

  // Parse ins(...)
  if (parser.parseKeyword("ins") || parser.parseLParen()) {
    return mlir::failure();
  }
  if (parser.parseOptionalRParen()) {
    if (parser.parseOperandList(inputOperands) || parser.parseColon() ||
        parser.parseTypeList(inputTypes) || parser.parseRParen()) {
      return mlir::failure();
    }
  }

  // Parse outs(...)
  if (parser.parseKeyword("outs") || parser.parseLParen()) {
    return mlir::failure();
  }
  if (parser.parseOptionalRParen()) {
    if (parser.parseOperandList(outputOperands) || parser.parseColon() ||
        parser.parseTypeList(outputTypes) || parser.parseRParen()) {
      return mlir::failure();
    }
  }

  // Resolve operands
  if (parser.resolveOperands(inputOperands, inputTypes, parser.getNameLoc(),
                             result.operands) ||
      parser.resolveOperands(outputOperands, outputTypes, parser.getNameLoc(),
                             result.operands)) {
    return mlir::failure();
  }

  // Set operand segment sizes
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(inputOperands.size()),
                           static_cast<int32_t>(outputOperands.size())}));

  // Parse attributes
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return mlir::failure();
  }

  // Parse region
  mlir::Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{})) {
    return mlir::failure();
  }

  // Parse result types
  mlir::SmallVector<mlir::Type> resultTypes;
  if (parser.parseArrow()) {
    return mlir::failure();
  }
  if (parser.parseOptionalLParen()) {
    // Single result type
    mlir::Type singleType;
    if (parser.parseType(singleType)) {
      return mlir::failure();
    }
    resultTypes.push_back(singleType);
  } else {
    // Multiple result types in parens
    if (parser.parseTypeList(resultTypes) || parser.parseRParen()) {
      return mlir::failure();
    }
  }
  result.addTypes(resultTypes);

  return mlir::success();
}

void mlir::tt::ttl::ComputeOp::print(mlir::OpAsmPrinter &p) {
  // Print inputs (ins operands)
  p << " ins(";
  p.printOperands(getInputs());
  p << " : ";
  llvm::interleaveComma(getInputs().getTypes(), p);
  p << ")";

  // Print outputs (outs operands)
  p << " outs(";
  p.printOperands(getOutputs());
  p << " : ";
  llvm::interleaveComma(getOutputs().getTypes(), p);
  p << ")";

  // Print attributes (excluding operandSegmentSizes which is internal)
  SmallVector<mlir::StringRef> elidedAttrs = {"operandSegmentSizes"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  // Print the region
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);

  // Print result types
  p << " -> ";
  if (getResults().size() == 1) {
    p.printType(getResults().front().getType());
  } else {
    p << "(";
    llvm::interleaveComma(getResultTypes(), p);
    p << ")";
  }
}

mlir::LogicalResult mlir::tt::ttl::ComputeOp::verify() {
  // Verify that the number of indexing maps matches inputs + outputs.
  size_t numOperands = getInputs().size() + getOutputs().size();
  if (getIndexingMaps().size() != numOperands) {
    return emitOpError("expected ") << numOperands << " indexing maps but got "
                                    << getIndexingMaps().size();
  }
  // Verify iterator_types contains only "parallel" or "reduction".
  for (mlir::Attribute attr : getIteratorTypes()) {
    auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (!strAttr || (strAttr.getValue() != "parallel" &&
                     strAttr.getValue() != "reduction")) {
      return emitOpError(
          "iterator_types must contain only 'parallel' or 'reduction'");
    }
  }
  return mlir::success();
}
