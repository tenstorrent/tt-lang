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

  // TODO(#88): Add support for pipes and blocks as ttl.copy operands once those
  // IR types/ops land.

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

  // TODO(#89): Verify that the tensor tile/block shape and element type match
  // the CB element_type and shape/buffer_factor semantics.

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

void mlir::tt::ttl::ComputeOp::print(mlir::OpAsmPrinter &p) {
  // Print inputs (ins operands)
  p << " ins(";
  p.printOperands(getInputs());
  p << " : ";
  llvm::interleaveComma(getInputs().getTypes(), p);
  p << ")";

  // Print outputs (outs operands).
  p << " outs(";
  p.printOperands(getOutputs());
  p << " : ";
  llvm::interleaveComma(getOutputs().getTypes(), p);
  p << ")";

  // Print attributes (excluding operandSegmentSizes which is internal).
  SmallVector<mlir::StringRef> elidedAttrs = {"operandSegmentSizes"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  // Print the region.
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);

  // Print result types.
  p << " -> ";
  if (getResults().size() == 1) {
    p.printType(getResults().front().getType());
  } else {
    p << "(";
    llvm::interleaveComma(getResultTypes(), p);
    p << ")";
  }
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
  mlir::Location loc = getLoc();

  // For full-tiling (sizes match output tensor shape), we generate:
  //   scf.for loops over the iteration domain
  //   tensor.extract to get tiles at each position
  //   body operations
  //   tensor.insert to put results back
  //
  // This method is called by TilingInterface utilities (e.g., scf::tileUsing*)
  // which handle the loop generation. We just need to produce a tiled version
  // of the operation for a single tile position.

  // Get the output tensor to build the result type.
  if (getOutputs().empty()) {
    return emitOpError("expected at least one output");
  }

  // For identity maps with unit tile sizes (sizes all = 1), we extract a single
  // tile from each input/output and run the body on that single tile.
  // This is the common case for lowering to per-tile loops.

  // Extract tiles from inputs at the given position.
  // Use tensor.extract to get the tile element directly (not a slice).
  // The body expects tile-typed values, not tensor<1x1x...xtile>.
  SmallVector<Value> tiledInputs;
  for (Value input : getInputs()) {
    SmallVector<Value> indexValues;
    for (auto offset : offsets) {
      if (auto val = llvm::dyn_cast<Value>(offset)) {
        indexValues.push_back(val);
      } else {
        auto attr = llvm::cast<Attribute>(offset);
        int64_t intVal = mlir::cast<IntegerAttr>(attr).getInt();
        indexValues.push_back(b.create<arith::ConstantIndexOp>(loc, intVal));
      }
    }
    Value extracted = b.create<tensor::ExtractOp>(loc, input, indexValues);
    tiledInputs.push_back(extracted);
  }

  // Extract tiles from outputs at the given position.
  SmallVector<Value> tiledOutputs;
  for (Value output : getOutputs()) {
    SmallVector<Value> indexValues;
    for (auto offset : offsets) {
      if (auto val = llvm::dyn_cast<Value>(offset)) {
        indexValues.push_back(val);
      } else {
        auto attr = llvm::cast<Attribute>(offset);
        int64_t intVal = mlir::cast<IntegerAttr>(attr).getInt();
        indexValues.push_back(b.create<arith::ConstantIndexOp>(loc, intVal));
      }
    }
    Value extracted =
        b.create<tensor::ExtractOp>(loc, output, indexValues);
    tiledOutputs.push_back(extracted);
  }

  // Clone the body operations with the extracted tiles as new block arguments.
  // Map old block args to extracted values.
  Block &bodyBlock = getBody().front();
  IRMapping mapping;

  // Map input block args (first numInputs args).
  for (auto [idx, arg] : llvm::enumerate(getInputs())) {
    mapping.map(bodyBlock.getArgument(idx), tiledInputs[idx]);
  }
  // Map output block args (remaining args).
  size_t numInputs = getInputs().size();
  for (auto [idx, arg] : llvm::enumerate(getOutputs())) {
    mapping.map(bodyBlock.getArgument(numInputs + idx), tiledOutputs[idx]);
  }

  // Clone body operations (except terminator) into current insertion point.
  SmallVector<Value> yieldedValues;
  for (Operation &op : bodyBlock.without_terminator()) {
    b.clone(op, mapping);
  }

  // Get the yielded values from the terminator.
  auto yieldOp = cast<YieldOp>(bodyBlock.getTerminator());
  for (Value yieldVal : yieldOp.getValues()) {
    yieldedValues.push_back(mapping.lookupOrDefault(yieldVal));
  }

  // Return tiling result. The caller (scf::tileUsing*) will handle inserting
  // results back into the output tensor and generating the loop structure.
  mlir::TilingResult result;
  result.tiledValues = yieldedValues;

  // Compute result tile positions (for identity maps, same as iteration tile).
  for (size_t i = 0; i < getNumResults(); ++i) {
    result.generatedSlices.push_back(nullptr); // No slice op generated here.
  }

  return result;
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

  if (parser.parseKeyword("ins") || parser.parseLParen()) {
    return mlir::failure();
  }
  // If we did not see a ')', parse the operand list and types, then consume
  // the closing ')'.
  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseOperandList(inputOperands) || parser.parseColon() ||
        parser.parseTypeList(inputTypes) || parser.parseRParen()) {
      return mlir::failure();
    }
  }

  if (parser.parseKeyword("outs") || parser.parseLParen()) {
    return mlir::failure();
  }
  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseOperandList(outputOperands) || parser.parseColon() ||
        parser.parseTypeList(outputTypes) || parser.parseRParen()) {
      return mlir::failure();
    }
  }

  if (parser.resolveOperands(inputOperands, inputTypes, parser.getNameLoc(),
                             result.operands) ||
      parser.resolveOperands(outputOperands, outputTypes, parser.getNameLoc(),
                             result.operands)) {
    return mlir::failure();
  }

  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(inputOperands.size()),
                           static_cast<int32_t>(outputOperands.size())}));

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return mlir::failure();
  }

  mlir::Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{})) {
    return mlir::failure();
  }

  mlir::SmallVector<mlir::Type> resultTypes;
  if (parser.parseArrow()) {
    return mlir::failure();
  }
  if (parser.parseOptionalLParen()) {
    mlir::Type singleType;
    if (parser.parseType(singleType)) {
      return mlir::failure();
    }
    resultTypes.push_back(singleType);
  } else {
    if (parser.parseTypeList(resultTypes) || parser.parseRParen()) {
      return mlir::failure();
    }
  }
  result.addTypes(resultTypes);
  return mlir::success();
}

// Verify CB ops with tensor results (cb_reserve, cb_wait).
// Checks that result tensor shape and element type match the CB
// configuration.
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

mlir::LogicalResult mlir::tt::ttl::ComputeOp::verify() {
  // Verify body has exactly one block.
  if (getBody().getBlocks().size() != 1) {
    return emitOpError("body must have exactly one block");
  }

  Block &bodyBlock = getBody().front();
  size_t numInputs = getInputs().size();
  size_t numOutputs = getOutputs().size();
  size_t numOperands = numInputs + numOutputs;

  // Verify block argument count matches inputs + outputs.
  if (bodyBlock.getNumArguments() != numOperands) {
    return emitOpError("body block must have ")
           << numOperands << " arguments (matching inputs + outputs), but got "
           << bodyBlock.getNumArguments();
  }

  // Verify the number of indexing maps matches inputs + outputs.
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

  // Verify terminator is YieldOp.
  if (!bodyBlock.mightHaveTerminator()) {
    return emitOpError("body block must have a terminator");
  }
  if (!mlir::isa<YieldOp>(bodyBlock.getTerminator())) {
    return emitOpError("body block must be terminated with ttl.yield");
  }

  return mlir::success();
}

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
