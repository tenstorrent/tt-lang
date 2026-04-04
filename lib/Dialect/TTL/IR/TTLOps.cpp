// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"

#include "TTLOpsVerifyUtils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h" // IWYU pragma: keep
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h" // IWYU pragma: keep
#include "ttlang/Dialect/TTL/IR/TTLOpsEnums.h" // IWYU pragma: keep
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "llvm/ADT/TypeSwitch.h" // IWYU pragma: keep
#include <cstdint>
#include <functional>
#include <numeric>

#define GET_OP_CLASSES
#include "ttlang/Dialect/TTL/IR/TTLOps.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrDefs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.cpp.inc"

namespace mlir::tt::ttl {

void TTLDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrDefs.cpp.inc"
      >();
}

void TTLDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.cpp.inc"
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

llvm::LogicalResult
LayoutAttr::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                   ArrayRef<int64_t> shape, Type elementType,
                   BufferType bufferType, ArrayRef<int64_t> grid,
                   TensorMemoryLayout memoryLayout) {
  if (shape.empty()) {
    return emitError() << "layout shape must not be empty";
  }
  if (grid.empty()) {
    return emitError() << "layout grid must not be empty";
  }
  for (int64_t dim : shape) {
    if (dim <= 0) {
      return emitError() << "layout shape dimensions must be positive, got "
                         << dim;
    }
  }
  for (int64_t dim : grid) {
    if (dim <= 0) {
      return emitError() << "layout grid dimensions must be positive, got "
                         << dim;
    }
  }
  return llvm::success();
}

} // namespace mlir::tt::ttl

mlir::LogicalResult mlir::tt::ttl::BindCBOp::verify() {
  auto cbTy = mlir::cast<CircularBufferType>(getResult().getType());

  int64_t idx = getCbIndexAttr().getInt();
  if (idx < 0 || idx >= kMaxCircularBuffers) {
    return emitOpError() << "cb_index must be in [0, "
                         << kMaxCircularBuffers - 1 << "]";
  }

  // Validate buffer factor against type for consistency.
  int64_t bufferFactor = getBufferFactor();
  if (bufferFactor <= 0) {
    return emitOpError() << "buffer_factor must be > 0";
  }
  if (bufferFactor != cbTy.getBufferFactor()) {
    return emitOpError()
           << "buffer_factor must match result type buffer factor ("
           << cbTy.getBufferFactor() << ")";
  }

  return mlir::success();
}

mlir::LogicalResult mlir::tt::ttl::AttachCBOp::verify() {
  auto tensorTy = mlir::cast<RankedTensorType>(getTensor().getType());
  auto cbTy = mlir::cast<CircularBufferType>(getCb().getType());

  // Element types must match.
  if (tensorTy.getElementType() != cbTy.getElementType()) {
    return emitOpError() << "tensor element type (" << tensorTy.getElementType()
                         << ") must match CB element type ("
                         << cbTy.getElementType() << ")";
  }

  // TODO: Revisit shape rank validation for tensors with TTL layout.
  // Device tensors have 4D device shape (grid + shard) while CBs have 2D shard
  // shape. For now, only validate element types match. The relationship between
  // tensor shape and CB shape needs further investigation.

  // Result type must equal input tensor type (identity).
  if (getResult().getType() != getTensor().getType()) {
    return emitOpError() << "result type must equal tensor operand type";
  }

  return mlir::success();
}

mlir::LogicalResult mlir::tt::ttl::TensorSliceOp::verify() {
  auto tensorTy = mlir::cast<RankedTensorType>(getTensor().getType());
  auto resultTy = mlir::cast<RankedTensorType>(getResult().getType());
  int64_t tensorRank = tensorTy.getRank();

  if (static_cast<int64_t>(getIndices().size()) != tensorRank) {
    return emitOpError() << "index count (" << getIndices().size()
                         << ") must match tensor rank (" << tensorRank << ")";
  }

  if (resultTy.getRank() != tensorRank) {
    return emitOpError() << "result rank (" << resultTy.getRank()
                         << ") must match tensor rank (" << tensorRank << ")";
  }

  if (resultTy.getElementType() != tensorTy.getElementType()) {
    return emitOpError() << "result element type (" << resultTy.getElementType()
                         << ") must match tensor element type ("
                         << tensorTy.getElementType() << ")";
  }

  return mlir::success();
}

mlir::LogicalResult mlir::tt::ttl::CopyOp::verify() {
  auto srcTy = getSrc().getType();
  auto dstTy = getDst().getType();

  const bool srcIsCb = mlir::isa<CircularBufferType>(srcTy);
  const bool dstIsCb = mlir::isa<CircularBufferType>(dstTy);
  const bool srcIsSlice = getSrc().getDefiningOp<TensorSliceOp>() != nullptr;
  const bool dstIsSlice = getDst().getDefiningOp<TensorSliceOp>() != nullptr;

  // Exactly one side must be a CB.
  if (srcIsCb == dstIsCb) {
    return emitOpError()
           << "expects exactly one operand to be !ttl.cb; got src=" << srcTy
           << " dst=" << dstTy;
  }

  // TODO(#88): Add support for pipes and blocks as ttl.copy operands once those
  // IR types/ops land.

  // Extract the underlying tensor type from the non-CB operand.
  // For slices, get the original tensor from the defining TensorSliceOp.
  Type nonCbTy = srcIsCb ? dstTy : srcTy;
  RankedTensorType rankedTensorTy;

  if (srcIsSlice || dstIsSlice) {
    auto sliceOp = srcIsSlice ? getSrc().getDefiningOp<TensorSliceOp>()
                              : getDst().getDefiningOp<TensorSliceOp>();
    rankedTensorTy =
        mlir::cast<RankedTensorType>(sliceOp.getTensor().getType());
  } else {
    rankedTensorTy = mlir::dyn_cast<RankedTensorType>(nonCbTy);
    if (!rankedTensorTy) {
      return emitOpError()
             << "expects the non-CB operand to be a ranked tensor or "
                "tensor_slice result; got "
             << nonCbTy;
    }
  }

  // TT-Lang programs require a TTL layout encoding on tensors so lowering
  // can derive tile/addressing information.
  auto enc = rankedTensorTy.getEncoding();
  if (!enc || !mlir::isa<LayoutAttr>(enc)) {
    return emitOpError()
           << "expects tensor operand to carry ttl.layout encoding; got "
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

mlir::LogicalResult mlir::tt::ttl::IterIndexOp::verify() {
  int64_t dim = getDim();

  // ParentOneOf<["ComputeOp"]> trait guarantees the parent is a ComputeOp.
  auto computeOp = (*this)->getParentOfType<ComputeOp>();
  assert(computeOp && "ParentOneOf trait should enforce ComputeOp parent");

  // Verify dim is within the iteration domain rank.
  unsigned iterRank = computeOp.getIteratorTypesArray().size();
  if (static_cast<unsigned>(dim) >= iterRank) {
    return emitOpError() << "dimension " << dim
                         << " is out of range for iteration domain of rank "
                         << iterRank;
  }

  return success();
}

mlir::LogicalResult mlir::tt::ttl::CopyTileOp::verify() {
  auto srcTy = mlir::cast<tt::ttcore::TileType>(getSrc().getType());

  // Verify that dst_tile type matches src type.
  auto dstTileTy = getDstTile().getType();
  if (dstTileTy != srcTy) {
    return emitOpError()
           << "dst_tile type must match src type, but got dst_tile: "
           << dstTileTy << ", src: " << srcTy;
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
// ComputeOp - Helper functions
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ComputeOp - DestinationStyleOpInterface implementations
//===----------------------------------------------------------------------===//

mlir::MutableOperandRange mlir::tt::ttl::ComputeOp::getDpsInitsMutable() {
  return getOutputsMutable();
}

//===----------------------------------------------------------------------===//
// ComputeOp - Helper methods (supplements IndexingMapOpInterface defaults)
//===----------------------------------------------------------------------===//

/// Convert the iterator_types attribute from string attrs ("parallel",
/// "reduction") to the utils::IteratorType enum.
mlir::SmallVector<mlir::utils::IteratorType>
mlir::tt::ttl::ComputeOp::getIteratorTypesArray() {
  mlir::SmallVector<mlir::utils::IteratorType> result;
  for (mlir::Attribute attr : getIteratorTypes()) {
    auto strAttr = mlir::cast<mlir::StringAttr>(attr);
    if (strAttr.getValue() == "parallel") {
      result.push_back(mlir::utils::IteratorType::parallel);
    } else {
      assert(strAttr.getValue() == "reduction" &&
             "verifier should have rejected non-parallel/reduction iterator");
      result.push_back(mlir::utils::IteratorType::reduction);
    }
  }
  return result;
}

/// Collect every dimension of every operand (inputs then outputs) into a flat
/// list of IndexAttrs. All dimensions are static (enforced by the verifier).
mlir::SmallVector<mlir::OpFoldResult>
mlir::tt::ttl::ComputeOp::createFlatListOfOperandDims(mlir::OpBuilder &b,
                                                      mlir::Location loc) {
  mlir::SmallVector<mlir::OpFoldResult> allDims;
  for (mlir::Value operand :
       llvm::concat<mlir::Value>(getInputs(), getOutputs())) {
    auto shape =
        mlir::cast<mlir::RankedTensorType>(operand.getType()).getShape();
    auto dims = getAsIndexOpFoldResult(b.getContext(), shape);
    allDims.append(dims.begin(), dims.end());
  }
  return allDims;
}

//===----------------------------------------------------------------------===//
// ComputeOp - TilingInterface implementations (used for subblocking)
//===----------------------------------------------------------------------===//

/// Map iteration-domain offsets/sizes to operand-space offsets/sizes/strides
/// via the indexing map. Simplified version of linalg's computeSliceParameters
/// (mlir/lib/Dialect/Linalg/Utils/Utils.cpp) for projected-permutation maps.
static void
mapOffsetsAndSizes(mlir::OpBuilder &b, mlir::Location loc, mlir::AffineMap map,
                   mlir::Value operand,
                   llvm::ArrayRef<mlir::OpFoldResult> offsets,
                   llvm::ArrayRef<mlir::OpFoldResult> sizes,
                   mlir::SmallVectorImpl<mlir::OpFoldResult> &operandOffsets,
                   mlir::SmallVectorImpl<mlir::OpFoldResult> &operandSizes,
                   mlir::SmallVectorImpl<mlir::OpFoldResult> &operandStrides) {
  auto operandTy = mlir::cast<mlir::RankedTensorType>(operand.getType());
  int64_t rank = operandTy.getRank();
  operandOffsets.resize(rank, b.getIndexAttr(0));
  // Default: full operand dim (used for broadcast dims not in the map).
  // All dimensions are static (enforced by the ComputeOp verifier).
  operandSizes = getAsIndexOpFoldResult(b.getContext(), operandTy.getShape());
  operandStrides.resize(rank, b.getIndexAttr(1));

  // Override with iteration-domain offsets/sizes for mapped dims.
  for (unsigned resIdx = 0; resIdx < map.getNumResults(); ++resIdx) {
    mlir::AffineExpr expr = map.getResult(resIdx);
    if (auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
      unsigned dimPos = dimExpr.getPosition();
      operandOffsets[resIdx] = offsets[dimPos];
      operandSizes[resIdx] = sizes[dimPos];
    }
  }
}

mlir::SmallVector<mlir::utils::IteratorType>
mlir::tt::ttl::ComputeOp::getLoopIteratorTypes() {
  return getIteratorTypesArray();
}

/// Use getShapesToLoopsMap() to look up which operand dimension provides
/// the bound for each loop.
mlir::SmallVector<mlir::Range>
mlir::tt::ttl::ComputeOp::getIterationDomain(mlir::OpBuilder &b) {
  mlir::SmallVector<mlir::Range> domain;
  mlir::Location loc = getLoc();

  mlir::SmallVector<mlir::OpFoldResult> allDims =
      createFlatListOfOperandDims(b, loc);
  mlir::AffineMap shapesToLoops = getShapesToLoopsMap();

  for (mlir::AffineExpr loopExpr : shapesToLoops.getResults()) {
    auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(loopExpr);
    assert(dimExpr &&
           "expected AffineDimExpr from inversePermutation of projected "
           "permutation indexing maps");
    mlir::OpFoldResult size = allDims[dimExpr.getPosition()];
    domain.push_back(mlir::Range{b.getIndexAttr(0), size, b.getIndexAttr(1)});
  }
  return domain;
}

mlir::SmallVector<int64_t>
mlir::tt::ttl::ComputeOp::getStaticIterationDomainSizes() {
  mlir::OpBuilder b(getOperation());
  mlir::SmallVector<mlir::Range> domain = getIterationDomain(b);
  mlir::SmallVector<int64_t> sizes;
  sizes.reserve(domain.size());
  for (auto &range : domain) {
    auto size = mlir::getConstantIntValue(range.size);
    assert(size && "ComputeOp verifier guarantees static shapes");
    sizes.push_back(*size);
  }
  return sizes;
}

int64_t mlir::tt::ttl::ComputeOp::getTotalIterationTiles() {
  auto sizes = getStaticIterationDomainSizes();
  return std::accumulate(sizes.begin(), sizes.end(), int64_t{1},
                         std::multiplies<>());
}

llvm::FailureOr<mlir::TilingResult>
mlir::tt::ttl::ComputeOp::getTiledImplementation(
    mlir::OpBuilder &b, llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes) {
  mlir::Location loc = getLoc();
  mlir::SmallVector<mlir::AffineMap> indexingMaps = getIndexingMapsArray();

  // Create extract_slice for each input operand.
  mlir::SmallVector<mlir::Value> tiledInputs;
  mlir::SmallVector<mlir::Operation *> generatedSlices;
  for (auto [idx, input] : llvm::enumerate(getInputs())) {
    mlir::SmallVector<mlir::OpFoldResult> operandOffsets, operandSizes,
        operandStrides;
    mapOffsetsAndSizes(b, loc, indexingMaps[idx], input, offsets, sizes,
                       operandOffsets, operandSizes, operandStrides);

    auto slice = mlir::tensor::ExtractSliceOp::create(
        b, loc, input, operandOffsets, operandSizes, operandStrides);
    tiledInputs.push_back(slice);
    generatedSlices.push_back(slice);
  }

  // Create extract_slice for each output operand.
  size_t numInputs = getInputs().size();
  mlir::SmallVector<mlir::Value> tiledOutputs;
  for (auto [idx, output] : llvm::enumerate(getOutputs())) {
    mlir::SmallVector<mlir::OpFoldResult> operandOffsets, operandSizes,
        operandStrides;
    mapOffsetsAndSizes(b, loc, indexingMaps[numInputs + idx], output, offsets,
                       sizes, operandOffsets, operandSizes, operandStrides);

    auto slice = mlir::tensor::ExtractSliceOp::create(
        b, loc, output, operandOffsets, operandSizes, operandStrides);
    tiledOutputs.push_back(slice);
    generatedSlices.push_back(slice);
  }

  // Build the tiled compute op with subblock operands.
  auto tiledOp = ComputeOp::create(
      b, loc, mlir::TypeRange(tiledOutputs), tiledInputs, tiledOutputs,
      getIndexingMapsAttr(), getIteratorTypesAttr());

  // Clone the body, remapping captured view references to tiled outputs.
  // The body's tile_store ops capture the cb_reserve view from outside the
  // compute. When tiling, these must reference the sliced output instead so
  // that downstream lowering can compute the correct global DFB offset from
  // the extract_slice. This applies uniformly to all computes (elementwise,
  // matmul, reduce, etc.): iter_index produces local (subblock) coordinates,
  // and addSliceOffset adds the global offset during TTL-to-TTKernel
  // conversion.
  mlir::IRMapping mapping;
  for (size_t i = 0; i < getOutputs().size(); ++i) {
    mlir::Value origOutput = getOutputs()[i];
    mlir::Value tiledOut = tiledOutputs[i];
    getBody().walk([&](TileStoreOp store) {
      mlir::Value view = store.getView();
      if (view.getParentRegion() == &getBody()) {
        return;
      }
      mlir::Value viewCB = getAttachedCB(view);
      mlir::Value outputCB = getAttachedCB(origOutput);
      if (viewCB && outputCB && viewCB == outputCB) {
        mapping.map(view, tiledOut);
      }
    });
  }
  getBody().cloneInto(&tiledOp.getBody(), mapping);

  mlir::TilingResult result;
  result.tiledOps.push_back(tiledOp);
  result.tiledValues = tiledOp.getResults();
  result.generatedSlices = std::move(generatedSlices);
  return result;
}

/// Map iteration-domain offsets/sizes to the result tensor's offsets/sizes
/// via the output's indexing map.
mlir::LogicalResult mlir::tt::ttl::ComputeOp::getResultTilePosition(
    mlir::OpBuilder &b, unsigned resultNumber,
    llvm::ArrayRef<mlir::OpFoldResult> offsets,
    llvm::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::SmallVector<mlir::OpFoldResult> &resultOffsets,
    mlir::SmallVector<mlir::OpFoldResult> &resultSizes) {
  mlir::Location loc = getLoc();
  mlir::SmallVector<mlir::AffineMap> indexingMaps = getIndexingMapsArray();
  mlir::AffineMap map = indexingMaps[getNumInputs() + resultNumber];
  mlir::Value output = getOutputs()[resultNumber];

  mlir::SmallVector<mlir::OpFoldResult> strides;
  mapOffsetsAndSizes(b, loc, map, output, offsets, sizes, resultOffsets,
                     resultSizes, strides);

  return mlir::success();
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
  if (succeeded(parser.parseOptionalLParen())) {
    if (parser.parseTypeList(resultTypes) || parser.parseRParen()) {
      return mlir::failure();
    }
  } else {
    mlir::Type singleType;
    if (parser.parseType(singleType)) {
      return mlir::failure();
    }
    resultTypes.push_back(singleType);
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

  // Verify result count matches output count (DPS semantics).
  if (getResults().size() != numOutputs) {
    return emitOpError("expected ")
           << numOutputs << " results (one per output) but got "
           << getResults().size();
  }

  // Verify block argument types match operand element types.
  for (size_t i = 0; i < numOperands; ++i) {
    Value operand =
        (i < numInputs) ? getInputs()[i] : getOutputs()[i - numInputs];
    auto tensorTy = mlir::dyn_cast<RankedTensorType>(operand.getType());
    if (!tensorTy) {
      continue;
    }
    Type expectedElemTy = tensorTy.getElementType();
    Type actualTy = bodyBlock.getArgument(i).getType();
    if (actualTy != expectedElemTy) {
      return emitOpError("block argument ")
             << i << " type " << actualTy
             << " does not match operand element type " << expectedElemTy;
    }
  }

  auto mapsAttr = getIndexingMaps();
  if (!mapsAttr) {
    return emitOpError("requires indexing_maps attribute");
  }

  // Verify the number of indexing maps matches inputs + outputs.
  size_t expectedMaps = numInputs + numOutputs;
  if (mapsAttr.size() != expectedMaps) {
    return emitOpError("expected ")
           << expectedMaps << " indexing maps but got " << mapsAttr.size();
  }

  // Verify iterator_types and track reduction dims for map validation.
  // Reduction dims may appear in input maps but must not appear in output maps.
  SmallVector<bool> isReductionDim(getIteratorTypes().size(), false);
  for (auto [idx, attr] : llvm::enumerate(getIteratorTypes())) {
    auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (!strAttr || (strAttr.getValue() != "parallel" &&
                     strAttr.getValue() != "reduction")) {
      return emitOpError(
          "iterator_types must contain only 'parallel' or 'reduction'");
    }
    if (strAttr.getValue() == "reduction") {
      isReductionDim[idx] = true;
    }
  }

  // Verify terminator is YieldOp.
  if (!bodyBlock.mightHaveTerminator()) {
    return emitOpError("body block must have a terminator");
  }
  if (!mlir::isa<YieldOp>(bodyBlock.getTerminator())) {
    return emitOpError("body block must be terminated with ttl.yield");
  }

  // Verify at least one output (required for SFPU packer configuration).
  // Zero inputs are allowed for ops like fill that produce output without
  // input.
  if (getOutputs().empty()) {
    return emitOpError(
        "requires at least one output for SFPU packer configuration");
  }

  // Verify indexing maps compatibility.
  auto iteratorCount = getIteratorTypes().size();
  auto maps = mapsAttr;

  // The iteration domain (from iterator_types) must be at least as large as the
  // maximum operand rank. Extra dimensions are reduction dims that do not
  // appear in any operand's shape (e.g., the K dimension in matmul: rank-2
  // operands with a 3D [M, N, K] iteration space).
  int64_t maxTensorRank = 0;
  for (Value operand : llvm::concat<Value>(getInputs(), getOutputs())) {
    auto ty = cast<RankedTensorType>(operand.getType());
    maxTensorRank = std::max(maxTensorRank, ty.getRank());
  }
  if (iteratorCount < static_cast<size_t>(maxTensorRank)) {
    return emitOpError("iterator_types count (")
           << iteratorCount << ") must be >= maximum tensor rank ("
           << maxTensorRank << ")";
  }

  auto verifyMapCommon = [&](AffineMap map,
                             size_t expectedResults) -> mlir::LogicalResult {
    if (map.getNumDims() != iteratorCount) {
      return emitOpError("indexing map expected ")
             << iteratorCount << " dims (iterator domain) but got "
             << map.getNumDims();
    }
    if (map.getNumResults() != expectedResults) {
      return emitOpError("indexing map expected ")
             << expectedResults << " results to match operand rank, but got "
             << map.getNumResults();
    }
    return success();
  };

  // Unlike linalg.generic (which allows arbitrary affine maps), ttl.compute
  // requires projected-permutation indexing maps: each result is a unique
  // dimension or a constant 0 (broadcast). This is sufficient for all spec
  // operations (element-wise, broadcast, matmul, reductions, transpose) and
  // enables downstream tiling and loop lowering to assume a direct
  // iteration-to-element mapping. Constant-0 results encode broadcast and
  // require the corresponding tensor dimension to be 1.
  // Examples of invalid maps: (d0, d1)->(d0 + d1), (d0, d1)->(1),
  // (d0, d1, d2)->(d0, d0), (d0)[s0]->(d0 + s0).
  auto validateMapStructure =
      [&](AffineMap map, RankedTensorType tensorTy, StringRef kind, size_t idx,
          SmallVectorImpl<bool> *dimsReferenced) -> mlir::LogicalResult {
    if (!map.isProjectedPermutation(/*allowZeroInResults=*/true)) {
      return emitOpError() << kind << " " << idx
                           << " indexing map must be a projected permutation"
                              " (unique dims or 0 constants)";
    }
    for (auto [resIdx, expr] : llvm::enumerate(map.getResults())) {
      if (auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
        if (dimsReferenced) {
          (*dimsReferenced)[dimExpr.getPosition()] = true;
        }
      } else if (auto cstExpr =
                     mlir::dyn_cast<mlir::AffineConstantExpr>(expr)) {
        if (tensorTy.getDimSize(resIdx) != 1) {
          return emitOpError() << kind << " " << idx << " broadcast dim "
                               << resIdx << " must have size 1";
        }
      }
    }
    return success();
  };

  // Ensure every tensor operand has an attached CB (via ttl.attach_cb).
  auto requireAttachedCB = [&](Value tensor, size_t idx,
                               StringRef kind) -> mlir::LogicalResult {
    Value cb = getAttachedCB(tensor);
    if (!cb) {
      return emitOpError() << kind << " " << idx
                           << " must have a circular buffer attached via "
                              "`ttl.attach_cb` or `ttl.cb_wait`";
    }
    return success();
  };

  // Inputs.
  SmallVector<bool> dimsReferencedByInputs(iteratorCount, false);
  for (size_t i = 0; i < numInputs; ++i) {
    auto tensorTy = mlir::cast<RankedTensorType>(getInputs()[i].getType());
    if (!tensorTy.hasStaticShape()) {
      return emitOpError("input ") << i << " must have a static shape";
    }
    if (failed(requireAttachedCB(getInputs()[i], i, "input"))) {
      return failure();
    }
    auto map = mlir::cast<AffineMapAttr>(maps[i]).getValue();
    if (failed(verifyMapCommon(map, tensorTy.getRank()))) {
      return failure();
    }
    if (failed(validateMapStructure(map, tensorTy, "input", i,
                                    &dimsReferencedByInputs))) {
      return failure();
    }
  }

  // Outputs.
  size_t outputStart = numInputs;
  for (size_t i = 0; i < numOutputs; ++i) {
    auto tensorTy = mlir::cast<RankedTensorType>(getOutputs()[i].getType());
    if (!tensorTy.hasStaticShape()) {
      return emitOpError("output ") << i << " must have a static shape";
    }
    if (failed(requireAttachedCB(getOutputs()[i], i, "output"))) {
      return failure();
    }
    size_t mapIdx = outputStart + i;
    auto map = mlir::cast<AffineMapAttr>(maps[mapIdx]).getValue();
    if (failed(verifyMapCommon(map, tensorTy.getRank()))) {
      return failure();
    }
    if (failed(validateMapStructure(map, tensorTy, "output", i,
                                    /*dimsReferenced=*/nullptr))) {
      return failure();
    }

    // Reduction dims must not appear in output maps. Like linalg.generic,
    // reduction dimensions are contracted: the body accumulates into the
    // output along these dims, so they do not index the output tensor.
    for (AffineExpr expr : map.getResults()) {
      if (auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
        if (isReductionDim[dimExpr.getPosition()]) {
          return emitOpError() << "output " << i
                               << " indexing map cannot reference reduction "
                                  "dimension "
                               << dimExpr.getPosition();
        }
      }
    }
  }

  // Every reduction dim must be referenced by at least one input map,
  // otherwise the reduction iterator has no operand to traverse.
  for (size_t d = 0; d < iteratorCount; ++d) {
    if (isReductionDim[d] && !dimsReferencedByInputs[d]) {
      return emitOpError()
             << "reduction dimension " << d
             << " must be referenced by at least one input indexing map";
    }
  }

  // The body must contain at least one tile_store. tile_store is the hardware
  // write (becomes pack_tile) and is the only mechanism for the compute to
  // produce observable output via pack to the output circular buffer.
  //
  // Each tile_store's target CB must match a formal output CB.
  DenseSet<Value> outputCBs;
  for (Value output : getOutputs()) {
    if (Value cb = getAttachedCB(output)) {
      outputCBs.insert(cb);
    }
  }

  DenseSet<Value> storedCBs;
  bool hasTileStore = false;
  for (Operation &op : bodyBlock.without_terminator()) {
    auto store = dyn_cast<TileStoreOp>(&op);
    if (!store) {
      continue;
    }
    hasTileStore = true;
    Value viewCB = getAttachedCB(store.getView());
    if (!viewCB) {
      return store.emitOpError() << "view must trace to a dataflow buffer";
    }
    if (!outputCBs.contains(viewCB)) {
      return store.emitOpError()
             << "stores to CB that is not a formal output of the compute";
    }
    storedCBs.insert(viewCB);
  }
  if (!hasTileStore) {
    return emitOpError("body must contain at least one ttl.tile_store");
  }

  for (Value output : getOutputs()) {
    if (Value cb = getAttachedCB(output)) {
      if (!storedCBs.contains(cb)) {
        return emitOpError("formal output CB has no tile_store in the body");
      }
    }
  }

  return success();
}

mlir::LogicalResult mlir::tt::ttl::CBReserveOp::verify() {
  auto cbTy = mlir::cast<CircularBufferType>(getCb().getType());
  auto resultTy = mlir::cast<RankedTensorType>(getResult().getType());

  // When `num_tiles` is present, the result shape is a subblock of the CB.
  // Verify element type match and that tile count is consistent.
  if (getNumTiles()) {
    auto cbElemTy = cbTy.getElementType();
    if (cbElemTy != resultTy.getElementType()) {
      return emitOpError() << "result element type ("
                           << resultTy.getElementType()
                           << ") must match DFB element type (" << cbElemTy
                           << ")";
    }
    int64_t resultTiles = 1;
    for (int64_t d : resultTy.getShape()) {
      resultTiles *= d;
    }
    if (resultTiles != static_cast<int64_t>(getNumTiles().value())) {
      return emitOpError() << "result tensor has " << resultTiles
                           << " tiles but num_tiles attribute is "
                           << getNumTiles().value();
    }
    int64_t cbCapacity = cbTy.getElementsPerBlock();
    if (resultTiles > cbCapacity) {
      return emitOpError() << "num_tiles (" << resultTiles
                           << ") exceeds DFB capacity (" << cbCapacity << ")";
    }
    return mlir::success();
  }

  return verifyCBOpWithResult(getOperation(), cbTy, resultTy);
}

mlir::LogicalResult mlir::tt::ttl::CBPushOp::verify() {
  if (getNumTiles()) {
    auto cbTy = mlir::cast<CircularBufferType>(getCb().getType());
    int64_t cbCapacity = cbTy.getElementsPerBlock();
    int64_t numTiles = static_cast<int64_t>(getNumTiles().value());
    if (numTiles > cbCapacity) {
      return emitOpError() << "num_tiles (" << numTiles
                           << ") exceeds DFB capacity (" << cbCapacity << ")";
    }
  }
  return success();
}

mlir::LogicalResult mlir::tt::ttl::CBWaitOp::verify() {
  auto cbTy = mlir::cast<CircularBufferType>(getCb().getType());
  auto resultTy = mlir::cast<RankedTensorType>(getResult().getType());
  return verifyCBOpWithResult(getOperation(), cbTy, resultTy);
}

mlir::Value mlir::tt::ttl::CBReserveOp::getViewSource() { return getCb(); }

mlir::Value mlir::tt::ttl::CBWaitOp::getViewSource() { return getCb(); }

mlir::LogicalResult mlir::tt::ttl::CBPopOp::verify() {
  // cb_pop has no result to verify; the CB type is already enforced by
  // tablegen constraints.
  return success();
}

mlir::LogicalResult mlir::tt::ttl::StoreOp::verify() {
  auto tensorTy = mlir::cast<RankedTensorType>(getTensor().getType());
  auto viewTy = mlir::cast<RankedTensorType>(getView().getType());

  if (tensorTy.getElementType() != viewTy.getElementType()) {
    return emitOpError() << "tensor element type (" << tensorTy.getElementType()
                         << ") must match view element type ("
                         << viewTy.getElementType() << ")";
  }

  if (tensorTy.getRank() != viewTy.getRank()) {
    return emitOpError() << "tensor rank (" << tensorTy.getRank()
                         << ") must match view rank (" << viewTy.getRank()
                         << ")";
  }

  for (int64_t i = 0; i < tensorTy.getRank(); ++i) {
    if (tensorTy.getDimSize(i) != viewTy.getDimSize(i)) {
      return emitOpError() << "tensor shape dimension " << i << " ("
                           << tensorTy.getDimSize(i)
                           << ") must match view shape dimension ("
                           << viewTy.getDimSize(i) << ")";
    }
  }

  if (!getView().getDefiningOp<CBReserveOp>()) {
    return emitOpError() << "view must come from ttl.cb_reserve";
  }

  return success();
}

mlir::LogicalResult mlir::tt::ttl::TileStoreOp::verify() {
  auto tileType = mlir::dyn_cast<ttcore::TileType>(getTile().getType());
  if (!tileType) {
    return emitOpError() << "tile operand must be !ttcore.tile, got "
                         << getTile().getType();
  }

  auto viewTy = mlir::cast<RankedTensorType>(getView().getType());
  auto viewElemTy = viewTy.getElementType();
  if (viewElemTy != tileType) {
    return emitOpError() << "view element type (" << viewElemTy
                         << ") must match tile type (" << tileType << ")";
  }

  // Inside a compute body, indices must match the view rank (populated by
  // convert-ttl-to-compute or assign-dst). Outside, allow empty indices.
  size_t numIndices = getIndices().size();
  bool insideCompute = (*this)->getParentOfType<ComputeOp>() != nullptr;
  if (insideCompute) {
    if (numIndices != static_cast<size_t>(viewTy.getRank())) {
      return emitOpError() << "expected " << viewTy.getRank()
                           << " indices inside compute body, got "
                           << numIndices;
    }
  } else if (numIndices != 0 &&
             numIndices != static_cast<size_t>(viewTy.getRank())) {
    return emitOpError() << "expected 0 or " << viewTy.getRank()
                         << " indices, got " << numIndices;
  }

  return success();
}

mlir::LogicalResult mlir::tt::ttl::MatmulOp::verify() {
  auto lhsType = mlir::cast<RankedTensorType>(getLhs().getType());
  auto rhsType = mlir::cast<RankedTensorType>(getRhs().getType());
  auto resultType = mlir::cast<RankedTensorType>(getResult().getType());

  if (lhsType.getRank() != 2) {
    return emitOpError() << "lhs must be rank 2, got rank "
                         << lhsType.getRank();
  }
  if (rhsType.getRank() != 2) {
    return emitOpError() << "rhs must be rank 2, got rank "
                         << rhsType.getRank();
  }
  if (resultType.getRank() != 2) {
    return emitOpError() << "result must be rank 2, got rank "
                         << resultType.getRank();
  }

  if (!lhsType.hasStaticShape()) {
    return emitOpError() << "lhs must have static shape";
  }
  if (!rhsType.hasStaticShape()) {
    return emitOpError() << "rhs must have static shape";
  }
  if (!resultType.hasStaticShape()) {
    return emitOpError() << "result must have static shape";
  }

  int64_t lhsK = lhsType.getDimSize(1);
  int64_t rhsK = rhsType.getDimSize(0);
  if (lhsK != rhsK) {
    return emitOpError() << "K dimension mismatch: lhs has " << lhsK
                         << " columns but rhs has " << rhsK << " rows";
  }

  int64_t expectedM = lhsType.getDimSize(0);
  int64_t expectedN = rhsType.getDimSize(1);
  if (resultType.getDimSize(0) != expectedM ||
      resultType.getDimSize(1) != expectedN) {
    return emitOpError() << "result shape [" << resultType.getDimSize(0) << ", "
                         << resultType.getDimSize(1) << "] does not match "
                         << "expected [" << expectedM << ", " << expectedN
                         << "]";
  }

  if (lhsType.getElementType() != rhsType.getElementType()) {
    return emitOpError() << "element type mismatch: lhs has "
                         << lhsType.getElementType() << " but rhs has "
                         << rhsType.getElementType();
  }

  if (resultType.getElementType() != lhsType.getElementType()) {
    return emitOpError() << "result element type "
                         << resultType.getElementType()
                         << " must match input element type "
                         << lhsType.getElementType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// ReduceOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::tt::ttl::ReduceOp::verify() {
  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());
  auto scalerType = mlir::cast<RankedTensorType>(getScaler().getType());
  auto resultType = mlir::cast<RankedTensorType>(getResult().getType());

  if (inputType.getRank() != 2) {
    return emitOpError() << "input must be rank 2, got rank "
                         << inputType.getRank();
  }
  if (scalerType.getRank() != 2) {
    return emitOpError() << "scaler must be rank 2, got rank "
                         << scalerType.getRank();
  }
  if (resultType.getRank() != 2) {
    return emitOpError() << "result must be rank 2, got rank "
                         << resultType.getRank();
  }

  if (!inputType.hasStaticShape() || !scalerType.hasStaticShape() ||
      !resultType.hasStaticShape()) {
    return emitOpError() << "all operands must have static shapes";
  }

  // Normalize and validate dims.
  ArrayRef<int64_t> dims = getDims();
  if (dims.empty()) {
    return emitOpError() << "dims must be non-empty";
  }

  int64_t rank = inputType.getRank();
  llvm::SmallDenseSet<int64_t> normDims;
  for (int64_t d : dims) {
    int64_t normalized = d < 0 ? d + rank : d;
    if (normalized < 0 || normalized >= rank) {
      return emitOpError() << "dim " << d << " is out of range for rank "
                           << rank;
    }
    if (!normDims.insert(normalized).second) {
      return emitOpError() << "duplicate dim " << d;
    }
  }

  // Verify result shape: reduced dims must be 1, others must match input.
  for (int64_t i = 0; i < rank; ++i) {
    int64_t expected = normDims.contains(i) ? 1 : inputType.getDimSize(i);
    if (resultType.getDimSize(i) != expected) {
      return emitOpError() << "result dim " << i << " is "
                           << resultType.getDimSize(i) << " but expected "
                           << expected;
    }
  }

  // Scaler must be a single tile (1, 1): one scaling value applied to every
  // reduction.  The hardware reduce_tile reads one scaler tile from srcB.
  for (int64_t i = 0; i < rank; ++i) {
    if (scalerType.getDimSize(i) != 1) {
      return emitOpError() << "scaler dim " << i << " is "
                           << scalerType.getDimSize(i) << " but must be 1";
    }
  }

  if (inputType.getElementType() != resultType.getElementType()) {
    return emitOpError() << "result element type "
                         << resultType.getElementType()
                         << " must match input element type "
                         << inputType.getElementType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::tt::ttl::TransposeOp::verify() {
  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());
  auto resultType = mlir::cast<RankedTensorType>(getResult().getType());

  if (inputType.getRank() != 2) {
    return emitOpError() << "input must be rank 2, got rank "
                         << inputType.getRank();
  }
  if (resultType.getRank() != 2) {
    return emitOpError() << "result must be rank 2, got rank "
                         << resultType.getRank();
  }

  if (!inputType.hasStaticShape() || !resultType.hasStaticShape()) {
    return emitOpError() << "all operands must have static shapes";
  }

  if (resultType.getDimSize(0) != inputType.getDimSize(1) ||
      resultType.getDimSize(1) != inputType.getDimSize(0)) {
    return emitOpError() << "result shape [" << resultType.getDimSize(0) << ", "
                         << resultType.getDimSize(1)
                         << "] must be the transpose of input shape ["
                         << inputType.getDimSize(0) << ", "
                         << inputType.getDimSize(1) << "]";
  }

  if (inputType.getElementType() != resultType.getElementType()) {
    return emitOpError() << "result element type "
                         << resultType.getElementType()
                         << " must match input element type "
                         << inputType.getElementType();
  }

  return success();
}
