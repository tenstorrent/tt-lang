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

#include "ttlang/Dialect/TTL/IR/TTLInterfaces.cpp.inc"

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
  if (idx < 0) {
    return emitOpError() << "cb_index must be non-negative";
  }

  int64_t blockCount = getBlockCount();
  if (blockCount <= 0) {
    return emitOpError() << "block_count must be > 0";
  }
  if (blockCount != cbTy.getBlockCount()) {
    return emitOpError() << "block_count must match result type block count ("
                         << cbTy.getBlockCount() << ")";
  }

  return mlir::success();
}

mlir::LogicalResult mlir::tt::ttl::AttachCBOp::verify() {
  auto tensorTy = mlir::cast<RankedTensorType>(getTensor().getType());
  auto cbTy = mlir::cast<CircularBufferType>(getCb().getType());

  if (tensorTy.getElementType() != cbTy.getElementType()) {
    return emitOpError() << "tensor element type (" << tensorTy.getElementType()
                         << ") must match CB element type ("
                         << cbTy.getElementType() << ")";
  }

  // TODO: Revisit shape rank validation for tensors with TTL layout.
  // Device tensors have 4D device shape (grid + shard) while CBs have 2D shard
  // shape. For now, only validate element types match. The relationship between
  // tensor shape and CB shape needs further investigation.

  if (getResult().getType() != getTensor().getType()) {
    return emitOpError() << "result type must equal tensor operand type";
  }

  return mlir::success();
}

mlir::LogicalResult mlir::tt::ttl::TensorSliceOp::verify() {
  auto tensorTy = mlir::cast<RankedTensorType>(getTensor().getType());
  auto resultTy = mlir::cast<RankedTensorType>(getResult().getType());
  int64_t tensorRank = tensorTy.getRank();
  int64_t resultRank = resultTy.getRank();

  if (static_cast<int64_t>(getIndices().size()) != tensorRank) {
    return emitOpError() << "index count (" << getIndices().size()
                         << ") must match tensor rank (" << tensorRank << ")";
  }

  // Rank-reducing slices are allowed (see the op description for the squeeze
  // semantics); only an oversized result rank is invalid here.
  if (resultRank > tensorRank) {
    return emitOpError() << "result rank (" << resultRank
                         << ") cannot exceed tensor rank (" << tensorRank
                         << ")";
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
  const bool srcIsPipe = mlir::isa<PipeType>(srcTy);
  const bool dstIsPipe = mlir::isa<PipeType>(dstTy);

  if (srcIsPipe || dstIsPipe) {
    if (srcIsPipe && dstIsPipe) {
      return emitOpError() << "cannot copy directly between pipes";
    }
    if (dstIsPipe) {
      if (!srcIsCb) {
        return emitOpError()
               << "pipe send requires source operand to be !ttl.cb";
      }
      return success();
    }
    if (!findCBReserveForPipeReceive(getDst())) {
      return emitOpError() << "pipe receive requires a cb_reserve destination";
    }
    return success();
  }

  if (srcIsCb == dstIsCb) {
    return emitOpError()
           << "expects exactly one operand to be !ttl.cb; got src=" << srcTy
           << " dst=" << dstTy;
  }

  // Extract the transfer tensor type from the non-CB operand. For slices, this
  // is the slice result type because ttl.copy moves one DFB block at a time.
  Type nonCbTy = srcIsCb ? dstTy : srcTy;
  RankedTensorType transferTensorTy = mlir::dyn_cast<RankedTensorType>(nonCbTy);
  if (!transferTensorTy) {
    return emitOpError()
           << "expects the non-CB operand to be a ranked tensor or "
              "tensor_slice result; got "
           << nonCbTy;
  }

  // TT-Lang programs require a TTL layout encoding on tensors so lowering can
  // derive tile/addressing information. For slices, validate the source tensor
  // too so malformed IR cannot hide a missing layout behind a typed slice.
  RankedTensorType layoutTensorTy = transferTensorTy;
  if (srcIsSlice || dstIsSlice) {
    auto sliceOp = srcIsSlice ? getSrc().getDefiningOp<TensorSliceOp>()
                              : getDst().getDefiningOp<TensorSliceOp>();
    layoutTensorTy =
        mlir::cast<RankedTensorType>(sliceOp.getTensor().getType());
  }

  auto enc = layoutTensorTy.getEncoding();
  if (!enc || !mlir::isa<LayoutAttr>(enc)) {
    return emitOpError()
           << "expects tensor operand to carry ttl.layout encoding; got "
           << layoutTensorTy;
  }

  auto cbTy = mlir::cast<CircularBufferType>(srcIsCb ? srcTy : dstTy);
  auto cbShape = cbTy.getShape();
  auto tensorShape = transferTensorTy.getShape();

  if (cbShape.size() != tensorShape.size()) {
    return emitOpError() << "tensor rank (" << tensorShape.size()
                         << ") must match CB shape rank (" << cbShape.size()
                         << ")";
  }

  for (size_t i = 0; i < cbShape.size(); ++i) {
    if (cbShape[i] != tensorShape[i]) {
      return emitOpError() << "tensor shape dimension " << i << " ("
                           << tensorShape[i]
                           << ") must match CB shape dimension (" << cbShape[i]
                           << ")";
    }
  }

  if (transferTensorTy.getElementType() != cbTy.getElementType()) {
    return emitOpError() << "tensor element type ("
                         << transferTensorTy.getElementType()
                         << ") must match CB element type ("
                         << cbTy.getElementType() << ")";
  }

  return success();
}

mlir::LogicalResult mlir::tt::ttl::PipeTransferCreateOp::verify() {
  auto pipeType = mlir::cast<PipeType>(getPipe().getType());
  int64_t expectedReceivers = static_cast<int64_t>(getExpectedReceivers());
  if (expectedReceivers <= 0) {
    return emitOpError() << "requires positive expectedReceivers";
  }
  if (expectedReceivers != pipeType.getNumDests()) {
    return emitOpError()
           << "expectedReceivers must match the pipe receiver count";
  }

  switch (getKind().getValue()) {
  case PipeTransferKind::PointToPoint:
    if (!pipeType.hasSingleReceiver()) {
      return emitOpError() << "point_to_point transfer requires one receiver";
    }
    break;
  case PipeTransferKind::Collective:
    break;
  }

  return success();
}

mlir::LogicalResult mlir::tt::ttl::PipeTransferPostOp::verify() {
  if (!findCBReserveForPipeReceive(getDst())) {
    return emitOpError() << "requires a cb_reserve destination";
  }
  auto createOp = findPipeTransferCreateForTransfer(getTransfer());
  if (!createOp) {
    return emitOpError()
           << "requires transfer derived from ttl.pipe_transfer.create";
  }
  auto pipeType = mlir::cast<PipeType>(createOp.getPipe().getType());
  auto tokenType = mlir::cast<PipeTokenType>(getToken().getType());
  if (tokenType.getPipeNetId() != pipeType.getPipeNetId()) {
    return emitOpError() << "token pipeNetId must match transfer pipeNetId";
  }

  return success();
}

mlir::LogicalResult mlir::tt::ttl::PipeTransferSendOp::verify() {
  if (!findPipeTransferCreateForTransfer(getTransfer())) {
    return emitOpError()
           << "requires transfer derived from ttl.pipe_transfer.create";
  }
  auto handleType = mlir::dyn_cast<TransferHandleType>(getXf().getType());
  if (!handleType || handleType.getKind() != TransferKind::write) {
    return emitOpError() << "requires a write transfer handle result";
  }

  return success();
}

mlir::LogicalResult mlir::tt::ttl::PipeTransferWaitOp::verify() {
  mlir::tt::ttl::PipeTransferPostOp postOp =
      mlir::tt::ttl::findPipeTransferPostForToken(getToken());
  if (!postOp) {
    return emitOpError()
           << "requires token derived from ttl.pipe_transfer.post";
  }
  auto waitTokenType =
      mlir::cast<mlir::tt::ttl::PipeTokenType>(getToken().getType());
  auto postTokenType =
      mlir::cast<mlir::tt::ttl::PipeTokenType>(postOp.getToken().getType());
  if (waitTokenType.getPipeNetId() != postTokenType.getPipeNetId()) {
    return emitOpError()
           << "token pipeNetId must match pipe transfer post pipeNetId";
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

  auto computeOp = (*this)->getParentOfType<ComputeOp>();
  assert(computeOp && "ParentOneOf trait should enforce ComputeOp parent");

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

  auto dstTileTy = getDstTile().getType();
  if (dstTileTy != srcTy) {
    return emitOpError()
           << "dst_tile type must match src type, but got dst_tile: "
           << dstTileTy << ", src: " << srcTy;
  }

  return success();
}

mlir::LogicalResult mlir::tt::ttl::TileTypecastOp::verify() {
  auto inputTy = mlir::cast<tt::ttcore::TileType>(getInput().getType());
  auto resultTy = mlir::cast<tt::ttcore::TileType>(getResult().getType());

  // The tile shape must be preserved; identity typecasts fold away.
  if (inputTy.getShape() != resultTy.getShape()) {
    return emitOpError()
           << "input and result tile shapes must match, but got input: "
           << inputTy << ", result: " << resultTy;
  }

  ttcore::DataType inputDtype = inputTy.getDataType();
  ttcore::DataType resultDtype = resultTy.getDataType();
  if (!ttcore::isFloat(inputDtype) || !ttcore::isFloat(resultDtype)) {
    return emitOpError()
           << "only supports floating-point tile data types, but got input: "
           << inputTy << ", result: " << resultTy;
  }

  return success();
}

mlir::OpFoldResult mlir::tt::ttl::TypecastOp::fold(FoldAdaptor /*adaptor*/) {
  if (getInput().getType() == getResult().getType()) {
    return getInput();
  }

  return {};
}

mlir::OpFoldResult
mlir::tt::ttl::TileTypecastOp::fold(FoldAdaptor /*adaptor*/) {
  if (getInput().getType() == getResult().getType()) {
    return getInput();
  }

  return {};
}

void mlir::tt::ttl::ComputeOp::print(mlir::OpAsmPrinter &p) {
  p << " ins(";
  p.printOperands(getInputs());
  p << " : ";
  llvm::interleaveComma(getInputs().getTypes(), p);
  p << ")";

  p << " outs(";
  p.printOperands(getOutputs());
  p << " : ";
  llvm::interleaveComma(getOutputs().getTypes(), p);
  p << ")";

  SmallVector<mlir::StringRef> elidedAttrs = {"operandSegmentSizes"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);

  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);

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
  // Default to full operand dim for broadcast dims not in the map. Operand
  // shapes are static (enforced by the ComputeOp verifier).
  operandSizes = getAsIndexOpFoldResult(b.getContext(), operandTy.getShape());
  operandStrides.resize(rank, b.getIndexAttr(1));

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

  auto tiledOp = ComputeOp::create(
      b, loc, mlir::TypeRange(tiledOutputs), tiledInputs, tiledOutputs,
      getIndexingMapsAttr(), getIteratorTypesAttr());

  // Body tile_store ops capture the cb_reserve view from outside the compute;
  // when tiling, they must reference the sliced output so downstream lowering
  // can compute the correct global DFB offset from the extract_slice.
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
  mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand> inputOperands;
  mlir::SmallVector<mlir::Type> inputTypes;
  mlir::SmallVector<mlir::OpAsmParser::UnresolvedOperand> outputOperands;
  mlir::SmallVector<mlir::Type> outputTypes;

  if (parser.parseKeyword("ins") || parser.parseLParen()) {
    return mlir::failure();
  }
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
  if (getBody().getBlocks().size() != 1) {
    return emitOpError("body must have exactly one block");
  }

  Block &bodyBlock = getBody().front();
  size_t numInputs = getInputs().size();
  size_t numOutputs = getOutputs().size();
  size_t numOperands = numInputs + numOutputs;

  if (bodyBlock.getNumArguments() != numOperands) {
    return emitOpError("body block must have ")
           << numOperands << " arguments (matching inputs + outputs), but got "
           << bodyBlock.getNumArguments();
  }

  if (getResults().size() != numOutputs) {
    return emitOpError("expected ")
           << numOutputs << " results (one per output) but got "
           << getResults().size();
  }

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

  size_t expectedMaps = numInputs + numOutputs;
  if (mapsAttr.size() != expectedMaps) {
    return emitOpError("expected ")
           << expectedMaps << " indexing maps but got " << mapsAttr.size();
  }

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

  if (!bodyBlock.mightHaveTerminator()) {
    return emitOpError("body block must have a terminator");
  }
  if (!mlir::isa<YieldOp>(bodyBlock.getTerminator())) {
    return emitOpError("body block must be terminated with ttl.yield");
  }

  // Zero inputs are allowed for ops like fill that produce output without
  // input.
  if (getOutputs().empty()) {
    return emitOpError(
        "requires at least one output for SFPU packer configuration");
  }

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

  for (size_t d = 0; d < iteratorCount; ++d) {
    if (isReductionDim[d] && !dimsReferencedByInputs[d]) {
      return emitOpError()
             << "reduction dimension " << d
             << " must be referenced by at least one input indexing map";
    }
  }

  // tile_store is the only op that writes to the output CB (lowers to
  // pack_tile); each store's target CB must match a formal output CB.
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

namespace {

/// Convert a verified enum-attribute list to enum values for interface
/// consumers. The verifier enforces the attribute class before this helper
/// runs, so callers read a typed policy without rechecking.
template <typename AttrT, typename EnumT>
llvm::SmallVector<EnumT> getVerifiedEnumValues(mlir::ArrayAttr attrs) {
  llvm::SmallVector<EnumT> values;
  values.reserve(attrs.size());
  for (mlir::Attribute attr : attrs) {
    values.push_back(mlir::cast<AttrT>(attr).getValue());
  }
  return values;
}

} // namespace

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::tt::ttl::YieldOp::verify() {
  Operation *parent = getOperation()->getParentOp();
  if (parent && mlir::isa<AccumulationScopeOp>(parent)) {
    return mlir::success();
  }
  if (!getValues().empty()) {
    return emitOpError("operands are only supported in ttl.accumulation_scope");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AccumulationScopeOp - AccumulationScopeOpInterface implementations
//===----------------------------------------------------------------------===//

static mlir::ParseResult parseAccumulationScopeValueList(
    mlir::OpAsmParser &parser, llvm::StringRef keyword,
    llvm::SmallVectorImpl<mlir::OpAsmParser::UnresolvedOperand> &operands,
    llvm::SmallVectorImpl<mlir::Type> &types) {
  if (parser.parseKeyword(keyword) || parser.parseLParen()) {
    return mlir::failure();
  }
  if (mlir::failed(parser.parseOptionalRParen()) &&
      (parser.parseOperandList(operands) || parser.parseColon() ||
       parser.parseTypeList(types) || parser.parseRParen())) {
    return mlir::failure();
  }
  return mlir::success();
}

static mlir::ParseResult parseAccumulationInitialModeList(
    mlir::OpAsmParser &parser, llvm::SmallVectorImpl<mlir::Attribute> &attrs) {
  if (parser.parseKeyword("initial_modes") || parser.parseLParen()) {
    return mlir::failure();
  }
  if (parser.parseCommaSeparatedList(
          mlir::OpAsmParser::Delimiter::Square,
          [&]() -> mlir::ParseResult {
            llvm::StringRef keyword;
            llvm::SMLoc loc = parser.getCurrentLocation();
            if (parser.parseKeyword(&keyword)) {
              return mlir::failure();
            }
            std::optional<mlir::tt::ttl::AccumulationInitialMode> mode =
                mlir::tt::ttl::symbolizeAccumulationInitialMode(keyword);
            if (!mode) {
              return parser.emitError(loc)
                     << "expected accumulation initial mode `overwrite`, "
                        "`accumulate_existing`, or `init`";
            }
            attrs.push_back(mlir::tt::ttl::AccumulationInitialModeAttr::get(
                parser.getContext(), *mode));
            return mlir::success();
          }) ||
      parser.parseRParen()) {
    return mlir::failure();
  }
  return mlir::success();
}

mlir::ParseResult
mlir::tt::ttl::AccumulationScopeOp::parse(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> outputs;
  llvm::SmallVector<mlir::Type> outputTypes;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> inits;
  llvm::SmallVector<mlir::Type> initTypes;
  if (parseAccumulationScopeValueList(parser, "outs", outputs, outputTypes)) {
    return mlir::failure();
  }
  if (mlir::succeeded(parser.parseOptionalKeyword("inits")) &&
      (parser.parseLParen() || parser.parseOperandList(inits) ||
       parser.parseColon() || parser.parseTypeList(initTypes) ||
       parser.parseRParen())) {
    return mlir::failure();
  }

  if (parser.resolveOperands(outputs, outputTypes, parser.getNameLoc(),
                             result.operands) ||
      parser.resolveOperands(inits, initTypes, parser.getNameLoc(),
                             result.operands)) {
    return mlir::failure();
  }
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(outputs.size()),
                           static_cast<int32_t>(inits.size())}));

  mlir::Region *body = result.addRegion();
  if (parser.parseRegion(*body, /*arguments=*/{}, /*argTypes=*/{})) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Attribute> initialModes;
  if (parseAccumulationInitialModeList(parser, initialModes)) {
    return mlir::failure();
  }
  result.addAttribute("initial_modes",
                      parser.getBuilder().getArrayAttr(initialModes));

  return parser.parseOptionalAttrDict(result.attributes);
}

void mlir::tt::ttl::AccumulationScopeOp::print(mlir::OpAsmPrinter &p) {
  p << " outs(";
  p.printOperands(getOutputs());
  p << " : ";
  llvm::interleaveComma(getOutputs().getTypes(), p);
  p << ")";

  if (!getInits().empty()) {
    p << " inits(";
    p.printOperands(getInits());
    p << " : ";
    llvm::interleaveComma(getInits().getTypes(), p);
    p << ")";
  }

  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);

  p << " initial_modes([";
  llvm::interleaveComma(getAccumulationInitialModes(), p,
                        [&](mlir::tt::ttl::AccumulationInitialMode mode) {
                          p << mlir::tt::ttl::stringifyAccumulationInitialMode(
                              mode);
                        });
  p << "])";

  llvm::SmallVector<llvm::StringRef> elidedAttrs = {"operandSegmentSizes",
                                                    "initial_modes"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

/// Return true for all instances; the op has no non-accumulating form.
bool mlir::tt::ttl::AccumulationScopeOp::isAccumulation() { return true; }

/// Return destination tensors whose stores are governed by the scope policy.
mlir::ValueRange mlir::tt::ttl::AccumulationScopeOp::getAccumulationOutputs() {
  return getOutputs();
}

/// Return init operands, ordered by the init-mode outputs.
mlir::ValueRange mlir::tt::ttl::AccumulationScopeOp::getAccumulationInits() {
  return getInits();
}

/// Return one initial-value mode per output tensor.
llvm::SmallVector<mlir::tt::ttl::AccumulationInitialMode>
mlir::tt::ttl::AccumulationScopeOp::getAccumulationInitialModes() {
  return getVerifiedEnumValues<AccumulationInitialModeAttr,
                               AccumulationInitialMode>(getInitialModes());
}

/// Return the region containing the accumulation body.
mlir::Region &mlir::tt::ttl::AccumulationScopeOp::getAccumulationBody() {
  return getBody();
}

/// Verify that `ttl.accumulation_scope` contains a complete accumulation
/// policy without encoding a storage mechanism.
mlir::LogicalResult mlir::tt::ttl::AccumulationScopeOp::verify() {
  size_t outputCount = getOutputs().size();
  if (outputCount == 0) {
    return emitOpError("requires at least one output");
  }

  if (getInitialModes().size() != outputCount) {
    return emitOpError("requires one initial mode per output, got ")
           << getInitialModes().size() << " modes for " << outputCount
           << " outputs";
  }

  size_t initModeCount = 0;
  llvm::SmallVector<AccumulationInitialMode> initialModes;
  for (mlir::Attribute attr : getInitialModes()) {
    auto modeAttr = mlir::dyn_cast<AccumulationInitialModeAttr>(attr);
    if (!modeAttr) {
      return emitOpError(
          "initial_modes must contain accumulation initial-mode enum "
          "attributes");
    }
    AccumulationInitialMode mode = modeAttr.getValue();
    initialModes.push_back(mode);
    if (mode == AccumulationInitialMode::Init) {
      ++initModeCount;
    }
  }

  if (getInits().size() != initModeCount) {
    return emitOpError("requires one init operand per init mode, got ")
           << getInits().size() << " inits for " << initModeCount
           << " init modes";
  }

  // The operand segment contains only init operands. This preserves the
  // output-to-policy correspondence without unused operands for overwrite and
  // accumulate-existing outputs.
  size_t initIndex = 0;
  for (auto [outputIndex, mode] : llvm::enumerate(initialModes)) {
    if (mode != AccumulationInitialMode::Init) {
      continue;
    }
    mlir::Value output = getOutputs()[outputIndex];
    mlir::Value init = getInits()[initIndex++];
    if (output.getType() != init.getType()) {
      return emitOpError("init operand ")
             << (initIndex - 1) << " type " << init.getType()
             << " must match output " << outputIndex << " type "
             << output.getType();
    }
  }

  if (getBody().getBlocks().size() != 1) {
    return emitOpError("body must have exactly one block");
  }

  mlir::Block &bodyBlock = getBody().front();
  if (!bodyBlock.mightHaveTerminator()) {
    return emitOpError("body block must have a terminator");
  }
  if (!mlir::isa<YieldOp>(bodyBlock.getTerminator())) {
    return emitOpError("body block must be terminated with ttl.yield");
  }
  auto yield = mlir::cast<YieldOp>(bodyBlock.getTerminator());

  size_t bodyArgCount = bodyBlock.getNumArguments();
  size_t yieldedValueCount = yield.getValues().size();
  if (bodyArgCount != outputCount) {
    return emitOpError("body requires one block argument per output, got ")
           << bodyArgCount << " block arguments for " << outputCount
           << " outputs";
  }
  if (yieldedValueCount != outputCount) {
    return emitOpError("body must yield one value per output, got ")
           << yieldedValueCount << " yielded values for " << outputCount
           << " outputs";
  }

  for (auto [outputIndex, output] : llvm::enumerate(getOutputs())) {
    mlir::Type expectedType = output.getType();
    mlir::Type bodyArgType = bodyBlock.getArgument(outputIndex).getType();
    if (bodyArgType != expectedType) {
      return emitOpError("body argument ")
             << outputIndex << " type " << bodyArgType
             << " must match output type " << expectedType;
    }

    mlir::Value yieldedValue = yield.getValues()[outputIndex];
    if (yieldedValue.getType() != expectedType) {
      return emitOpError("yielded value ")
             << outputIndex << " type " << yieldedValue.getType()
             << " must match output type " << expectedType;
    }
  }

  bool hasNestedAccumulationScope = false;
  getBody().walk([&](AccumulationScopeOp) {
    hasNestedAccumulationScope = true;
    return mlir::WalkResult::interrupt();
  });
  // TODO(#648): Define nested and conditional accumulation scope semantics
  // before accepting nested ttl.accumulation_scope operations.
  if (hasNestedAccumulationScope) {
    return emitOpError(
        "nested ttl.accumulation_scope is not supported (#648); split nested "
        "accumulations into separate scopes");
  }

  return mlir::success();
}

// Verify a `num_tiles`-bearing acquire (cb_reserve / cb_wait): the result
// tensor must agree with the CB's element type, the tile-count attribute,
// and `num_tiles` must not exceed the CB's total tile capacity. The bound
// is across blocks (elementsPerBlock * blockCount) so coalesced acquires
// can span multiple CB blocks.
static mlir::LogicalResult
verifyCBAcquireWithNumTiles(mlir::Operation *op,
                            mlir::tt::ttl::CircularBufferType cbTy,
                            mlir::RankedTensorType resultTy, int64_t numTiles) {
  auto cbElemTy = cbTy.getElementType();
  if (cbElemTy != resultTy.getElementType()) {
    return op->emitOpError()
           << "result element type (" << resultTy.getElementType()
           << ") must match DFB element type (" << cbElemTy << ")";
  }
  int64_t resultTiles = 1;
  for (int64_t d : resultTy.getShape()) {
    resultTiles *= d;
  }
  if (resultTiles != numTiles) {
    return op->emitOpError()
           << "result tensor has " << resultTiles
           << " tiles but num_tiles attribute is " << numTiles;
  }
  int64_t cbCapacity = cbTy.getTotalElements();
  if (numTiles > cbCapacity) {
    return op->emitOpError() << "num_tiles (" << numTiles
                             << ") exceeds DFB capacity (" << cbCapacity << ")";
  }
  return mlir::success();
}

mlir::LogicalResult mlir::tt::ttl::CBReserveOp::verify() {
  auto cbTy = mlir::cast<CircularBufferType>(getCb().getType());
  auto resultTy = mlir::cast<RankedTensorType>(getResult().getType());

  if (getNumTiles()) {
    return verifyCBAcquireWithNumTiles(
        getOperation(), cbTy, resultTy,
        static_cast<int64_t>(getNumTiles().value()));
  }

  return verifyCBOpWithResult(getOperation(), cbTy, resultTy);
}

mlir::LogicalResult mlir::tt::ttl::CBPushOp::verify() {
  if (getNumTiles()) {
    auto cbTy = mlir::cast<CircularBufferType>(getCb().getType());
    int64_t cbCapacity = cbTy.getTotalElements();
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

  if (getNumTiles()) {
    return verifyCBAcquireWithNumTiles(
        getOperation(), cbTy, resultTy,
        static_cast<int64_t>(getNumTiles().value()));
  }

  return verifyCBOpWithResult(getOperation(), cbTy, resultTy);
}

mlir::Value mlir::tt::ttl::CBReserveOp::getViewSource() { return getCb(); }

mlir::Value mlir::tt::ttl::CBWaitOp::getViewSource() { return getCb(); }

mlir::LogicalResult mlir::tt::ttl::CBPopOp::verify() {
  if (getNumTiles()) {
    auto cbTy = mlir::cast<CircularBufferType>(getCb().getType());
    int64_t cbCapacity = cbTy.getTotalElements();
    int64_t numTiles = static_cast<int64_t>(getNumTiles().value());
    if (numTiles > cbCapacity) {
      return emitOpError() << "num_tiles (" << numTiles
                           << ") exceeds DFB capacity (" << cbCapacity << ")";
    }
  }
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

  // The view must ultimately come from a `ttl.cb_reserve`, possibly
  // through intervening `tensor.extract_slice` ops.
  if (!findCBReserveForView(getView())) {
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

//===----------------------------------------------------------------------===//
// DFBInputOpInterface implementations
//===----------------------------------------------------------------------===//

llvm::SmallVector<unsigned>
mlir::tt::ttl::ReduceOp::getDFBInputOperandIndices() {
  return {0, 1}; // input and scaler
}

llvm::SmallVector<unsigned>
mlir::tt::ttl::BlockBroadcastOp::getDFBInputOperandIndices() {
  return {0}; // input is the only operand; output CB is resolved downstream
}

llvm::SmallVector<unsigned>
mlir::tt::ttl::MatmulOp::getDFBInputOperandIndices() {
  return {0, 1}; // lhs and rhs
}

llvm::SmallVector<unsigned>
mlir::tt::ttl::TransposeOp::getDFBInputOperandIndices() {
  return {0}; // input
}

// True if `operand`'s producer is one whose result cannot fuse with a
// downstream compute and so must be packed out to a DFB.
static bool needsDFBMaterialization(mlir::Value operand) {
  mlir::Operation *defOp = operand.getDefiningOp();
  return defOp &&
         mlir::isa<mlir::tt::ttl::ReduceOp, mlir::tt::ttl::MatmulOp>(defOp);
}

llvm::SmallVector<unsigned>
mlir::tt::ttl::MulUnaryConstOp::getDFBInputOperandIndices() {
  if (needsDFBMaterialization(getInput())) {
    return {0};
  }
  return {};
}

llvm::SmallVector<unsigned> mlir::tt::ttl::MulOp::getDFBInputOperandIndices() {
  llvm::SmallVector<unsigned> indices;
  for (unsigned idx : {0u, 1u}) {
    if (needsDFBMaterialization(getOperand(idx))) {
      indices.push_back(idx);
    }
  }
  return indices;
}

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

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

  // When transpose_rhs is set, rhs is the transpose B stored as [N, K], so K
  // is rhs.shape[1] and N is rhs.shape[0].
  bool transposeRhs = getTransposeRhs();
  int64_t lhsK = lhsType.getDimSize(1);
  int64_t rhsK = transposeRhs ? rhsType.getDimSize(1) : rhsType.getDimSize(0);
  if (lhsK != rhsK) {
    return emitOpError() << "K dimension mismatch: lhs has " << lhsK
                         << " columns but rhs has " << rhsK
                         << (transposeRhs ? " columns" : " rows");
  }

  int64_t expectedM = lhsType.getDimSize(0);
  int64_t expectedN =
      transposeRhs ? rhsType.getDimSize(0) : rhsType.getDimSize(1);
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
// BlockBroadcastOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::tt::ttl::BlockBroadcastOp::verify() {
  auto inputType = mlir::cast<RankedTensorType>(getInput().getType());
  auto resultType = mlir::cast<RankedTensorType>(getResult().getType());

  if (!isa<ttcore::TileType>(inputType.getElementType())) {
    return emitOpError()
           << "row-major broadcast is not supported; input element type must "
              "be !ttcore.tile";
  }

  if (!inputType.hasStaticShape() || !resultType.hasStaticShape()) {
    return emitOpError() << "all operands must have static shapes";
  }

  ArrayRef<int64_t> dims = getDims();
  ArrayRef<int64_t> shape = getShape();

  int64_t rank = inputType.getRank();
  if (static_cast<int64_t>(shape.size()) != rank) {
    return emitOpError() << "shape size " << shape.size()
                         << " does not match input rank " << rank;
  }
  if (resultType.getRank() != rank) {
    return emitOpError() << "result rank " << resultType.getRank()
                         << " does not match input rank " << rank;
  }

  if (dims.empty()) {
    return emitOpError() << "dims must be non-empty";
  }

  llvm::SmallDenseSet<int64_t> normDims;
  for (int64_t d : dims) {
    int64_t normalized = normalizeDim(d, rank);
    if (normalized < 0 || normalized >= rank) {
      return emitOpError() << "dim " << d << " is out of range for rank "
                           << rank;
    }
    if (!normDims.insert(normalized).second) {
      return emitOpError() << "duplicate dim " << d;
    }
  }

  for (int64_t i = 0; i < rank; ++i) {
    if (normDims.contains(i)) {
      if (shape[i] <= 0) {
        return emitOpError()
               << "shape[" << i << "] = " << shape[i] << " must be positive";
      }
      if (inputType.getDimSize(i) != 1) {
        return emitOpError()
               << "input dim " << i << " is " << inputType.getDimSize(i)
               << " but must be 1 for broadcast dim " << i;
      }
    } else if (inputType.getDimSize(i) != shape[i]) {
      return emitOpError() << "input dim " << i << " is "
                           << inputType.getDimSize(i)
                           << " but must match shape[" << i
                           << "] = " << shape[i] << " for non-broadcast dim";
    }
    if (resultType.getDimSize(i) != shape[i]) {
      return emitOpError() << "result dim " << i << " is "
                           << resultType.getDimSize(i) << " but expected shape["
                           << i << "] = " << shape[i];
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
// FillOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult mlir::tt::ttl::FillOp::verify() {
  auto resultType = mlir::cast<RankedTensorType>(getResult().getType());
  if (!isa<ttcore::TileType>(resultType.getElementType())) {
    return emitOpError() << "result element type must be !ttcore.tile, got "
                         << resultType.getElementType();
  }
  if (!resultType.hasStaticShape()) {
    return emitOpError() << "result must have a static shape";
  }
  for (auto [i, dim] : llvm::enumerate(resultType.getShape())) {
    if (dim <= 0) {
      return emitOpError() << "result shape[" << i << "] = " << dim
                           << " must be positive";
    }
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

mlir::LogicalResult mlir::tt::ttl::CreatePipeOp::verify() {
  auto pipeType = mlir::cast<PipeType>(getResult().getType());

  // Verify consistency between attributes and result type.
  // Cast to int64_t to match the type's storage.
  int64_t srcX = static_cast<int64_t>(getSrcX());
  int64_t srcY = static_cast<int64_t>(getSrcY());
  int64_t dstStartX = static_cast<int64_t>(getDstStartX());
  int64_t dstStartY = static_cast<int64_t>(getDstStartY());
  int64_t dstEndX = static_cast<int64_t>(getDstEndX());
  int64_t dstEndY = static_cast<int64_t>(getDstEndY());

  int64_t pipeNetId = static_cast<int64_t>(getPipeNetId());

  if (pipeType.getSrcX() != srcX || pipeType.getSrcY() != srcY ||
      pipeType.getDstStartX() != dstStartX ||
      pipeType.getDstStartY() != dstStartY ||
      pipeType.getDstEndX() != dstEndX || pipeType.getDstEndY() != dstEndY ||
      pipeType.getPipeNetId() != pipeNetId) {
    return emitOpError() << "attributes must match result pipe type";
  }

  // Validate coordinates are non-negative.
  if (srcX < 0 || srcY < 0) {
    return emitOpError() << "source coordinates must be non-negative";
  }
  if (dstStartX < 0 || dstStartY < 0 || dstEndX < 0 || dstEndY < 0) {
    return emitOpError() << "destination coordinates must be non-negative";
  }

  // Spec NodeRange: each axis is `0 <= c_i < G_i`, so the destination
  // is a non-empty contiguous hypercube with `start <= end`.
  if (dstStartX > dstEndX || dstStartY > dstEndY) {
    return emitOpError()
           << "destination start must not exceed destination end on any axis";
  }

  bool hasMultipleReceivers = dstStartX != dstEndX || dstStartY != dstEndY;
  if (auto isCollectiveAttr = getIsCollectiveAttr();
      isCollectiveAttr && !isCollectiveAttr.getValue() &&
      hasMultipleReceivers) {
    return emitOpError()
           << "isCollective=false is invalid for a multi-receiver pipe";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Raw element access verifiers (shared logic + per-op entry points)
//===----------------------------------------------------------------------===//

/// Shared verification for raw_element_read and raw_element_write. Checks:
///   1. Enclosing function is a data movement (noc) kernel thread.
///   2. Block must trace to the expected CB acquire op (CBWaitOp for reads,
///      CBReserveOp for writes).
///   3. Block must be at least rank 1 (rank-0 not supported).
///   4. Coordinate count matches block tensor rank.
///   5. Scalar type matches block's underlying element dtype.
///
/// `ExpectedAcquireOp` is the CB acquire op type that the block must trace
/// to (CBWaitOp for reads, CBReserveOp for writes). `acquireName` is the
/// human-readable op name used in the diagnostic (e.g. "ttl.cb_wait").
template <typename ExpectedAcquireOp>
static mlir::LogicalResult
verifyRawElementOp(mlir::Operation *op, mlir::Value block,
                   mlir::RankedTensorType blockTy, mlir::ValueRange coords,
                   mlir::Type scalarTy, llvm::StringRef acquireName) {
  // 1. Must be inside a noc kernel thread function.
  auto func = mlir::tt::ttl::getEnclosingKernelThread(op);
  if (!func) {
    return op->emitOpError()
           << "must be inside a function with '"
           << mlir::tt::ttl::kKernelThreadAttrName << "' attribute";
  }
  auto threadAttr = func->getAttrOfType<mlir::tt::ttkernel::ThreadTypeAttr>(
      mlir::tt::ttl::kKernelThreadAttrName);
  if (!threadAttr ||
      threadAttr.getValue() != mlir::tt::ttkernel::ThreadType::Noc) {
    return op->emitOpError()
           << "is only allowed in data movement (noc) threads";
  }

  // 2. Block must trace to the expected CB acquire op.
  mlir::Operation *acquireOp = mlir::tt::ttl::findCBAcquireOp(block);
  if (!acquireOp) {
    return op->emitOpError()
           << "block must be a tensor view acquired from " << acquireName;
  }
  if (!mlir::isa<ExpectedAcquireOp>(acquireOp)) {
    return op->emitOpError() << "block must be acquired from " << acquireName
                             << ", but traces to " << acquireOp->getName();
  }

  // 3. Block must have at least one dimension.
  int64_t blockRank = blockTy.getRank();
  if (blockRank == 0) {
    return op->emitOpError()
           << "block must be at least rank 1, got rank-0 tensor";
  }

  // 4. Coordinate count must match block tensor rank.
  if (static_cast<int64_t>(coords.size()) != blockRank) {
    return op->emitOpError()
           << "coordinate count (" << coords.size()
           << ") must match block tensor rank (" << blockRank << ")";
  }

  // 5. Resolve the expected scalar type from the block element type.
  mlir::Type elemTy = blockTy.getElementType();
  mlir::Type expectedScalarTy;
  if (auto tileTy = mlir::dyn_cast<mlir::tt::ttcore::TileType>(elemTy)) {
    expectedScalarTy = mlir::tt::ttcore::dataTypeToElementType(
        op->getContext(), tileTy.getDataType());
  } else {
    expectedScalarTy = elemTy;
  }

  if (scalarTy != expectedScalarTy) {
    return op->emitOpError()
           << "scalar type (" << scalarTy
           << ") must match block element dtype (" << expectedScalarTy << ")";
  }

  return mlir::success();
}

mlir::LogicalResult mlir::tt::ttl::RawElementReadOp::verify() {
  auto blockTy = mlir::cast<RankedTensorType>(getBlock().getType());
  return verifyRawElementOp<mlir::tt::ttl::CBWaitOp>(
      getOperation(), getBlock(), blockTy, getCoords(), getResult().getType(),
      "ttl.cb_wait");
}

mlir::LogicalResult mlir::tt::ttl::RawElementWriteOp::verify() {
  auto blockTy = mlir::cast<RankedTensorType>(getBlock().getType());
  return verifyRawElementOp<mlir::tt::ttl::CBReserveOp>(
      getOperation(), getBlock(), blockTy, getCoords(), getValue().getType(),
      "ttl.cb_reserve");
}

//===----------------------------------------------------------------------===//
// PipeNetPredicateOpInterface implementations.
//===----------------------------------------------------------------------===//

int64_t mlir::tt::ttl::IsSrcOp::getReferencedPipeNetId() {
  return getPipeNetId();
}
mlir::tt::ttl::PipeRole mlir::tt::ttl::IsSrcOp::getReferencedRole() {
  return PipeRole::Source;
}

int64_t mlir::tt::ttl::IsDstOp::getReferencedPipeNetId() {
  return getPipeNetId();
}
mlir::tt::ttl::PipeRole mlir::tt::ttl::IsDstOp::getReferencedRole() {
  return PipeRole::Destination;
}

int64_t mlir::tt::ttl::IsActiveOp::getReferencedPipeNetId() {
  return getPipeNetId();
}
mlir::tt::ttl::PipeRole mlir::tt::ttl::IsActiveOp::getReferencedRole() {
  return PipeRole::Active;
}

//===----------------------------------------------------------------------===//
// RegionBranchOpInterface implementations for TTL region ops.
//
// `IfSrcOp` / `IfDstOp` execute the body conditionally on coord; from a
// type-system perspective both successors (body and parent-after-op) are
// possible, and the analysis decides which path applies via the lattice.
// `PipeNetScopeOp` is unconditional: control always enters the body.
//===----------------------------------------------------------------------===//

void mlir::tt::ttl::IfSrcOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getBody()));
    regions.push_back(RegionSuccessor(getOperation()));
    return;
  }
  regions.push_back(RegionSuccessor(getOperation()));
}

void mlir::tt::ttl::IfDstOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getBody()));
    regions.push_back(RegionSuccessor(getOperation()));
    return;
  }
  regions.push_back(RegionSuccessor(getOperation()));
}

void mlir::tt::ttl::PipeNetScopeOp::getSuccessorRegions(
    RegionBranchPoint point, SmallVectorImpl<RegionSuccessor> &regions) {
  if (point.isParent()) {
    regions.push_back(RegionSuccessor(&getBody()));
    return;
  }
  regions.push_back(RegionSuccessor(getOperation()));
}
