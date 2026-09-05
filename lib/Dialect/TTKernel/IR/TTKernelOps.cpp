// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.h"

#include "ttlang/Dialect/TTCore/IR/TTCoreOpsTypes.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsTypes.h"
#include "ttlang/Dialect/Utils/OpaqueCallVerifyUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

#define GET_OP_CLASSES
#include "ttlang/Dialect/TTKernel/IR/TTKernelOps.cpp.inc"

namespace mlir::tt::ttkernel {

::mlir::LogicalResult ExperimentalRowNormalizationBlockOp::verify() {
  if (getNumTiles() < 1 || getNumTiles() > 8) {
    return emitOpError("num_tiles must be in the range [1, 8]");
  }
  if (!getHasGamma() && getGammaCb() != getInputCb()) {
    return emitOpError("gamma_cb must equal input_cb when has_gamma is false");
  }
  if (getDtype() != ttcore::DataType::BFloat16) {
    return emitOpError("supports bf16 DFBs only");
  }

  const llvm::APFloat &scale = getScaleAttr().getValue();
  const llvm::APFloat &epsilon = getEpsilonAttr().getValue();
  if (!scale.isFinite() || scale.isZero() || scale.isNegative()) {
    return emitOpError("scale must be finite and positive");
  }
  if (!epsilon.isFinite() || epsilon.isZero() || epsilon.isNegative()) {
    return emitOpError("epsilon must be finite and positive");
  }

  auto inputCBType = mlir::cast<CBType>(getInputCb().getType());
  auto gammaCBType = mlir::cast<CBType>(getGammaCb().getType());
  auto outputCBType = mlir::cast<CBType>(getOutputCb().getType());
  if (inputCBType.getElementType() != outputCBType.getElementType()) {
    return emitOpError("input and output dataflow buffer types must match");
  }
  if (getHasGamma() &&
      gammaCBType.getElementType() != outputCBType.getElementType()) {
    return emitOpError("gamma and output dataflow buffer types must match");
  }
  auto outputTileType =
      mlir::dyn_cast<ttcore::TileType>(outputCBType.getElementType());
  if (!outputTileType || outputTileType.getDataType() != getDtype()) {
    return emitOpError("dtype must match the output tile data type");
  }
  return success();
}

void ComputeKernelHWStartupOp::print(::mlir::OpAsmPrinter &printer) {
  printer << "(" << getIcb0();
  if (getIcb1()) {
    printer << ", " << getIcb1();
  }
  printer << ", " << getOcb() << ")";
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : ";
  printer.printFunctionalType(getOperation()->getOperandTypes(),
                              getOperation()->getResultTypes());
}

::mlir::ParseResult
ComputeKernelHWStartupOp::parse(::mlir::OpAsmParser &parser,
                                ::mlir::OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand, 3> operands;
  OpAsmParser::UnresolvedOperand operand;

  if (parser.parseLParen() || parser.parseOperand(operand)) {
    return failure();
  }
  operands.push_back(operand);

  while (succeeded(parser.parseOptionalComma())) {
    if (parser.parseOperand(operand)) {
      return failure();
    }
    operands.push_back(operand);
  }

  if (operands.size() != 2 && operands.size() != 3) {
    return parser.emitError(parser.getNameLoc()) << "expected 2 or 3 operands";
  }

  if (parser.parseRParen() || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColon()) {
    return failure();
  }

  FunctionType functionType;
  if (parser.parseType(functionType)) {
    return failure();
  }

  ArrayRef<Type> operandTypes = functionType.getInputs();
  if (operandTypes.size() != operands.size()) {
    return parser.emitError(parser.getNameLoc())
           << "expected " << operands.size() << " operand types";
  }

  result.addTypes(functionType.getResults());
  if (parser.resolveOperands(operands, operandTypes, parser.getNameLoc(),
                             result.operands)) {
    return failure();
  }

  return success();
}

static bool insideKernelFunction(mlir::Operation *op) {
  mlir::Operation *parentOp = op->getParentOp();

  if (!parentOp) {
    return false;
  }

  if (dyn_cast_if_present<func::FuncOp>(parentOp) &&
      dyn_cast_if_present<mlir::ModuleOp>(parentOp->getParentOp())) {
    return true;
  }
  return insideKernelFunction(parentOp);
}

::mlir::LogicalResult CBPushBackOp::verify() {
  if (!insideKernelFunction(getOperation())) {
    return emitOpError("CBPushBackOp must be inside a kernel function");
  }
  return success();
}

::mlir::LogicalResult CBPopFrontOp::verify() {
  if (!insideKernelFunction(getOperation())) {
    return emitOpError("CBPopFrontOp must be inside a kernel function");
  }
  return success();
}

::mlir::LogicalResult CBReserveBackOp::verify() {
  if (!insideKernelFunction(getOperation())) {
    return emitOpError("CBReserveBackOp must be inside a kernel function");
  }
  return success();
}

::mlir::LogicalResult CBWaitFrontOp::verify() {
  if (!insideKernelFunction(getOperation())) {
    return emitOpError("CBWaitFrontOp must be inside a kernel function");
  }
  return success();
}

::mlir::LogicalResult PackWaitedTileOp::verify() {
  if (!getOutOfOrder()) {
    return emitOpError("requires out_of_order packing");
  }
  auto dfbType = getOutCb().getType();
  uint64_t capacityTiles = static_cast<uint64_t>(dfbType.getNumElements());
  if (getAcquiredTiles() != capacityTiles) {
    return emitOpError() << "acquired_tiles must equal DFB capacity "
                         << capacityTiles << ", got " << getAcquiredTiles();
  }
  return success();
}

static std::string verifyTilizeUntilizeCBs(CBType tilizedCB, CBType scalarCB) {
  if (mlir::isa<ttcore::TileType>(scalarCB.getElementType())) {
    return "Input to TilizeOp or Output to UntilizeOp must have scalar "
           "element type";
  }
  if (!mlir::isa<ttcore::TileType>(tilizedCB.getElementType())) {
    return "Input to UntilizeOp or Output to TilizeOp must have tile "
           "element type";
  }
  return std::string();
}

static ::mlir::LogicalResult verifyPackUntilizeDims(Operation *op,
                                                    int32_t colsPerDstPass,
                                                    int32_t totalColTiles) {
  if (colsPerDstPass <= 0 || totalColTiles <= 0) {
    return op->emitOpError(
        "cols_per_dst_pass and total_col_tiles must both be positive");
  }
  if (totalColTiles % colsPerDstPass != 0) {
    return op->emitOpError("cols_per_dst_pass must divide total_col_tiles");
  }
  return success();
}

::mlir::LogicalResult TilizeInitOp::verify() {
  if (!insideKernelFunction(getOperation())) {
    return emitOpError("TilizeInitOp must be inside a kernel function");
  }
  std::string err =
      verifyTilizeUntilizeCBs(getCbOut().getType(), getCbIn().getType());
  if (!err.empty()) {
    return emitOpError(err);
  }
  return success();
}

::mlir::LogicalResult UntilizeInitOp::verify() {
  if (!insideKernelFunction(getOperation())) {
    return emitOpError("UntilizeInitOp must be inside a kernel function");
  }
  auto inputCBType = getCbIn().getType();
  if (!mlir::isa<ttcore::TileType>(inputCBType.getElementType())) {
    return emitOpError("Input to UntilizeInitOp must have tile element type");
  }
  return success();
}

::mlir::LogicalResult TilizeBlockOp::verify() {
  if (!insideKernelFunction(getOperation())) {
    return emitOpError("TilizeBlockOp must be inside a kernel function");
  }
  std::string err =
      verifyTilizeUntilizeCBs(getCbOut().getType(), getCbIn().getType());
  if (!err.empty()) {
    return emitOpError(err);
  }
  return success();
}

::mlir::LogicalResult ExperimentalTilizeBlockOp::verify() {
  if (!insideKernelFunction(getOperation())) {
    return emitOpError(
        "ExperimentalTilizeBlockOp must be inside a kernel function");
  }
  std::string err =
      verifyTilizeUntilizeCBs(getCbOut().getType(), getCbIn().getType());
  if (!err.empty()) {
    return emitOpError(err);
  }
  return success();
}

::mlir::LogicalResult UntilizeBlockOp::verify() {
  if (!insideKernelFunction(getOperation())) {
    return emitOpError("UntilizeBlockOp must be inside a kernel function");
  }
  std::string err =
      verifyTilizeUntilizeCBs(getCbIn().getType(), getCbOut().getType());
  if (!err.empty()) {
    return emitOpError(err);
  }
  return success();
}

::mlir::LogicalResult ExperimentalUntilizeBlockOp::verify() {
  if (!insideKernelFunction(getOperation())) {
    return emitOpError(
        "ExperimentalUntilizeBlockOp must be inside a kernel function");
  }
  std::string err =
      verifyTilizeUntilizeCBs(getCbIn().getType(), getCbOut().getType());
  if (!err.empty()) {
    return emitOpError(err);
  }
  return success();
}

::mlir::LogicalResult PackUntilizeInitOp::verify() {
  if (!insideKernelFunction(getOperation())) {
    return emitOpError("PackUntilizeInitOp must be inside a kernel function");
  }
  std::string err =
      verifyTilizeUntilizeCBs(getCbIn().getType(), getCbOut().getType());
  if (!err.empty()) {
    return emitOpError(err);
  }
  if (failed(verifyPackUntilizeDims(getOperation(), getColsPerDstPass(),
                                    getTotalColTiles()))) {
    return failure();
  }
  return success();
}

::mlir::LogicalResult ExperimentalPackUntilizeBlockOp::verify() {
  if (!insideKernelFunction(getOperation())) {
    return emitOpError(
        "ExperimentalPackUntilizeBlockOp must be inside a kernel function");
  }
  std::string err =
      verifyTilizeUntilizeCBs(getCbIn().getType(), getCbOut().getType());
  if (!err.empty()) {
    return emitOpError(err);
  }
  if (failed(verifyPackUntilizeDims(getOperation(), getColsPerDstPass(),
                                    getTotalColTiles()))) {
    return failure();
  }

  // block_c is an SSA operand. If it is constant here, validate the additional
  // compatibility requirement used by the LLK implementation.
  if (auto blockC = getConstantIntValue(getBlockC())) {
    int64_t blockColTiles = *blockC;
    if (blockColTiles <= 0) {
      return emitOpError("block_c must be positive");
    }
    if (blockColTiles % static_cast<int64_t>(getColsPerDstPass()) != 0) {
      return emitOpError("block_c must be divisible by cols_per_dst_pass");
    }
  }
  return success();
}

::mlir::LogicalResult TransposeInitOp::verify() {
  if (!insideKernelFunction(getOperation())) {
    return emitOpError("TransposeInitOp must be inside a kernel function");
  }

  // Both input and output should have tile element types for transpose.
  auto inputCBType = getCbIn().getType();
  auto outputCBType = getCbOut().getType();

  if (!mlir::isa<ttcore::TileType>(inputCBType.getElementType())) {
    return emitOpError("Input to TransposeInitOp must have tile element type");
  }

  if (!mlir::isa<ttcore::TileType>(outputCBType.getElementType())) {
    return emitOpError("Output to TransposeInitOp must have tile element type");
  }

  return success();
}

::mlir::LogicalResult TransposeTileOp::verify() {
  if (!insideKernelFunction(getOperation())) {
    return emitOpError("TransposeWHTileOp must be inside a kernel function");
  }

  // Only need to check the input CB since this is a single-tile operation
  // The output is implicit (DST register)
  auto inputCBType = getIcb().getType();

  if (!mlir::isa<ttcore::TileType>(inputCBType.getElementType())) {
    return emitOpError(
        "Input to TransposeWHTileOp must have tile element type");
  }

  return success();
}

// The D2M pipeline today only exercises sfpu_reduce for signed int32
// reductions (the float path goes through reduce_tile). Restrict these ops to
// Int32 until we grow coverage for other data formats.
static bool isSFPUReduceTypeSupported(ttkernel::ReduceType rt) {
  return rt == ttkernel::ReduceType::Sum || rt == ttkernel::ReduceType::Max;
}
static bool isSFPUReduceDataFormatSupported(ttcore::DataType dt) {
  return dt == ttcore::DataType::Int32;
}

::mlir::LogicalResult SFPUReduceInitOp::verify() {
  if (!isSFPUReduceTypeSupported(getReduceType())) {
    return emitOpError("sfpu_reduce only supports reduce_type Sum or Max");
  }
  if (!isSFPUReduceDataFormatSupported(getDataFormat())) {
    return emitOpError("sfpu_reduce only supports data_format Int32");
  }
  return success();
}

::mlir::LogicalResult SFPUReduceTileOp::verify() {
  if (!isSFPUReduceTypeSupported(getReduceType())) {
    return emitOpError("sfpu_reduce only supports reduce_type Sum or Max");
  }
  if (!isSFPUReduceDataFormatSupported(getDataFormat())) {
    return emitOpError("sfpu_reduce only supports data_format Int32");
  }
  // The sfpu_reduce kernel only performs intra-tile reductions along a single
  // dim (Row or Col). Scalar reductions must be decomposed into Col + Row by
  // the caller.
  if (getReduceDim() == ttkernel::ReduceDim::Scalar) {
    return emitOpError("sfpu_reduce does not support reduce_dim Scalar; "
                       "decompose into Col + Row");
  }
  return success();
}

::mlir::LogicalResult DPrintOp::verify() {
  StringRef fmt = getFmt();
  size_t numFormatSpecifiers = fmt.count("{}");

  if (numFormatSpecifiers != getOperands().size()) {
    return emitOpError("number of format specifiers must match number of "
                       "operands");
  }
  return success();
}
static ::mlir::LogicalResult verifyNocAsyncAddressMode(Operation *op,
                                                       OperandRange core,
                                                       OperandRange bankId) {
  if (core.empty() == bankId.empty()) {
    return op->emitOpError("must specify exactly one NoC address mode");
  }
  if (!core.empty() && core.size() != 2) {
    return op->emitOpError("core address mode requires x and y coordinates");
  }
  if (!bankId.empty() && bankId.size() != 1) {
    return op->emitOpError("bank address mode requires one bank id");
  }
  return success();
}

::mlir::LogicalResult NocAsyncReadOp::verify() {
  return verifyNocAsyncAddressMode(getOperation(), getSrcCoreXY(),
                                   getSrcBankId());
}

::mlir::LogicalResult NocAsyncWriteOp::verify() {
  return verifyNocAsyncAddressMode(getOperation(), getDstCoreXY(),
                                   getDstBankId());
}

::mlir::LogicalResult NocAsyncReadOnePacketSetStateOp::verify() {
  return verifyNocAsyncAddressMode(getOperation(), getSrcCoreXY(),
                                   getSrcBankId());
}

::mlir::LogicalResult NocAsyncReadOnePacketWithStateOp::verify() {
  return verifyNocAsyncAddressMode(getOperation(), getSrcCoreXY(),
                                   getSrcBankId());
}

::mlir::LogicalResult NocAsyncWriteOnePacketWithTridOp::verify() {
  return verifyNocAsyncAddressMode(getOperation(), getDstCoreXY(),
                                   getDstBankId());
}

using ConditionAssignment = std::pair<Value, bool>;

// Return the branch conditions that must hold for `operation` to execute.
static SmallVector<ConditionAssignment>
getEnclosingConditions(Operation *operation) {
  SmallVector<ConditionAssignment> conditions;
  for (Operation *ancestor = operation->getParentOp(); ancestor;
       ancestor = ancestor->getParentOp()) {
    auto ifOp = dyn_cast<scf::IfOp>(ancestor);
    if (!ifOp) {
      continue;
    }
    Region *operationRegion = operation->getParentRegion();
    bool executesInThenRegion =
        ifOp.getThenRegion().isAncestor(operationRegion);
    bool executesInElseRegion =
        ifOp.getElseRegion().isAncestor(operationRegion);
    assert(executesInThenRegion != executesInElseRegion &&
           "operation nested in scf.if must belong to exactly one branch");
    conditions.emplace_back(ifOp.getCondition(), executesInThenRegion);
  }
  return conditions;
}

// Return whether every execution of `use` also executes `setup`'s guards.
static bool useExecutionImpliesSetupExecution(Operation *setup,
                                              Operation *use) {
  SmallVector<ConditionAssignment> setupConditions =
      getEnclosingConditions(setup);
  SmallVector<ConditionAssignment> useConditions = getEnclosingConditions(use);
  return llvm::all_of(setupConditions, [&](ConditionAssignment setupCondition) {
    return llvm::is_contained(useConditions, setupCondition);
  });
}

// Return false only when a shared branch condition proves mutual exclusion.
static bool executionsMayOverlap(Operation *lhs, Operation *rhs) {
  SmallVector<ConditionAssignment> lhsConditions = getEnclosingConditions(lhs);
  SmallVector<ConditionAssignment> rhsConditions = getEnclosingConditions(rhs);
  return llvm::none_of(lhsConditions, [&](ConditionAssignment lhsCondition) {
    return llvm::is_contained(
        rhsConditions,
        ConditionAssignment{lhsCondition.first, !lhsCondition.second});
  });
}

// Exclude setup operations in loops that do not also contain the state use.
static bool setupLoopExecutionsReachUse(Operation *setup, Operation *use) {
  for (Operation *ancestor = setup->getParentOp(); ancestor;
       ancestor = ancestor->getParentOp()) {
    if (isa<LoopLikeOpInterface>(ancestor) && !ancestor->isAncestor(use)) {
      return false;
    }
  }
  return true;
}

// Compare execution order after projecting nested operations into a common
// enclosing block.
static bool structurallyPrecedes(Operation *before, Operation *after) {
  for (Operation *ancestor = before; ancestor;
       ancestor = ancestor->getParentOp()) {
    if (ancestor->isProperAncestor(after)) {
      return false;
    }
    if (Operation *projectedAfter =
            ancestor->getBlock()->findAncestorOpInBlock(*after)) {
      return ancestor->isBeforeInBlock(projectedAfter);
    }
  }
  return false;
}

// Prove selector equality from SSA identity, equal constants, or two omitted
// selectors that both select the default NoC.
static bool haveProvablySameNocSelector(Value lhs, Value rhs) {
  if (!lhs || !rhs) {
    return !lhs && !rhs;
  }
  if (lhs == rhs) {
    return true;
  }
  std::optional<int64_t> lhsConstant = getConstantIntValue(lhs);
  std::optional<int64_t> rhsConstant = getConstantIntValue(rhs);
  return lhsConstant && rhsConstant && *lhsConstant == *rhsConstant;
}

// Return false only when two explicit constant selectors are distinct.
static bool nocSelectorsMayAlias(Value lhs, Value rhs) {
  if (lhs == rhs) {
    return true;
  }
  if (!lhs || !rhs) {
    return true;
  }
  std::optional<int64_t> lhsConstant = getConstantIntValue(lhs);
  std::optional<int64_t> rhsConstant = getConstantIntValue(rhs);
  return !lhsConstant || !rhsConstant || *lhsConstant == *rhsConstant;
}

// Find the last state setup proven to execute before every execution of `use`
// on the same NoC.
static NocAsyncWriteOnePacketSetStateOp
findReachingWriteStateSetup(NocAsyncWriteOnePacketWithStateOp use) {
  func::FuncOp function = use->getParentOfType<func::FuncOp>();
  if (!function) {
    return {};
  }

  NocAsyncWriteOnePacketSetStateOp reachingSetup;
  function.walk([&](NocAsyncWriteOnePacketSetStateOp setup) {
    if (!haveProvablySameNocSelector(setup.getNoc(), use.getNoc()) ||
        !setupLoopExecutionsReachUse(setup, use) ||
        !useExecutionImpliesSetupExecution(setup, use) ||
        !structurallyPrecedes(setup, use)) {
      return;
    }
    if (!reachingSetup || structurallyPrecedes(reachingSetup, setup)) {
      reachingSetup = setup;
    }
  });
  return reachingSetup;
}

// Return a state setup that may replace `reachingSetup` on only a subset of the
// executions that reach `use`.
static NocAsyncWriteOnePacketSetStateOp
findInterveningWriteStateSetup(NocAsyncWriteOnePacketSetStateOp reachingSetup,
                               NocAsyncWriteOnePacketWithStateOp use) {
  func::FuncOp function = use->getParentOfType<func::FuncOp>();
  NocAsyncWriteOnePacketSetStateOp interveningSetup;
  function.walk([&](NocAsyncWriteOnePacketSetStateOp setup) {
    if (setup == reachingSetup ||
        !nocSelectorsMayAlias(setup.getNoc(), use.getNoc()) ||
        !executionsMayOverlap(setup, use)) {
      return;
    }

    bool precedesUse = structurallyPrecedes(setup, use);
    bool followsReachingSetup = structurallyPrecedes(reachingSetup, setup);
    if (precedesUse && followsReachingSetup) {
      interveningSetup = setup;
      return;
    }

    // A setup after the use becomes the reaching state on the next iteration
    // unless that loop also contains the selected setup.
    if (!structurallyPrecedes(use, setup)) {
      return;
    }
    for (Operation *ancestor = use->getParentOp(); ancestor;
         ancestor = ancestor->getParentOp()) {
      if (!isa<LoopLikeOpInterface>(ancestor) ||
          ancestor->isAncestor(reachingSetup)) {
        continue;
      }
      if (ancestor->isAncestor(setup)) {
        interveningSetup = setup;
        return;
      }
    }
  });
  return interveningSetup;
}

::mlir::LogicalResult NocAsyncWriteOnePacketWithStateOp::verify() {
  NocAsyncWriteOnePacketSetStateOp setup = findReachingWriteStateSetup(*this);
  if (!setup) {
    return emitOpError(
        "requires a preceding one-packet write state setup on the same NoC "
        "whose execution conditions cover this operation");
  }
  if (NocAsyncWriteOnePacketSetStateOp interveningSetup =
          findInterveningWriteStateSetup(setup, *this)) {
    InFlightDiagnostic diagnostic = emitOpError(
        "cannot identify one preceding write state setup for every execution");
    diagnostic.attachNote(interveningSetup.getLoc())
        << "this setup may replace the selected state before a later issue";
    return failure();
  }
  bool setupIsPosted = setup.getPosted().value_or(false);
  bool useIsPosted = getPosted().value_or(false);
  if (setupIsPosted != useIsPosted) {
    return emitOpError(
        "posted mode must match the preceding one-packet write state setup");
  }
  return success();
}

::mlir::LogicalResult TensorAccessorArgsOp::verify() {
  // Validation rules:
  // 1. If prev_args is present, cta_base and crta_base should NOT be present.
  // 2. If prev_args is NOT present, both cta_base and crta_base MUST be present
  //    and must be constants (unless expr attrs are provided).

  if (getPrevArgs()) {
    // When chaining, we shouldn't have cta_base/crta_base
    if (getCtaBase() || getCrtaBase()) {
      return emitOpError(
          "cta_base and crta_base should not be provided when using prev_args");
    }
  } else {
    // When not chaining, both cta_base and crta_base are required.
    if (!getCtaBase() || !getCrtaBase()) {
      return emitOpError(
          "both cta_base and crta_base are required when prev_args is not "
          "provided");
    }

    // If no expr attribute, the base must be a constant.
    if (!getCtaExprAttr()) {
      if (!getCtaBase().getDefiningOp<arith::ConstantOp>()) {
        return emitOpError(
            "cta_base must be a constant when cta_expr is not provided");
      }
    }

    if (!getCrtaExprAttr()) {
      if (!getCrtaBase().getDefiningOp<arith::ConstantOp>()) {
        return emitOpError(
            "crta_base must be a constant when crta_expr is not provided");
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TensorAccessorArgsOp custom assembly format
//===----------------------------------------------------------------------===//
// Format:
// - Without prev_args: TensorAccessorArgs(%cta, %crta) [cta_expr = "..."]
//                      [crta_expr = "..."] {attr-dict}
// - With prev_args:    TensorAccessorArgs(prev = %prev) [cta_expr = "..."]
//                      [crta_expr = "..."] {attr-dict}

void TensorAccessorArgsOp::print(::mlir::OpAsmPrinter &p) {
  p << "(";
  if (getPrevArgs()) {
    p << "prev = " << getPrevArgs();
  } else {
    p << getCtaBase() << ", " << getCrtaBase();
  }
  p << ")";

  if (getCtaExprAttr()) {
    p << " cta_expr = " << getCtaExprAttr();
  }
  if (getCrtaExprAttr()) {
    p << " crta_expr = " << getCrtaExprAttr();
  }

  llvm::SmallVector<::llvm::StringRef, 3> elidedAttrs = {
      "cta_expr", "crta_expr", "operandSegmentSizes"};
  p.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

::mlir::ParseResult
TensorAccessorArgsOp::parse(::mlir::OpAsmParser &parser,
                            ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand ctaBaseOperand;
  ::mlir::OpAsmParser::UnresolvedOperand crtaBaseOperand;
  ::mlir::OpAsmParser::UnresolvedOperand prevArgsOperand;
  bool hasPrevArgs = false;

  auto i32Type = parser.getBuilder().getI32Type();
  auto tensorAccessorArgsType =
      TensorAccessorArgsType::get(parser.getContext());

  if (parser.parseLParen()) {
    return failure();
  }

  if (succeeded(parser.parseOptionalKeyword("prev"))) {
    if (parser.parseEqual() || parser.parseOperand(prevArgsOperand)) {
      return failure();
    }
    hasPrevArgs = true;
  } else {
    // Parse cta_base, crta_base.
    if (parser.parseOperand(ctaBaseOperand) || parser.parseComma() ||
        parser.parseOperand(crtaBaseOperand)) {
      return failure();
    }
  }

  if (parser.parseRParen()) {
    return failure();
  }

  StringAttr ctaExprAttr;
  if (succeeded(parser.parseOptionalKeyword("cta_expr"))) {
    if (parser.parseEqual() || parser.parseAttribute(ctaExprAttr)) {
      return failure();
    }
    result.addAttribute("cta_expr", ctaExprAttr);
  }

  StringAttr crtaExprAttr;
  if (succeeded(parser.parseOptionalKeyword("crta_expr"))) {
    if (parser.parseEqual() || parser.parseAttribute(crtaExprAttr)) {
      return failure();
    }
    result.addAttribute("crta_expr", crtaExprAttr);
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // Resolve operands and build operandSegmentSizes
  // Arguments order: cta_base (optional), crta_base (optional), prev_args
  // (optional). When prev_args is present, cta_base and crta_base are not
  // provided.
  int32_t ctaBaseCount = 0;
  int32_t crtaBaseCount = 0;
  int32_t prevArgsCount = 0;

  if (hasPrevArgs) {
    // Only prev_args operand
    if (parser.resolveOperand(prevArgsOperand, tensorAccessorArgsType,
                              result.operands)) {
      return failure();
    }
    prevArgsCount = 1;
  } else {
    // cta_base and crta_base operands
    if (parser.resolveOperand(ctaBaseOperand, i32Type, result.operands) ||
        parser.resolveOperand(crtaBaseOperand, i32Type, result.operands)) {
      return failure();
    }
    ctaBaseCount = 1;
    crtaBaseCount = 1;
  }

  // Add operandSegmentSizes attribute (required by AttrSizedOperandSegments)
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {ctaBaseCount, crtaBaseCount, prevArgsCount}));

  // Add result type
  result.addTypes(tensorAccessorArgsType);

  return success();
}

static mlir::ConstantIntRanges getIndexRange(uint64_t umin, uint64_t umax) {
  unsigned width = mlir::IndexType::kInternalStorageBitWidth;
  return mlir::ConstantIntRanges::fromUnsigned(mlir::APInt(width, umin),
                                               mlir::APInt(width, umax));
}

void MyLogicalXOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges,
    mlir::SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 getIndexRange(0, std::numeric_limits<uint32_t>::max()));
}

void MyLogicalYOp::inferResultRanges(
    ::llvm::ArrayRef<::mlir::ConstantIntRanges> argRanges,
    mlir::SetIntRangeFn setResultRange) {
  setResultRange(getResult(),
                 getIndexRange(0, std::numeric_limits<uint32_t>::max()));
}

static FailureOr<int64_t> lookupConstantTableValue(int64_t index,
                                                   ArrayRef<int64_t> values) {
  if (index < 0 || static_cast<std::size_t>(index) >= values.size()) {
    return failure();
  }
  return values[index];
}

void ConstantTableLookupOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *) {
  patterns.add(+[](ConstantTableLookupOp lookupOp,
                   PatternRewriter &rewriter) -> LogicalResult {
    APInt indexValue;
    if (!matchPattern(lookupOp.getIndex(), m_ConstantInt(&indexValue))) {
      return rewriter.notifyMatchFailure(lookupOp, "index is not constant");
    }

    FailureOr<int64_t> tableValue = lookupConstantTableValue(
        indexValue.getSExtValue(), lookupOp.getValues());
    if (failed(tableValue)) {
      return rewriter.notifyMatchFailure(lookupOp,
                                         "index is outside table bounds");
    }

    rewriter.replaceOpWithNewOp<arith::ConstantIndexOp>(lookupOp, *tableValue);
    return success();
  });
}

LogicalResult ConstantTableLookupOp::verify() {
  ArrayRef<int64_t> values = getValues();
  if (values.empty()) {
    return emitOpError("requires at least one table value");
  }
  if (llvm::any_of(values, [](int64_t value) { return value < 0; })) {
    return emitOpError("requires non-negative table values");
  }
  APInt indexValue;
  if (matchPattern(getIndex(), m_ConstantInt(&indexValue))) {
    int64_t index = indexValue.getSExtValue();
    if (index < 0 || static_cast<std::size_t>(index) >= values.size()) {
      return emitOpError() << "constant index " << index
                           << " is outside the table bounds [0, "
                           << values.size() << ")";
    }
  }
  return success();
}

void NocAsyncReadBarrierOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add(+[](NocAsyncReadBarrierOp op,
                   mlir::PatternRewriter &rewriter) -> LogicalResult {
    for (Operation *it = op->getPrevNode(); it != nullptr;
         it = it->getPrevNode()) {
      if (mlir::isa<NocAsyncReadBarrierOp>(it)) {
        auto previousBarrier = mlir::cast<NocAsyncReadBarrierOp>(it);
        if (previousBarrier.getNoc() == op.getNoc()) {
          rewriter.eraseOp(op);
          return success();
        }
      }
      if (mlir::isa<NocAsyncReadOp, NocAsyncReadTileOp,
                    NocAsyncReadOnePacketSetStateOp,
                    NocAsyncReadOnePacketWithStateOp>(it) ||
          it->getNumRegions() > 0) {
        break;
      }
    }
    return failure();
  });
}

void NocAsyncWriteBarrierOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add(+[](NocAsyncWriteBarrierOp op,
                   mlir::PatternRewriter &rewriter) -> LogicalResult {
    for (Operation *it = op->getPrevNode(); it != nullptr;
         it = it->getPrevNode()) {
      if (mlir::isa<NocAsyncWriteBarrierOp>(it)) {
        auto previousBarrier = mlir::cast<NocAsyncWriteBarrierOp>(it);
        if (previousBarrier.getNoc() == op.getNoc()) {
          rewriter.eraseOp(op);
          return success();
        }
      }
      bool issuesOrConfiguresWrite =
          accessesNocCommand(it, NocCommandClass::Write) &&
          !mlir::isa<NocAsyncWriteBarrierOp, NocAsyncWritesFlushedOp>(it);
      if (issuesOrConfiguresWrite || it->getNumRegions() > 0) {
        break;
      }
    }
    return failure();
  });
}

void UnpackStallOnPackOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add(+[](UnpackStallOnPackOp op,
                   mlir::PatternRewriter &rewriter) -> mlir::LogicalResult {
    if (!mlir::isa_and_nonnull<UnpackStallOnPackOp>(op->getPrevNode())) {
      return mlir::failure();
    }

    rewriter.eraseOp(op);
    return mlir::success();
  });
}

::mlir::LogicalResult OpaqueCallOp::verify() {
  if (failed(mlir::tt::utils::verifyOpaqueCallNames(getOperation(), getCallee(),
                                                    getHeader()))) {
    return failure();
  }
  if (failed(mlir::tt::utils::verifyOpaqueCallUnsignedArgIndices(
          getOperation(), getUnsignedArgIndices(), getArgOperands()))) {
    return failure();
  }
  // Absence represents no descriptor requirement. A canonical nonnegative set
  // lets downstream annotation merge physical DFB indices without normalizing.
  if (std::optional<ArrayRef<int32_t>> requiredPhysicalDFBIndices =
          getDfbResourceIndices()) {
    if (requiredPhysicalDFBIndices->empty()) {
      return emitOpError("DFB resource indices must not be empty");
    }
    if (!llvm::all_of(*requiredPhysicalDFBIndices,
                      [](int32_t index) { return index >= 0; })) {
      return emitOpError("DFB resource indices must be nonnegative");
    }
    if (!mlir::tt::utils::areIndicesStrictlyIncreasing(
            *requiredPhysicalDFBIndices)) {
      return emitOpError(
          "DFB resource indices must be strictly increasing without "
          "duplicates");
    }
  }
  std::optional<ArrayAttr> templateArgs = getTemplateArgs();
  if (!templateArgs) {
    return success();
  }
  for (Attribute templateArg : *templateArgs) {
    if (isa<BoolAttr, DFBDescriptorAttr>(templateArg)) {
      continue;
    }
    auto integerArg = dyn_cast<IntegerAttr>(templateArg);
    if (!integerArg) {
      return emitOpError("template arg must be a signed i32, boolean, "
                         "unsigned i32, or DFB descriptor attribute");
    }
    auto integerType = dyn_cast<IntegerType>(integerArg.getType());
    if (!integerType || integerType.getWidth() != 32 ||
        (!integerType.isSigned() && !integerType.isUnsigned())) {
      return emitOpError("integer template arg must have type si32 or ui32");
    }
  }
  return success();
}

} // namespace mlir::tt::ttkernel
