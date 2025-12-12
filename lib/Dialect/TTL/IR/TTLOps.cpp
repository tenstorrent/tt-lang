// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "ttmlir/Dialect/TTNN/IR/TTNNOps.h"
#include "llvm/ADT/TypeSwitch.h"

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

} // namespace mlir::tt::ttl

mlir::LogicalResult mlir::tt::ttl::DmKernelOp::verify() {
  auto fnType = getFunctionType();
  for (Type argTy : fnType.getInputs()) {
    if (auto t = mlir::dyn_cast<RankedTensorType>(argTy)) {
      auto enc = t.getEncoding();
      if (!enc || !mlir::isa<tt::ttnn::TTNNLayoutAttr>(enc)) {
        return emitOpError()
               << "expects tensor arguments to carry TTNNLayout encoding";
      }
      continue;
    }
    if (mlir::isa<IntegerType>(argTy) || mlir::isa<FloatType>(argTy)) {
      continue;
    }
    return emitOpError() << "unsupported argument type " << argTy;
  }
  for (Type resTy : fnType.getResults()) {
    if (mlir::isa<IntegerType>(resTy) || mlir::isa<FloatType>(resTy) ||
        mlir::isa<RankedTensorType>(resTy)) {
      continue;
    }
    return emitOpError() << "unsupported result type " << resTy;
  }
  return success();
}

mlir::ParseResult mlir::tt::ttl::DmKernelOp::parse(OpAsmParser &parser,
                                                   OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void mlir::tt::ttl::DmKernelOp::print(OpAsmPrinter &p) {
  function_interface_impl::printFunctionOp(
      p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}
