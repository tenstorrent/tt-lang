// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTL.h"

#include "mlir/IR/DialectImplementation.h"
#include "ttlang/Dialect/TTCore/IR/TTCore.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "llvm/ADT/TypeSwitch.h"

#include "ttlang/Dialect/TTL/IR/TTLOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// TTL dialect
//===----------------------------------------------------------------------===//

void mlir::tt::ttl::TTLDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ttlang/Dialect/TTL/IR/TTLOps.cpp.inc"
      >();
  registerTypes();
  registerAttributes();
}

mlir::LogicalResult mlir::tt::ttl::TTLDialect::verifyOperationAttribute(
    mlir::Operation *operation, mlir::NamedAttribute attribute) {
  if (attribute.getName() != kSelectedComputePipelineScheduleAttrName) {
    return mlir::success();
  }
  if (!mlir::isa<ComputePipelineScheduleAttr>(attribute.getValue())) {
    return operation->emitError()
           << "attribute '" << attribute.getName().getValue()
           << "' must be a #ttl.compute_pipeline_schedule attribute";
  }
  return mlir::success();
}
