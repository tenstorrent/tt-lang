// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTL.h"

#include "mlir/IR/DialectImplementation.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
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
