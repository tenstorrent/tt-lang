// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"

#define GET_OP_CLASSES
#include "ttlang/Dialect/TTL/IR/TTLOps.cpp.inc"

namespace mlir::tt::ttl {

void TTLDialect::registerAttributes() {
  // Attributes will be registered here
}

mlir::Attribute TTLDialect::parseAttribute(mlir::DialectAsmParser &parser,
                                           mlir::Type type) const {
  // No attributes defined yet, will be implemented when attributes are added.
  parser.emitError(parser.getNameLoc(), "TTL dialect has no attributes yet");
  return {};
}

void TTLDialect::printAttribute(mlir::Attribute attr,
                                mlir::DialectAsmPrinter &printer) const {
  // No attributes defined yet, will be implemented when attributes are added.
}

} // namespace mlir::tt::ttl
