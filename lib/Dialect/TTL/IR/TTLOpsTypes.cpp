// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"

#define GET_TYPEDEF_CLASSES
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.cpp.inc"

namespace mlir::tt::ttl {

void TTLDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.cpp.inc"
      >();
}

mlir::Type TTLDialect::parseType(mlir::DialectAsmParser &parser) const {
  // No types defined yet; update when TTL types are introduced.
  parser.emitError(parser.getNameLoc(), "TTL dialect has no types yet");
  return {};
}

void TTLDialect::printType(mlir::Type type,
                           mlir::DialectAsmPrinter &printer) const {
  // No types defined yet; update when TTL types are introduced.
}

} // namespace mlir::tt::ttl
