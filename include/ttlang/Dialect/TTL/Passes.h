#ifndef TTLANG_DIALECT_TTL_PASSES_H
#define TTLANG_DIALECT_TTL_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "ttlang/Dialect/TTL/Passes.h.inc"

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_PASSES_H
