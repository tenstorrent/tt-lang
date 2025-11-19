#ifndef TTLANG_TRANSFORMS_PASSES_H
#define TTLANG_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::tt::ttlang {

std::unique_ptr<Pass> createTTLangAllocate();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "ttlang/Transforms/Passes.h.inc"

} // namespace mlir::tt::ttlang

#endif // TTLANG_TRANSFORMS_PASSES_H
