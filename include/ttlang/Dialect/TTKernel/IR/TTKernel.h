// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTKERNEL_IR_TTKERNEL_H
#define TTLANG_DIALECT_TTKERNEL_IR_TTKERNEL_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

#include "ttlang/Dialect/TTKernel/IR/TTKernelOpsDialect.h.inc"

namespace mlir::tt::ttkernel {

enum class NocCommandClass;

/// Effects on one resident command class, independent of the selected NoC.
struct NocCommandEffects {
  bool mayReprogram = false;
  bool mayUseState = false;
};

/// Summarize command effects through calls on immutable IR. Unknown and
/// recursive callees conservatively both use and replace resident state.
class NocCommandEffectsAnalysis {
public:
  explicit NocCommandEffectsAnalysis(NocCommandClass commandClass)
      : commandClass(commandClass) {}

  /// Return the effects of `operation`, including callees but excluding its
  /// nested regions. Callers walking regions therefore visit each op once.
  NocCommandEffects getEffects(Operation *operation);

private:
  NocCommandClass commandClass;
  llvm::DenseMap<Operation *, NocCommandEffects> callableEffects;
};

/// Core ranges on which an internally lowered control-flow region executes.
constexpr llvm::StringLiteral
    kExecutionCoreRangesAttrName("ttkernel.execution_core_ranges");

/// Return whether enclosing `ttkernel.execution_core_ranges` attributes prove
/// that two operations execute on disjoint worker cores. A nonnull `limit`
/// must be a common ancestor and is excluded from the inspected metadata.
bool haveDisjointExecutionCoreRanges(Operation *lhs, Operation *rhs,
                                     Operation *limit = nullptr);

} // namespace mlir::tt::ttkernel

#endif
