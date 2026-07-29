// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/Transforms/ControlFlowUtils.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::tt::ttl {

bool arePairwiseInsideMutuallyExclusiveRegions(ArrayRef<Operation *> ops) {
  // TODO(#685): Add predicate-based proof for analyzable sibling `scf.if`
  // chains that upstream structural region analysis cannot prove.
  for (auto [lhsIndex, lhsOp] : llvm::enumerate(ops)) {
    for (Operation *rhsOp : ops.drop_front(lhsIndex + 1)) {
      if (!insideMutuallyExclusiveRegions(lhsOp, rhsOp)) {
        return false;
      }
    }
  }
  return true;
}

} // namespace mlir::tt::ttl
