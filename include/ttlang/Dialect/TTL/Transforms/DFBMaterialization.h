// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBMATERIALIZATION_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBMATERIALIZATION_H

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LLVM.h"

/// \file
/// Helpers for materializing tensor SSA values through compiler-managed
/// dataflow buffers.

namespace mlir::tt::ttl {

/// Allocates a fresh compiler-managed dataflow buffer and emits its `bind_cb`
/// at kernel entry, where finalization can assign physical indices
/// consistently. The provisional index is unique within the kernel;
/// finalization assigns module-wide indices, performs lifetime-based reuse, and
/// validates the selected target's physical DFB-index capacity. The completed
/// compiler pipeline emits a balanced reserve/push/wait/pop lifecycle whose
/// pop executes before the same static reserve repeats, so one slot is
/// sufficient. The builder's insertion point is left at the new `bind_cb`;
/// callers that need to emit elsewhere should wrap the call in
/// `OpBuilder::InsertionGuard`.
BindCBOp createCompilerAllocatedDFB(RankedTensorType tensorType, Location loc,
                                    func::FuncOp kernel, OpBuilder &builder);

/// Reserves the next slot of `dfb` and stores `tensor` into it. The caller must
/// publish the stored slot with a matching `cb_push`.
StoreOp createDFBStore(Value tensor, Value dfb, OpBuilder &builder);

/// Waits for one slot of `dfb` and exposes it as a tensor SSA value. The caller
/// must release the acquired slot with a matching `cb_pop` after its last use.
AttachCBOp createDFBWaitAndAttach(Value dfb, RankedTensorType tensorType,
                                  Location loc, OpBuilder &builder);

/// Routes a non-`ttl.compute` tensor value through a fresh compiler-allocated
/// DFB after `insertionAnchor`. The source must dominate the anchor, and the
/// returned attached value may serve every consumer that the anchor properly
/// dominates. Compute results are materialized atomically by
/// `TTLInsertIntermediateDFBs` so one producer compute is rebuilt at most once.
Value materializeToDFB(Value intermediate, Operation *insertionAnchor,
                       func::FuncOp kernel, OpBuilder &builder);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBMATERIALIZATION_H
