// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBMATERIALIZATION_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBMATERIALIZATION_H

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

/// \file
/// Helpers for materializing tensor SSA values through compiler-managed
/// dataflow buffers.

namespace mlir::tt::ttl {

struct DFBMaterializedValue {
  /// DFB-attached tensor value that should replace the consumer operand.
  Value materialized;

  /// Source tensor value used to key fallback materialization reuse.
  Value source;
};

/// Allocates a fresh compiler-managed dataflow buffer and emits its
/// `bind_cb` at function entry, where `finalize-dfb-indices` requires every
/// compiler-allocated bind to live. The assigned DFB index is provisional;
/// `finalize-dfb-indices` performs physical index reuse and validates the
/// hardware DFB-index limit. Compiler-managed intermediates use one slot: they
/// carry a single SSA value inside one compute thread, not a pipelined transfer
/// between independently scheduled threads. The builder's insertion point is
/// left at the new `bind_cb`; callers that need to emit elsewhere should wrap
/// the call in `OpBuilder::InsertionGuard`.
BindCBOp createCompilerAllocatedDFB(RankedTensorType tensorType, Location loc,
                                    func::FuncOp funcOp, ModuleOp moduleOp,
                                    OpBuilder &builder);

/// Pushes `tensor` into the next slot of `dfb` via `cb_reserve` + `store`.
StoreOp createDFBStore(Value tensor, Value dfb, OpBuilder &builder);

/// Consumes one slot of `dfb` as tensor SSA via `cb_wait` + `attach_cb`.
/// Callers use the returned attach's result wherever a tensor view of the slot
/// is needed.
AttachCBOp createDFBWaitAndAttach(Value dfb, RankedTensorType tensorType,
                                  Location loc, OpBuilder &builder);

/// Routes a non-`ttl.compute` tensor value through a fresh compiler-allocated
/// DFB for one consumer. Compute results are materialized atomically by
/// `TTLInsertIntermediateDFBs` so one producer compute is rebuilt at most once.
FailureOr<DFBMaterializedValue>
materializeToDFB(Value intermediate, ModuleOp moduleOp, OpBuilder &builder);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBMATERIALIZATION_H
