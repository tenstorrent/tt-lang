// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_ACCUMULATIONUTILS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_ACCUMULATIONUTILS_H

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"

namespace mlir::tt::ttl {

/// Additive tensor recurrence matched before tensor loop-state materialization.
///
/// The match records both the loop-carried SSA recurrence and the final store
/// that gives the recurrence a dataflow buffer destination. Consumers may
/// choose a concrete accumulation strategy without rediscovering those links.
struct TensorAccumulationMatch {
  unsigned resultIndex;
  RankedTensorType tensorType;
  Value initialValue;
  BlockArgument iterArg;
  Value yieldedValue;
  StoreOp finalStore;
  CBReserveOp reserve;
  AddOp add;
  Value contribution;
  SmallVector<AttachCBOp> deadReserveAttachOps;
};

/// Return true when the loop result at `resultIndex` carries ranked tensor
/// state that must be removed before compute lowering.
bool isTensorLoopState(scf::ForOp loop, unsigned resultIndex);

/// Match `acc = add(acc, contribution)` where the loop result is consumed by
/// one non-accumulating store to a dataflow buffer reservation.
///
/// Expected non-matches return failure without diagnostics; callers decide
/// whether another strategy or the general loop-state lowering should handle
/// the loop.
FailureOr<TensorAccumulationMatch>
matchAdditiveTensorAccumulation(scf::ForOp loop, unsigned resultIndex);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_ACCUMULATIONUTILS_H
