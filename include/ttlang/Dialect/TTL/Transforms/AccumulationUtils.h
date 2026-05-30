// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_ACCUMULATIONUTILS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_ACCUMULATIONUTILS_H

#include "ttlang/Dialect/TTL/IR/TTLOps.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
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

/// Placement constraint for the output reservation associated with a matched
/// tensor accumulation.
enum class TensorAccumulationReservePlacement {
  /// The output reservation and the loop must be siblings. This is the
  /// pre-scope form produced by ordinary loop-to-store IR.
  SameBlock,
  /// The output reservation may be captured by an enclosing operation. This is
  /// the scoped form after the reservation has become an accumulation-scope
  /// operand.
  ExternalAllowed,
};

/// Return true when the loop result at `resultIndex` carries ranked tensor
/// state that must be removed before compute lowering.
bool isTensorLoopState(scf::ForOp loop, unsigned resultIndex);

/// Match `acc = add(acc, contribution)` where the loop result is consumed by
/// one non-accumulating store to a dataflow buffer reservation.
///
/// `allowedReserveUsers` contains wrapper operations that are permitted to use
/// the output reservation in addition to the final store and dead attach views.
/// This is required after scope formation, where `ttl.accumulation_scope` owns
/// the same reservation as an operand.
///
/// Expected non-matches return failure without diagnostics; callers decide
/// whether another strategy or the general loop-state lowering should handle
/// the loop.
FailureOr<TensorAccumulationMatch> matchAdditiveTensorAccumulation(
    scf::ForOp loop, unsigned resultIndex,
    TensorAccumulationReservePlacement reservePlacement =
        TensorAccumulationReservePlacement::SameBlock,
    ArrayRef<Operation *> allowedReserveUsers = {});

/// Lower a matched additive tensor recurrence to one reduction compute whose
/// DST acquisition spans all reduction iterations.
LogicalResult lowerTensorAccumulationToDst(TensorAccumulationMatch &match,
                                           scf::ForOp loop,
                                           RewriterBase &rewriter);

/// Lower a matched additive tensor recurrence to one initial output store plus
/// per-iteration accumulating stores. The generated loop is annotated for L1
/// packer accumulation guard insertion.
LogicalResult lowerTensorAccumulationToL1Pack(TensorAccumulationMatch &match,
                                              scf::ForOp loop,
                                              RewriterBase &rewriter);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_ACCUMULATIONUTILS_H
