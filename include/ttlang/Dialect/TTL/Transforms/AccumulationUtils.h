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

#include <optional>

namespace mlir::tt::ttl {

/// Connected operations that define one loop-carried additive tensor
/// recurrence and its final dataflow buffer store.
struct TensorAccumulationMatch {
  /// Loop result number that carries the accumulator.
  unsigned resultIndex;

  /// Tensor type shared by the accumulator, contribution, and result.
  RankedTensorType tensorType;

  /// Initial value passed to the loop-carried accumulator.
  Value initialValue;

  /// Store that publishes the loop result to the output dataflow buffer.
  StoreOp finalStore;

  /// Reservation consumed by `finalStore`.
  CBReserveOp reserve;

  /// In-loop add whose result is yielded as the next accumulator value.
  AddOp add;

  /// Non-accumulator add operand for the current loop iteration.
  Value contribution;

  /// Unused attachment ops on the reserved output tensor. These are safe to
  /// erase when normalizing the final store placement.
  SmallVector<AttachCBOp> deadReserveAttachOps;
};

/// Properties that must remain stable between the precondition scan and the
/// rewrite to a DST-resident reduction compute.
struct TensorDstAccumulationInfo {
  /// Number of source loop iterations folded into the reduction dimension.
  int64_t tripCount;

  /// Number of tiles in one per-iteration contribution tensor.
  int64_t unitTileCount;

  /// Total number of contribution tiles consumed by the coalesced wait.
  int64_t totalContributionTiles;

  /// Loop-local wait that provides the per-iteration contribution tensor.
  CBWaitOp contributionWait;

  /// Optional attachment between `contributionWait` and the add operand.
  AttachCBOp attachedContribution;

  /// Tensor type returned by `contributionWait`.
  RankedTensorType contributionType;
};

/// Placement constraint for the output reservation associated with a matched
/// tensor accumulation.
enum class TensorAccumulationReservePlacement {
  /// The reservation must be in the same block as the source loop.
  SameBlock,

  /// The reservation may be outside the source loop block, as it is after the
  /// loop has been wrapped in an accumulation scope.
  ExternalAllowed,
};

/// Match `acc = add(acc, contribution)` where the loop result is consumed by
/// one non-accumulating store to a dataflow buffer reservation. The final store
/// is non-accumulating because the loop-carried add already encodes the
/// accumulation semantics.
FailureOr<TensorAccumulationMatch> matchAdditiveTensorAccumulation(
    scf::ForOp loop, unsigned resultIndex,
    TensorAccumulationReservePlacement reservePlacement =
        TensorAccumulationReservePlacement::SameBlock,
    ArrayRef<Operation *> allowedReserveUsers = {},
    ArrayRef<Operation *> allowedLoopResultUsers = {});

/// Return the static trip count, accepting constant bounds that have been cast
/// to index. scf::ForOp::getStaticTripCount does not fold arith.index_cast.
std::optional<int64_t> getStaticAccumulationTripCount(scf::ForOp loop);

/// Return the number of tiles represented by a statically ranked tensor.
FailureOr<int64_t> getStaticTensorTileCount(RankedTensorType tensorType);

/// Return DST-resident accumulation properties for `match` when the source
/// loop can be deleted without dropping side effects.
FailureOr<TensorDstAccumulationInfo>
analyzeTensorAccumulationForDst(TensorAccumulationMatch &match,
                                scf::ForOp loop);

/// Lower a matched additive tensor recurrence to one reduction compute whose
/// DST acquisition spans all reduction iterations. Callers must either run the
/// same analysis before mutation or handle failure without leaving partial IR.
LogicalResult lowerTensorAccumulationToDst(TensorAccumulationMatch &match,
                                           scf::ForOp loop,
                                           RewriterBase &rewriter);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_ACCUMULATIONUTILS_H
