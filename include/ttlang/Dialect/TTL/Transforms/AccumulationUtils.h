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

struct DFBAcquireReleaseIndex;

/// Describes how the contribution operand is acquired relative to the source
/// recurrence loop.
enum class TensorAccumulationContributionResidency {
  /// Each loop iteration acquires and releases one contribution block.
  Streamed,

  /// One contribution block is acquired before the loop and reused by every
  /// iteration.
  Resident,
};

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
/// rewrite to a streaming DST-resident recurrence.
struct TensorDstAccumulationInfo {
  /// Number of accumulator tiles resident for the whole DST section.
  int64_t unitTileCount;

  /// DFB-backed tensor copied into the DST accumulator before the source loop.
  Value initialValue;

  /// Whether the contribution is acquired per iteration or held across the
  /// recurrence loop.
  TensorAccumulationContributionResidency contributionResidency;

  /// Wait that provides the contribution tensor.
  CBWaitOp contributionWait;

  /// Optional attachment between `contributionWait` and the add operand.
  AttachCBOp attachedContribution;

  /// Tensor type returned by `contributionWait`.
  RankedTensorType contributionType;

  /// Existing owned resident release, when one is already present.
  CBPopOp residentContributionPop;
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

/// Return the number of tiles represented by a statically ranked tensor.
FailureOr<int64_t> getStaticTensorTileCount(RankedTensorType tensorType);

/// Return DST-resident accumulation properties for `match` when the source
/// loop can be deleted without dropping side effects.
FailureOr<TensorDstAccumulationInfo> analyzeTensorAccumulationForDst(
    const TensorAccumulationMatch &match, scf::ForOp loop,
    const DFBAcquireReleaseIndex *dfbIndex = nullptr);

/// Return DST-resident accumulation properties using `initialValue` as the
/// tensor copied into the accumulator before the source loop executes.
FailureOr<TensorDstAccumulationInfo> analyzeTensorAccumulationForDst(
    const TensorAccumulationMatch &match, scf::ForOp loop, Value initialValue,
    const DFBAcquireReleaseIndex *dfbIndex = nullptr);

/// Lower a matched additive tensor recurrence to a streaming DST section whose
/// DST acquisition spans the original source loop.
LogicalResult
lowerTensorAccumulationToDst(const TensorAccumulationMatch &match,
                             const TensorDstAccumulationInfo &info,
                             scf::ForOp loop, RewriterBase &rewriter);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_ACCUMULATIONUTILS_H
