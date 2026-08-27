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

class DFBAcquireReleaseIndex;

/// Accumulation scope family selected by scope insertion and lowering passes.
enum class AccumulationScopeKind {
  Tensor,
  DFB,
};

/// Parse an accumulation scope kind pass option.
FailureOr<AccumulationScopeKind> parseAccumulationScopeKind(StringRef kind);

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
  /// Loop that carries the matched accumulator.
  scf::ForOp loop;

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
/// rewrite to a recurrence whose accumulator stays resident in DST.
struct TensorDstAccumulationInfo {
  /// Source loop trip count when statically known.
  std::optional<int64_t> tripCount;

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

  /// Last operation that uses the slot acquired by the resident wait.
  Operation *residentContributionLastUse;
};

/// Recurrence properties required for packer L1 tensor accumulation.
struct TensorL1PackAccumulationInfo {
  /// Static trip count of the contributing loop, when known.
  std::optional<int64_t> tripCount;

  /// Number of tiles in one contribution tensor, when statically known.
  std::optional<int64_t> unitTileCount;
};

/// Loop-carried recurrence value that would be read after its DFB release.
enum class TensorAccumulationReleasedValue {
  Initial,
  Contribution,
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

/// Return true when the loop result at `resultIndex` carries ranked tensor
/// state that must be removed before compute lowering.
bool isTensorLoopState(scf::ForOp loop, unsigned resultIndex);

/// Return the static trip count, accepting constant bounds that have been cast
/// to index. `scf::ForOp::getStaticTripCount` does not fold index casts.
std::optional<int64_t> getStaticAccumulationTripCount(scf::ForOp loop);

/// Return DST-resident accumulation properties for `match` when the source
/// loop can be deleted without dropping side effects.
FailureOr<TensorDstAccumulationInfo>
analyzeTensorAccumulationForDst(const TensorAccumulationMatch &match,
                                const DFBAcquireReleaseIndex &dfbIndex);

/// Return DST-resident accumulation properties using `initialValue` as the
/// tensor copied into the accumulator before the source loop executes.
FailureOr<TensorDstAccumulationInfo>
analyzeTensorAccumulationForDst(const TensorAccumulationMatch &match,
                                Value initialValue,
                                const DFBAcquireReleaseIndex &dfbIndex);

/// Return the first loop-carried recurrence operand whose acquired DFB slot has
/// an owned release before the lowering would read the value.
std::optional<TensorAccumulationReleasedValue>
getTensorAccumulationUseAfterOwnedRelease(
    const TensorAccumulationMatch &match,
    const DFBAcquireReleaseIndex &dfbIndex);

/// Return packer L1 accumulation properties for `match` when legal.
FailureOr<TensorL1PackAccumulationInfo> analyzeTensorAccumulationForL1Pack(
    const TensorAccumulationMatch &match,
    const DFBAcquireReleaseIndex *dfbIndex = nullptr);

/// Lower a matched additive tensor recurrence to a DST section whose
/// accumulator stays resident across the original source loop. Contribution
/// acquisition follows `info.contributionResidency`. The caller assigns
/// `synthesizeResidentContributionPop` to exactly one lowering plan for each
/// resident acquisition that has no existing release.
void lowerTensorAccumulationToDst(const TensorAccumulationMatch &match,
                                  const TensorDstAccumulationInfo &info,
                                  bool synthesizeResidentContributionPop,
                                  RewriterBase &rewriter);

/// Lower a matched additive tensor recurrence to one initial output store plus
/// per-iteration accumulating stores. The generated loop is annotated for L1
/// packer accumulation reconfiguration insertion.
LogicalResult lowerTensorAccumulationToL1Pack(
    const TensorAccumulationMatch &match, int64_t scopeId,
    const DFBAcquireReleaseIndex &dfbIndex, RewriterBase &rewriter);

/// Return one more than the maximum L1 accumulation scope id under `root`.
int64_t getNextL1AccScopeId(Operation *root);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_ACCUMULATIONUTILS_H
