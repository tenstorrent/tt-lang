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

/// Additive tensor recurrence over one loop-carried tensor value.
///
/// The match records both the loop-carried SSA recurrence and the final store
/// that gives the recurrence a dataflow buffer destination.
struct TensorAccumulationMatch {
  unsigned resultIndex;
  RankedTensorType tensorType;
  Value initialValue;
  StoreOp finalStore;
  CBReserveOp reserve;
  AddOp add;
  Value contribution;
  SmallVector<AttachCBOp> deadReserveAttachOps;
};

/// Recurrence properties required for DST-resident tensor accumulation.
struct TensorDstAccumulationInfo {
  /// Static trip count of the contributing loop.
  int64_t tripCount;
  /// Number of tiles in one contribution tensor.
  int64_t unitTileCount;
  /// Number of contribution tiles consumed across all loop iterations.
  int64_t totalContributionTiles;
  /// Dataflow-buffer wait that produces the per-iteration contribution.
  CBWaitOp contributionWait;
  /// Tensor view produced by `contributionWait`, if present.
  AttachCBOp attachedContribution;
  /// Ranked tensor type of one per-iteration contribution.
  RankedTensorType contributionType;
};

/// Recurrence properties required for packer L1 tensor accumulation.
struct TensorL1PackAccumulationInfo {
  /// Static trip count of the contributing loop, when known.
  std::optional<int64_t> tripCount;
  /// Number of tiles in one contribution tensor, when statically known.
  std::optional<int64_t> unitTileCount;
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
/// This is required after scope insertion, where `ttl.accumulation_scope` owns
/// the same reservation as an operand.
///
/// `allowedLoopResultUsers` contains wrapper terminators that may consume the
/// loop result without defining an externally visible destination.
///
/// Expected non-matches return failure without diagnostics; callers decide
/// whether another strategy or the general loop-state lowering should handle
/// the loop.
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

/// Return DST-resident accumulation properties for `match` when legal.
FailureOr<TensorDstAccumulationInfo>
analyzeTensorAccumulationForDst(TensorAccumulationMatch &match,
                                scf::ForOp loop);

/// Return packer L1 accumulation properties for `match` when legal.
FailureOr<TensorL1PackAccumulationInfo>
analyzeTensorAccumulationForL1Pack(TensorAccumulationMatch &match,
                                   scf::ForOp loop);

/// Lower a matched additive tensor recurrence to one reduction compute whose
/// DST acquisition spans all reduction iterations.
LogicalResult lowerTensorAccumulationToDst(TensorAccumulationMatch &match,
                                           scf::ForOp loop,
                                           RewriterBase &rewriter);

/// Lower a matched additive tensor recurrence to one initial output store plus
/// per-iteration accumulating stores. The generated loop is annotated for L1
/// packer accumulation reconfiguration insertion.
LogicalResult lowerTensorAccumulationToL1Pack(TensorAccumulationMatch &match,
                                              scf::ForOp loop, int64_t scopeId,
                                              RewriterBase &rewriter);

/// Return one more than the maximum L1 accumulation scope id under `root`.
///
/// Callers use this when introducing new annotated loops into IR that may
/// already contain accumulation scopes from earlier lowering.
int64_t getNextL1AccScopeId(Operation *root);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_ACCUMULATIONUTILS_H
