// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Pipe Tensor Regions
//===----------------------------------------------------------------------===//
//
// This file declares the enumeration and overlap rules for DRAM tensor regions
// written by pipe receives, and the receiver readiness rule derived from them.
// Schedule verification and pipe lowering share these rules so both select the
// same synchronization protocol.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPETENSORREGIONS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPETENSORREGIONS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "ttlang/Analysis/IntegerExpressionEvaluator.h"
#include "ttlang/Analysis/LoopIterationUtils.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::tt::ttl {

/// Return the global tensor index of the kernel-function argument sliced by
/// `slice`, or no value when the function has no runtime index for it.
std::optional<int64_t> getTensorSliceGlobalIndex(TensorSliceOp slice);

/// Tensor-region destination of one receive at one execution location.
struct TensorRegionOccurrences {
  int64_t globalTensorIndex = 0;
  /// Logical device that executes the receive; null when unknown.
  DeviceRefAttr device;
  ArrayRef<int64_t> tensorGridShape;
  ArrayRef<int64_t> sliceShape;
  /// Slice start of every receive execution, in execution order.
  ArrayRef<SmallVector<int64_t>> startIndices;

  /// Return the region extent in each tensor tile-grid dimension.
  SmallVector<int64_t> getExtents() const;
};

/// Return the extent in each tensor tile-grid dimension of a region of shape
/// `sliceShape`, whose rank may be lower than the tensor's.
SmallVector<int64_t> getTensorRegionExtents(ArrayRef<int64_t> tensorGridShape,
                                            ArrayRef<int64_t> sliceShape);

/// Evaluate the start index of every dimension of `slice`. On failure, set
/// `failedDimension`, when provided, to the first dimension whose start is not
/// a signed 64-bit value.
FailureOr<SmallVector<int64_t>>
evaluateTensorSliceStart(TensorSliceOp slice,
                         IntegerExpressionEvaluator &evaluator,
                         std::size_t *failedDimension = nullptr);

/// Call `visit` with the indices of each pair of overlapping occurrences of
/// `lhs` and `rhs` until it returns true, and return whether it did.
/// Occurrences of tensors with different tile-grid shapes are treated as
/// overlapping. The global tensor index and device are not compared.
///
/// Occurrences in one tile grid are indexed by cells as large as the largest
/// region in at most four dimensions, so each occurrence is compared only with
/// occurrences in at most 81 neighboring cells, plus the overlapping pairs
/// visited. Occurrence counts follow user loop trip counts, so overlap checks
/// must go through this index and must not compare occurrences pairwise.
bool forEachOverlappingOccurrencePair(
    const TensorRegionOccurrences &lhs, const TensorRegionOccurrences &rhs,
    llvm::function_ref<bool(std::size_t, std::size_t)> visit);

/// Return true when two occurrences of `region` overlap.
bool hasOverlappingTensorRegionOccurrences(
    const TensorRegionOccurrences &region);

/// Return true when an occurrence of `lhs` overlaps an occurrence of `rhs`.
/// The global tensor index and device are not compared.
bool tensorRegionOccurrencesOverlap(const TensorRegionOccurrences &lhs,
                                    const TensorRegionOccurrences &rhs);

/// Return true when two destinations may write a common tile: they target the
/// same global tensor, are not proven to execute on different devices, and an
/// occurrence of one overlaps an occurrence of the other.
bool tensorRegionDestinationsMayAlias(const TensorRegionOccurrences &lhs,
                                      const TensorRegionOccurrences &rhs);

/// For each destination, return whether its occurrences are pairwise disjoint
/// and no other destination may alias it. Uses the cell index of
/// `forEachOverlappingOccurrencePair` over the distinct starts of every
/// destination; schedule verification passes one destination per expanded
/// receive, so this must not compare destinations pairwise.
SmallVector<bool> computeDisjointTensorRegionDestinations(
    ArrayRef<TensorRegionOccurrences> destinations);

/// Supplies values of one enumerated occurrence that launch-location
/// evaluation cannot determine. `inductionValues` binds the enumerated
/// `scf.for` induction variables of that occurrence and is empty while the
/// enclosing loops and their bounds are resolved. A callback may set
/// `failureReason` to explain a value it cannot resolve.
using TensorSliceOccurrenceValueEvaluator =
    llvm::function_ref<std::optional<llvm::APInt>(
        Value value, const LoopInductionBindings &inductionValues,
        std::string &failureReason)>;

/// Start indices of a slice at each enumerated execution, in execution order,
/// with the induction values of `loops` (outermost first) at that execution.
struct TensorSliceOccurrences {
  SmallVector<scf::ForOp> loops;
  SmallVector<SmallVector<int64_t>> startIndices;
  SmallVector<SmallVector<int64_t>> inductionValues;
};

/// Largest number of loop iterations, and of executions, enumerated for one
/// tensor-slice receive.
constexpr std::uint64_t kMaxEnumeratedTensorSliceOccurrences = 1ULL << 20;

/// Return the start indices of `slice` for each of its
/// `expectedExecutionCount` executions at `location`, in execution order.
/// Enclosing `scf.for` loops whose induction variables are not evaluable are
/// enumerated, and enclosing `scf.if` conditions select the executing
/// occurrences. A slice outside every enumerated loop repeats its single start.
/// Fails when a loop bound, condition, or start index cannot be evaluated,
/// when the enumeration count differs from `expectedExecutionCount`, when an
/// occurrence leaves the tensor tile grid, or when the enclosing loops or
/// `expectedExecutionCount` exceed `kMaxEnumeratedTensorSliceOccurrences`. The
/// failure is reported through `emitError` when it is provided.
FailureOr<TensorSliceOccurrences> enumerateTensorSliceOccurrences(
    TensorSliceOp slice, const LaunchExecutionLocation &location,
    const LaunchNodeDomainState &state, std::uint64_t expectedExecutionCount,
    TensorSliceOccurrenceValueEvaluator evaluateContextValue,
    llvm::function_ref<InFlightDiagnostic()> emitError);

/// Return the start indices of `slice` for every iteration of the `scf.for`
/// loops enclosing `user` whose induction variables are not evaluable at
/// `location`, whether or not enclosing `scf.if` conditions execute `user`.
/// Fails when a loop bound or start index cannot be evaluated, a start leaves
/// the tensor tile grid, or the iteration count exceeds `maxIterations`.
FailureOr<TensorSliceOccurrences>
enumerateTensorSliceIterationStarts(TensorSliceOp slice, Operation *user,
                                    const LaunchExecutionLocation &location,
                                    const LaunchNodeDomainState &state,
                                    std::uint64_t maxIterations);

/// Return true when a transfer needs no receiver readiness signal: it is a
/// point-to-point fabric transfer to one receiver whose tensor-region
/// destination is disjoint as defined by
/// `computeDisjointTensorRegionDestinations`. Such a destination is never
/// reused during the invocation, so the sender does not wait for receiver
/// posts. Completion notification is still required.
bool canOmitFabricReceiverRendezvous(bool isDeviceTransfer, bool isPointToPoint,
                                     bool hasSingleReceiver,
                                     bool hasDisjointTensorRegionDestination);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPETENSORREGIONS_H
