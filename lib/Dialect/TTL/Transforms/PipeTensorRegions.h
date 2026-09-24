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

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
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

/// Tile-grid box within one global tensor.
struct TensorRegionBounds {
  int64_t globalTensorIndex = 0;
  SmallVector<int64_t> tensorGridShape;
  SmallVector<int64_t> startIndices;
  SmallVector<int64_t> extents;
};

/// Return false only when the boxes are proven disjoint. Boxes in grids of
/// different shapes are treated as overlapping. The global tensor index is not
/// compared.
bool tensorRegionsOverlap(const TensorRegionBounds &lhs,
                          const TensorRegionBounds &rhs);

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

  TensorRegionBounds getBounds(ArrayRef<int64_t> occurrenceStart) const;
};

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
/// and no other destination may alias it.
SmallVector<bool> computeDisjointTensorRegionDestinations(
    ArrayRef<TensorRegionOccurrences> destinations);

/// Supplies values of one enumerated occurrence that launch-location
/// evaluation cannot determine. `inductionValues` binds the enumerated
/// `scf.for` induction variables of that occurrence and is empty while the
/// enclosing loops and their bounds are resolved. A callback may set
/// `failureReason` to explain a value it cannot resolve.
using TensorSliceOccurrenceValueEvaluator =
    llvm::function_ref<std::optional<llvm::APInt>(
        Value value, const llvm::DenseMap<Value, llvm::APInt> &inductionValues,
        std::string &failureReason)>;

/// Return the start indices of `slice` for each of its
/// `expectedExecutionCount` executions at `location`, in execution order.
/// Enclosing `scf.for` loops whose induction variables are not evaluable are
/// enumerated, and enclosing `scf.if` conditions select the executing
/// occurrences. A slice outside every enumerated loop repeats its single start.
/// Fails when a loop bound, condition, or start index cannot be evaluated,
/// when the enumeration count differs from `expectedExecutionCount`, or when an
/// occurrence leaves the tensor tile grid. The failure is reported through
/// `emitError` when it is provided.
FailureOr<SmallVector<SmallVector<int64_t>>> enumerateTensorSliceOccurrences(
    TensorSliceOp slice, const LaunchExecutionLocation &location,
    const LaunchNodeDomainState &state, std::uint64_t expectedExecutionCount,
    TensorSliceOccurrenceValueEvaluator evaluateContextValue,
    llvm::function_ref<InFlightDiagnostic()> emitError);

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
