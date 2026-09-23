// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_SRAMALLOCATOR_LOCATION_INTERNAL_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_SRAMALLOCATOR_LOCATION_INTERNAL_H

#include "ttlang/Dialect/TTL/Transforms/SRAMAllocator.h"

namespace mlir::tt::ttl::detail {

struct LocationPlacementVariable {
  llvm::SmallVector<unsigned> regionIndices;
  std::optional<uint64_t> fixedOffset;
};

struct LocationPlacementState {
  llvm::SmallVector<uint64_t> offsets;
  llvm::SmallVector<unsigned> placedRegions;
  llvm::SmallVector<uint64_t> highWaterBytes;
};

llvm::SmallVector<LocationPlacementVariable>
buildLocationPlacementVariables(const SRAMLocationAllocationProblem &problem);

bool fitsLocationVariableAtOffset(const LocationPlacementVariable &variable,
                                  uint64_t offset,
                                  const SRAMLocationAllocationProblem &problem,
                                  const LocationPlacementState &state);

void placeLocationVariable(const LocationPlacementVariable &variable,
                           uint64_t offset,
                           const SRAMLocationAllocationProblem &problem,
                           LocationPlacementState &state);

llvm::SmallVector<unsigned>
getLocationVariableOrder(llvm::ArrayRef<LocationPlacementVariable> variables,
                         const SRAMLocationAllocationProblem &problem,
                         bool degreeAware);

LocationPlacementState
makeEmptyLocationState(const SRAMLocationAllocationProblem &problem);

uint64_t
getLocationReservationBytes(const SRAMLocationAllocationProblem &problem,
                            llvm::ArrayRef<uint64_t> highWaterBytes);

} // namespace mlir::tt::ttl::detail

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_SRAMALLOCATOR_LOCATION_INTERNAL_H
