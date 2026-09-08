// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_SRAMALLOCATOR_INTERNAL_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_SRAMALLOCATOR_INTERNAL_H

#include "SRAMAllocator.h"

namespace mlir::tt::ttl::detail {

enum class GreedyGapSelection { FirstFit, BestFit };

/// Requires a validated problem; the result may exceed its budget and serve as
/// an upper bound for exact search before common result validation.
FailureOr<SRAMAllocationSolution>
allocateGreedy(const SRAMAllocationProblem &problem,
               GreedyGapSelection selection, std::string &failureReason);

std::unique_ptr<SRAMAllocator> createFirstFitDecreasingSRAMAllocator();
std::unique_ptr<SRAMAllocator> createBestFitDecreasingSRAMAllocator();
std::unique_ptr<SRAMAllocator> createMultiOrderDecreasingSRAMAllocator();
std::unique_ptr<SRAMAllocator>
createExactSRAMAllocator(uint64_t searchWorkLimit);

} // namespace mlir::tt::ttl::detail

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_SRAMALLOCATOR_INTERNAL_H
