// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeCounter.h"

#include "ttlang/Dialect/TTL/IR/TTL.h"

#include <algorithm>
#include <cassert>

namespace mlir::tt::ttl {

PipeCounterInfo PipeCounterInfo::localSemaphore(int64_t semaphoreIndex) {
  return PipeCounterInfo(PipeCounterStorage::LocalSemaphore, semaphoreIndex);
}

PipeCounterInfo PipeCounterInfo::globalSemaphore(int64_t globalSemaphoreIndex) {
  return PipeCounterInfo(PipeCounterStorage::GlobalSemaphore,
                         globalSemaphoreIndex);
}

PipeCounterInfo PipeCounterAllocator::allocate() {
  assert(counts.localSemaphoreCount <= kMaxHardwareSemaphoreIds &&
         "pipe counter allocator has an invalid local semaphore count");
  if (counts.localSemaphoreCount < kMaxHardwareSemaphoreIds) {
    return PipeCounterInfo::localSemaphore(counts.localSemaphoreCount++);
  }
  return allocateGlobal();
}

PipeCounterInfo PipeCounterAllocator::allocateGlobal() {
  return PipeCounterInfo::globalSemaphore(counts.globalSemaphoreCount++);
}

void PipeCounterAllocationCounts::include(PipeCounterInfo counter) {
  int64_t requiredCount = counter.getIndex() + 1;
  if (counter.getStorage() == PipeCounterStorage::LocalSemaphore) {
    localSemaphoreCount = std::max(localSemaphoreCount, requiredCount);
    return;
  }
  globalSemaphoreCount = std::max(globalSemaphoreCount, requiredCount);
}

} // namespace mlir::tt::ttl
