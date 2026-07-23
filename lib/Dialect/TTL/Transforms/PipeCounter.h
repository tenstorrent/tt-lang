// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPECOUNTER_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPECOUNTER_H

#include <cstdint>

namespace mlir::tt::ttl {

/// Storage selected for a compiler-managed PipeNet synchronization counter.
enum class PipeCounterStorage {
  LocalSemaphore,
  GlobalSemaphore,
};

/// Identifies one PipeNet counter and its allocated storage.
class PipeCounterInfo {
public:
  /// Construct a counter backed by a TTKernel local semaphore id.
  static PipeCounterInfo localSemaphore(int64_t semaphoreIndex);

  /// Construct a counter backed by a host-created GlobalSemaphore.
  static PipeCounterInfo globalSemaphore(int64_t globalSemaphoreIndex);

  PipeCounterStorage getStorage() const { return storage; }
  int64_t getIndex() const { return index; }

  bool operator==(const PipeCounterInfo &other) const {
    return storage == other.storage && index == other.index;
  }

private:
  PipeCounterInfo(PipeCounterStorage storage, int64_t index)
      : storage(storage), index(index) {}

  PipeCounterStorage storage;
  int64_t index;
};

/// Counts allocated counter slots in each storage class.
struct PipeCounterAllocationCounts {
  int64_t localSemaphoreCount = 0;
  int64_t globalSemaphoreCount = 0;

  /// Include one allocated counter in these resource totals.
  void include(PipeCounterInfo counter);
};

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPECOUNTER_H
