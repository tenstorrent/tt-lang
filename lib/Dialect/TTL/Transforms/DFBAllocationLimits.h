// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_DFBALLOCATIONLIMITS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_DFBALLOCATIONLIMITS_H

#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsAttrs.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsTypes.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::tt::ttl {

class DFBLogicalIdentityAnalysis;

constexpr int64_t kDFBResetStateWordCount = 4;
constexpr int64_t kDFBResetStateBytes =
    kDFBResetStateWordCount * static_cast<int64_t>(sizeof(uint32_t));

/// Collects one reset declaration per ordinal in deterministic order.
LogicalResult
collectSynchronizedDFBResets(ModuleOp module,
                             SmallVectorImpl<SynchronizedDFBResetAttr> &resets);

/// Returns the payload bytes required by all reset synchronization records.
FailureOr<uint64_t> getSynchronizedDFBResetStateBytes(ModuleOp module);

/// Returns the runtime scratch allocation for all reset synchronization state.
FailureOr<uint64_t>
getSynchronizedDFBResetStateAllocationBytes(ModuleOp module);

/// Returns the per-node L1 bytes occupied by one physical DFB descriptor.
/// On failure, `failureReason` describes the invalid allocation type.
FailureOr<uint64_t> getDFBAllocationSizeBytes(CircularBufferType type,
                                              std::string &failureReason);

/// Rounds one runtime allocation to the target's maximum L1 quantum.
FailureOr<uint64_t> getL1AllocationSizeBytes(ModuleOp module,
                                             uint64_t payloadBytes);

/// Returns the target-aligned L1 allocation for one physical DFB descriptor.
FailureOr<uint64_t> getDFBL1AllocationSizeBytes(ModuleOp module,
                                                CircularBufferType type,
                                                std::string &failureReason);

/// Returns the logical per-node payload bytes for all unique reconfiguration
/// boundaries, before L1 allocator rounding, or failure when the total is not
/// representable.
FailureOr<uint64_t> getDFBReconfigurationStateBytes(ModuleOp module);

/// Returns the target-aligned L1 allocations for all reconfiguration records.
FailureOr<uint64_t> getDFBReconfigurationStateAllocationBytes(ModuleOp module);

/// Verifies that the selected target implements DFB reconfiguration.
LogicalResult validateDFBReconfigurationTarget(ModuleOp module);

/// Static storage capacity and page alignment shared by one or more DFBs.
struct DFBStorageLayout {
  uint64_t capacityBytes = 0;
  uint64_t alignmentBytes = 1;
};

/// Extends a shared storage layout for one DFB allocation. The capacity is
/// rounded to a multiple of every member's page size so each runtime format
/// descriptor can use the same backing allocation.
FailureOr<DFBStorageLayout>
mergeDFBStorageLayout(const DFBStorageLayout &layout, uint64_t memberBytes,
                      uint64_t memberPageSize, std::string &failureReason);

/// Per-node L1 footprint aggregated by compiler-selected storage index.
class DFBStorageFootprint {
public:
  /// Adds one DFB. On failure, `failureReason` describes the invalid type.
  LogicalResult add(int64_t storageIndex, CircularBufferType type,
                    std::string &failureReason);

  bool empty() const { return layoutByIndex.empty(); }
  /// Returns the sum after target allocation rounding is applied per storage.
  FailureOr<uint64_t> getL1AllocationBytes(ModuleOp module) const;
  uint64_t getBytes(int64_t storageIndex) const;
  llvm::SmallVector<int64_t> getSortedStorageIndices() const;

private:
  llvm::DenseMap<int64_t, DFBStorageLayout> layoutByIndex;
};

/// L1 footprint aggregated by unique physical DFB index.
class DFBAllocationFootprint {
public:
  /// Adds one assignment and returns true when it increases the index size.
  /// On failure, `failureReason` describes the invalid allocation type.
  FailureOr<bool> add(ModuleOp module, int64_t physicalIndex,
                      CircularBufferType type, std::string &failureReason);

  bool empty() const { return maxBytesByIndex.empty(); }
  /// Returns the total allocation size, or failure when the sum overflows.
  FailureOr<uint64_t> getTotalBytes() const;
  /// Returns the total after applying per-index minimum allocation sizes.
  FailureOr<uint64_t> getTotalBytesWithMinimumAllocations(
      const llvm::DenseMap<int64_t, uint64_t> &minimumBytesByIndex) const;
  uint64_t getBytes(int64_t physicalIndex) const;
  llvm::SmallVector<int64_t> getSortedPhysicalIndices() const;

private:
  llvm::DenseMap<int64_t, uint64_t> maxBytesByIndex;
};

/// Finalized static DFB storage, including launch-node residency metadata.
struct FinalizedDFBStorageFootprint {
  using MembersByStorageIndex =
      llvm::DenseMap<int64_t, llvm::SmallVector<int64_t>>;

  DFBStorageFootprint globalFootprint;
  MembersByStorageIndex globalMembers;
  llvm::SmallVector<LaunchNodeCoord> launchNodes;
  llvm::SmallVector<DFBStorageFootprint> footprintsByNode;
  llvm::SmallVector<MembersByStorageIndex> membersByNode;
  llvm::DenseMap<int64_t, int64_t> storageIndexByPhysicalIndex;
  bool usesPerNodeAccounting = false;

  /// Returns the maximum target-rounded storage allocation on any launch node.
  FailureOr<uint64_t> getPeakL1AllocationBytes(
      ModuleOp module,
      std::optional<LaunchNodeCoord> *peakNode = nullptr) const;
};

/// Reads finalized storage and node-residency metadata and aggregates all
/// non-tensor-backed DFB declarations.
FailureOr<FinalizedDFBStorageFootprint>
getFinalizedDFBStorageFootprint(ModuleOp module);

/// Returns a conservative footprint that assigns each logical DFB separate
/// storage. Tensor-backed declarations do not allocate additional L1.
FailureOr<DFBAllocationFootprint>
getLogicalDFBAllocationFootprint(ModuleOp module,
                                 const DFBLogicalIdentityAnalysis &identities);

/// Returns the per-core L1 allocation reserved for GlobalSemaphore objects.
FailureOr<uint64_t> getGlobalSemaphoreL1Bytes(ModuleOp module,
                                              int64_t semaphoreCount);

/// Validates finalized DFB storage plus hidden runtime allocations.
LogicalResult validateCombinedDFBResourceL1Bytes(
    ModuleOp module, uint64_t dfbBytes, uint64_t scratchBytes,
    int64_t globalSemaphoreCount,
    std::optional<uint64_t> overrideBytes = std::nullopt);

/// Verifies that the selected target implements synchronized DFB reset.
LogicalResult validateSynchronizedDFBResetTarget(ModuleOp module);

/// Returns the target's usable per-node L1 bytes or the supported fallback.
uint64_t
getUsableDFBL1Bytes(ModuleOp module,
                    std::optional<uint64_t> overrideBytes = std::nullopt);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_DFBALLOCATIONLIMITS_H
