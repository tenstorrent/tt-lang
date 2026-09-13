// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "SRAMAllocationReport.h"

#include "DFBConcurrentKernelLivenessAnalysis.h"
#include "DFBPhysicalAllocationPlan.h"
#include "SRAMAllocationPlan.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <utility>

namespace mlir::tt::ttl {
namespace {

static llvm::json::Value sourceLocation(Operation *operation) {
  if (!operation) {
    return nullptr;
  }
  std::string text;
  llvm::raw_string_ostream stream(text);
  operation->getLoc().print(stream);
  return text;
}

static llvm::json::Array eventIds(ArrayRef<unsigned> events) {
  llvm::json::Array result;
  for (unsigned event : events) {
    result.push_back(event);
  }
  return result;
}

static llvm::json::Array nodeLifetimes(ArrayRef<DFBPerNodeLifetime> lifetimes) {
  llvm::json::Array result;
  for (const auto &lifetime : lifetimes) {
    result.push_back(llvm::json::Object{
        {"core", llvm::json::Array{lifetime.node.x, lifetime.node.y}},
        {"active", lifetime.mayBeActive},
        {"completion_proven", lifetime.completionProof.proven()},
        {"entry_events", eventIds(lifetime.earliestEntryEvents)},
        {"completion_events", eventIds(lifetime.terminalCompletionEvents)},
        {"entry_location", sourceLocation(lifetime.entryEvidence)}});
  }
  return result;
}

} // namespace

void printSRAMAllocationReport(
    llvm::raw_ostream &output, const SRAMAllocationPlan &plan,
    const DFBConcurrentKernelLivenessAnalysis &liveness,
    const DFBPhysicalConflictModel &conflicts, llvm::StringRef strategy,
    bool reuseEnabled, uint64_t alignmentBytes, uint64_t controlBytes,
    uint64_t budgetBytes) {
  llvm::json::Array owners;
  llvm::SmallVector<std::pair<uint64_t, uint64_t>> intervals;
  uint64_t extentSum = 0;
  for (auto [ownerIndex, storage] : llvm::enumerate(plan.storage)) {
    llvm::json::Array members;
    for (unsigned member : storage.members) {
      members.push_back(plan.regions[member].logicalId);
    }
    llvm::json::Object owner{{"owner", ownerIndex},
                             {"logical_dfbs", std::move(members)},
                             {"control_offset", storage.stateOffset},
                             {"control_bytes", kSRAMControlRecordBytes},
                             {"arena_payload_bytes", storage.allocationBytes}};
    if (storage.allocationBytes != 0) {
      owner["arena_payload_offset"] = storage.offset;
      intervals.emplace_back(storage.offset,
                             storage.offset + storage.allocationBytes);
      extentSum += storage.allocationBytes;
    }
    owners.push_back(std::move(owner));
  }
  llvm::sort(intervals);
  uint64_t unionBytes = 0;
  uint64_t previousEnd = controlBytes;
  for (auto [begin, end] : intervals) {
    unionBytes += end > previousEnd ? end - std::max(begin, previousEnd) : 0;
    previousEnd = std::max(previousEnd, end);
  }

  llvm::json::Array regions;
  for (const SRAMRegion &region : plan.regions) {
    llvm::json::Object entry{
        {"logical_dfb", region.logicalId},
        {"owner", region.storageIndex},
        {"declaration", sourceLocation(region.declarations.front())},
        {"domain_known", region.launchDomain.known}};
    llvm::json::Array cores;
    for (auto node : region.launchDomain.nodes) {
      cores.push_back(llvm::json::Array{node.x, node.y});
    }
    entry["cores"] = std::move(cores);
    if (region.tensorBacking) {
      entry["tensor"] = llvm::json::Object{
          {"index", region.tensorBacking.getTensorIndex()},
          {"byte_offset", region.tensorBacking.getByteOffset()},
          {"bytes", region.tensorBacking.getByteSize()}};
    }
    regions.push_back(std::move(entry));
  }

  llvm::json::Array reusedRanges;
  for (unsigned left = 0; left < plan.storage.size(); ++left) {
    const auto &lhs = plan.storage[left];
    if (lhs.allocationBytes == 0) {
      continue;
    }
    for (unsigned right = left + 1; right < plan.storage.size(); ++right) {
      const auto &rhs = plan.storage[right];
      if (rhs.allocationBytes == 0) {
        continue;
      }
      uint64_t begin = std::max(lhs.offset, rhs.offset);
      uint64_t end = std::min(lhs.offset + lhs.allocationBytes,
                              rhs.offset + rhs.allocationBytes);
      if (begin < end) {
        reusedRanges.push_back(
            llvm::json::Object{{"owners", llvm::json::Array{left, right}},
                               {"offset", begin},
                               {"bytes", end - begin}});
      }
    }
  }

  llvm::json::Array evidence;
  for (const DFBConflictEvidence &conflict : conflicts.getEvidence()) {
    llvm::json::Object entry{
        {"logical_dfbs",
         llvm::json::Array{conflict.lhsLogicalId, conflict.rhsLogicalId}},
        {"reason", getDFBConflictReasonName(conflict.reason)},
        {"left_location", sourceLocation(conflict.lhsOperation)},
        {"right_location", sourceLocation(conflict.rhsOperation)}};
    if (conflict.node) {
      entry["core"] = llvm::json::Array{conflict.node->x, conflict.node->y};
    }
    evidence.push_back(std::move(entry));
  }
  llvm::json::Array lifetimes;
  for (const auto &lifecycle : liveness.getLogicalDFBLifecycles()) {
    lifetimes.push_back(llvm::json::Object{
        {"logical_dfb", lifecycle.logicalId},
        {"known_cores", nodeLifetimes(lifecycle.nodeLifetimes)},
        {"possible_cores", nodeLifetimes(lifecycle.possibleNodeLifetimes)}});
  }
  llvm::json::Object report{
      {"schema_version", 1},
      {"phase", "compiler"},
      {"strategy", strategy},
      {"reuse_enabled", reuseEnabled},
      {"alignment_bytes", alignmentBytes},
      {"budget_bytes_per_core", budgetBytes},
      {"arena_bytes_per_core", plan.arenaBytes},
      {"control_and_padding_bytes", controlBytes},
      {"control_record_bytes", plan.storage.size() * kSRAMControlRecordBytes},
      {"control_padding_bytes",
       controlBytes - plan.storage.size() * kSRAMControlRecordBytes},
      {"payload_high_water_bytes", plan.arenaBytes - controlBytes},
      {"payload_extent_sum_bytes", extentSum},
      {"payload_union_bytes", unionBytes},
      {"payload_gap_bytes", plan.arenaBytes - controlBytes - unionBytes},
      {"payload_reuse_bytes", extentSum - unionBytes},
      {"owners", std::move(owners)},
      {"regions", std::move(regions)},
      {"reused_ranges", std::move(reusedRanges)},
      {"logical_conflicts", std::move(evidence)},
      {"lifetimes", std::move(lifetimes)}};
  if (!plan.coreLayouts.empty()) {
    report["allocation_mode"] = "per-core";
    report["domain"] = plan.coreLayouts.front().domain;
    llvm::json::Array cores;
    for (const SRAMCoreLayout &layout : plan.coreLayouts) {
      cores.push_back(llvm::json::Array{layout.node.x, layout.node.y});
    }
    report["cores"] = std::move(cores);
  }
  output << "ttlang-sram-report: ";
  llvm::json::OStream json(output);
  json.value(std::move(report));
  output << "\n";
}
} // namespace mlir::tt::ttl
