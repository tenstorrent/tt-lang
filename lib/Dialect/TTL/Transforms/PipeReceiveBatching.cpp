// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeReceiveBatching.h"

#include "PipeGraph.h"
#include "PipeLowering.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/CheckedArithmetic.h"

namespace mlir::tt::ttl {

void annotateInitialPipeReceiveBatches(
    ModuleOp module, const PipeForeachLoweringInfo &foreachInfo,
    const PipeGraph &graph, const PipeResourcePlan &resources) {
  DenseMap<int64_t, SmallVector<CBReserveOp>> reservesByIndex;
  bool hasUnmodeledState = false;
  // Count all physical-index reservations, including aliases in other kernels.
  // External calls and reconfiguration require a separate initial-state proof.
  module.walk([&](Operation *operation) {
    if (isa<OpaqueCallOp, func::CallOp, ResetDFBsOp, ResetAllDFBsOp,
            DFBReconfigurationOp>(operation)) {
      hasUnmodeledState = true;
    }
    if (auto reserve = dyn_cast<CBReserveOp>(operation)) {
      auto bind = reserve.getCb().getDefiningOp<BindCBOp>();
      if (!bind) {
        hasUnmodeledState = true;
        return;
      }
      reservesByIndex[bind.getCbIndex().getSExtValue()].push_back(reserve);
    }
  });
  if (hasUnmodeledState) {
    return;
  }

  SmallVector<std::pair<scf::ForOp, int64_t>> batches;
  for (const auto &[operation, recordInfo] : foreachInfo.recordLoops) {
    auto loop = dyn_cast<scf::ForOp>(operation);
    if (!loop || !loop->hasAttr(kPipeNetLocalRecordLoopAttrName)) {
      continue;
    }
    auto function = loop->getParentOfType<func::FuncOp>();
    if (!function || !function.getBody().hasOneBlock()) {
      continue;
    }
    // A conditional executes at most once; an enclosing loop could leave
    // unread payload from an earlier invocation of this receive sequence.
    bool executesAtMostOnce = true;
    for (Operation *parent = loop->getParentOp(); parent != function;
         parent = parent->getParentOp()) {
      if (!isa<scf::IfOp>(parent)) {
        executesAtMostOnce = false;
        break;
      }
    }
    if (!executesAtMostOnce) {
      continue;
    }
    SmallVector<PipeTransferPostOp> posts;
    for (Operation &nested : loop.getBody()->without_terminator()) {
      if (auto post = dyn_cast<PipeTransferPostOp>(nested)) {
        posts.push_back(post);
      }
    }
    if (posts.size() != 1) {
      continue;
    }
    PipeTransferPostOp post = posts.front();
    auto resourceIt = resources.selectedResources.find(post);
    if (resourceIt == resources.selectedResources.end() ||
        resourceIt->second.empty() ||
        !llvm::all_of(resourceIt->second, [](const PipeResourceInfo &resource) {
          return resource.readyCounter &&
                 resource.addressStorage.usesComputedReceiverDFB();
        })) {
      continue;
    }
    int64_t physicalIndex =
        resourceIt->second.front()
            .addressStorage.computedAddress->receiverDFBIndex;
    if (!llvm::all_of(resourceIt->second,
                      [&](const PipeResourceInfo &resource) {
                        return resource.addressStorage.computedAddress
                                   ->receiverDFBIndex == physicalIndex;
                      })) {
      continue;
    }
    auto reserves = reservesByIndex.find(physicalIndex);
    if (reserves == reservesByIndex.end() || reserves->second.size() != 1 ||
        reserves->second.front()->getBlock() != loop.getBody()) {
      continue;
    }
    auto dfbType =
        cast<CircularBufferType>(reserves->second.front().getCb().getType());
    std::optional<int64_t> capacity = llvm::checkedMul(
        dfbType.getBlockCount(), dfbType.getElementsPerBlock());
    if (!capacity || *capacity <= 0) {
      continue;
    }
    bool hasReceiver = false;
    bool valid = true;
    for (const PipeReceiverDFBNode &receiver : graph.getReceiverDFBNodes()) {
      if (receiver.receiverDFB.dfbIndex != physicalIndex) {
        continue;
      }
      hasReceiver = true;
      llvm::SmallSet<int64_t, 8> slots;
      valid &= receiver.hasProvenPipeOnlyProducerStream;
      for (PipeReceiverEndpointId endpointId : receiver.writerEndpoints) {
        const PipeReceiverEndpoint &endpoint =
            graph.getPipeReceiverEndpoint(endpointId);
        const auto &sequence = endpoint.addressSequence;
        if (endpoint.postOp != post || sequence.executionCount != 1 ||
            !sequence.recurrence ||
            graph.getPipeTransferNode(endpoint.transferNode).blockSpan != 1 ||
            !slots.insert(sequence.recurrence->initialSlot).second) {
          valid = false;
          break;
        }
      }
    }
    if (hasReceiver && valid) {
      batches.emplace_back(loop, *capacity);
    }
  }
  for (auto [loop, capacity] : batches) {
    loop->setAttr(
        kPipeNetInitialReceiveCapacityAttrName,
        IntegerAttr::get(IntegerType::get(module.getContext(), 64), capacity));
  }
}

} // namespace mlir::tt::ttl
