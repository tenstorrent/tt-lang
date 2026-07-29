// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeCapacityAnalysis.h"

#include "DFBAcquireReleaseAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "ttl-pipe-capacity-analysis"

namespace mlir::tt::ttl {

void PipeCapacityEndpointFacts::print(llvm::raw_ostream &os) const {
  os << "src(" << releaseTarget.logicalX << ", " << releaseTarget.logicalY
     << ") -> ";
  printReceiverDFB(os, receiverDFB);
  os << " capacity " << initialCapacity;
}

bool PipeCapacityAnalysisResult::hasEndpointFacts(
    PipeReceiverEndpointId endpoint) const {
  return factsIndexByEndpoint.contains(endpoint);
}

const PipeCapacityEndpointFacts &PipeCapacityAnalysisResult::getEndpointFacts(
    PipeReceiverEndpointId endpoint) const {
  auto factsIt = factsIndexByEndpoint.find(endpoint);
  assert(factsIt != factsIndexByEndpoint.end() &&
         "receiver endpoint has no proven capacity facts");
  return endpointFacts[factsIt->second];
}

void PipeCapacityAnalysisResult::addEndpointFacts(
    PipeCapacityEndpointFacts facts) {
  auto [indexIt, inserted] =
      factsIndexByEndpoint.try_emplace(facts.endpoint, endpointFacts.size());
  (void)indexIt;
  assert(inserted && "receiver endpoint has duplicate capacity facts");
  endpointFacts.push_back(std::move(facts));
}

namespace {

static void
debugCandidateEndpoint(const PipeCapacityEndpointFacts &endpointFacts) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeCapacity: candidate ";
    endpointFacts.print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

static void debugRejectEndpoint(const PipeCapacityEndpointFacts &endpointFacts,
                                const llvm::Twine &reason) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeCapacity: reject ";
    endpointFacts.print(llvm::dbgs());
    llvm::dbgs() << ": " << reason << "\n";
  });
}

static void debugAcceptEndpoint(const PipeCapacityEndpointFacts &endpointFacts,
                                int64_t sendCount, int64_t popCount) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeCapacity: accept ";
    endpointFacts.print(llvm::dbgs());
    llvm::dbgs() << ": sends=" << sendCount << " pops=" << popCount << "\n";
  });
}

static bool isExactlyDomain(Operation *op, const LaunchNodeDomain &expected,
                            const PipeGraph &pipeGraph) {
  return pipeGraph.getOperationLaunchDomain(op) == expected;
}

static std::optional<int64_t> getDFBIndexFromView(Value view) {
  Value cb = getAttachedCB(view);
  if (!cb) {
    return std::nullopt;
  }
  return getCBIndex(cb);
}

static bool isReceiverDFBView(Value view,
                              const PipeReceiverDFBKey &receiverDFB) {
  std::optional<int64_t> maybeDFBIndex = getDFBIndexFromView(view);
  return maybeDFBIndex && *maybeDFBIndex == receiverDFB.dfbIndex;
}

static bool isMatchingTransfer(const PipeCapacityEndpointFacts &endpointFacts,
                               Operation *protocolOp,
                               const PipeGraph &pipeGraph) {
  const PipeTransferNode *transferNode =
      pipeGraph.getPipeTransferNodeForProtocolOp(protocolOp);
  return transferNode && transferNode->id == endpointFacts.transferNode;
}

static PipeCapacityEndpointFacts
getCapacityEndpointFacts(const PipeGraph &pipeGraph,
                         const PipeReceiverEndpoint &receiverEndpoint) {
  const PipeTransferNode &transferNode =
      pipeGraph.getPipeTransferNode(receiverEndpoint.transferNode);
  return PipeCapacityEndpointFacts{
      transferNode.id,
      receiverEndpoint.id,
      receiverEndpoint.receiverDFBNode,
      receiverEndpoint.receiverDFB,
      PipeCapacityReleaseTarget{transferNode.pipe.srcX, transferNode.pipe.srcY},
      receiverEndpoint.receiverDFBInfo.blockCount,
      {},
      {},
  };
}

static bool collectAndCheckPosts(ModuleOp mod,
                                 const PipeCapacityEndpointFacts &endpointFacts,
                                 const PipeGraph &pipeGraph) {
  LaunchNodeDomain receiverDomain =
      getSingleLaunchNodeDomain({endpointFacts.receiverDFB.receiver.x,
                                 endpointFacts.receiverDFB.receiver.y});
  // Posts establish endpoint validity but carry no facts needed by the caller.
  SmallVector<PipeTransferPostOp> posts;
  bool valid = true;
  mod.walk([&](PipeTransferPostOp postOp) {
    if (!isMatchingTransfer(endpointFacts, postOp.getOperation(), pipeGraph)) {
      return;
    }
    LaunchNodeDomain postDomain =
        pipeGraph.getOperationLaunchDomain(postOp.getOperation());
    if (!launchNodeDomainsOverlap(postDomain, receiverDomain)) {
      return;
    }
    if (!(postDomain == receiverDomain) || !isNocKernelThread(postOp)) {
      debugRejectEndpoint(endpointFacts,
                          "post is not in the receiver NOC domain");
      valid = false;
      return;
    }
    if (!isReceiverDFBView(postOp.getDst(), endpointFacts.receiverDFB)) {
      debugRejectEndpoint(endpointFacts,
                          "post destination is not the receiver DFB");
      valid = false;
      return;
    }
    posts.push_back(postOp);
  });
  if (valid && posts.empty()) {
    debugRejectEndpoint(endpointFacts, "no matching receiver posts");
    valid = false;
  }
  return valid;
}

static bool collectAndCheckSends(ModuleOp mod,
                                 const PipeCapacityEndpointFacts &endpointFacts,
                                 const PipeGraph &pipeGraph,
                                 SmallVectorImpl<PipeTransferSendOp> &sends) {
  const PipeCapacityReleaseTarget &target = endpointFacts.releaseTarget;
  LaunchNodeDomain sourceDomain =
      getSingleLaunchNodeDomain({target.logicalX, target.logicalY});
  bool valid = true;
  mod.walk([&](PipeTransferSendOp sendOp) {
    if (!isMatchingTransfer(endpointFacts, sendOp.getOperation(), pipeGraph)) {
      return;
    }
    if (!isExactlyDomain(sendOp, sourceDomain, pipeGraph) ||
        !isNocKernelThread(sendOp)) {
      debugRejectEndpoint(endpointFacts,
                          "send is not in the sender NOC domain");
      valid = false;
      return;
    }
    sends.push_back(sendOp);
  });
  if (valid && sends.empty()) {
    debugRejectEndpoint(endpointFacts, "no matching sends");
    valid = false;
  }
  return valid;
}

static bool collectAndCheckPops(ModuleOp mod,
                                const PipeCapacityEndpointFacts &endpointFacts,
                                const PipeGraph &pipeGraph,
                                SmallVectorImpl<CBPopOp> &pops) {
  LaunchNodeDomain receiverDomain =
      getSingleLaunchNodeDomain({endpointFacts.receiverDFB.receiver.x,
                                 endpointFacts.receiverDFB.receiver.y});
  bool valid = true;
  mod.walk([&](CBPopOp popOp) {
    if (!isReceiverDFB(popOp.getCb(), endpointFacts.receiverDFB)) {
      return;
    }
    LaunchNodeDomain popDomain =
        pipeGraph.getOperationLaunchDomain(popOp.getOperation());
    if (!launchNodeDomainsOverlap(popDomain, receiverDomain)) {
      return;
    }
    if (!(popDomain == receiverDomain) || !isNocKernelThread(popOp)) {
      debugRejectEndpoint(endpointFacts,
                          "pop is not in the receiver NOC domain");
      valid = false;
      return;
    }
    std::optional<int64_t> maybeReleasedBlocks = getReleasedBlockCount(popOp);
    if (!maybeReleasedBlocks || *maybeReleasedBlocks != 1) {
      debugRejectEndpoint(endpointFacts, "pop does not release one DFB block");
      valid = false;
      return;
    }
    const DFBReleaseOwnerMaps &owners = pipeGraph.getDFBReleaseOwnerMaps();
    auto waitOp = lookupOwner<CBWaitOp>(owners.waitByPop, popOp.getOperation());
    if (!waitOp) {
      debugRejectEndpoint(endpointFacts,
                          "pop is not owned by a matching receiver wait");
      valid = false;
      return;
    }
    LaunchNodeDomain waitDomain =
        pipeGraph.getOperationLaunchDomain(waitOp.getOperation());
    if (!(waitDomain == receiverDomain) || !isNocKernelThread(waitOp)) {
      debugRejectEndpoint(endpointFacts,
                          "wait is not in the receiver NOC domain");
      valid = false;
      return;
    }
    std::optional<int64_t> maybeWaitedBlocks = getWaitedBlockCount(waitOp);
    if (!maybeWaitedBlocks || *maybeWaitedBlocks != *maybeReleasedBlocks) {
      debugRejectEndpoint(endpointFacts,
                          "wait and pop release different DFB block counts");
      valid = false;
      return;
    }
    pops.push_back(popOp);
  });
  if (valid && pops.empty()) {
    debugRejectEndpoint(endpointFacts, "no matching receiver pops");
    valid = false;
  }
  return valid;
}

} // namespace

PipeCapacityAnalysisResult analyzePipeCapacity(ModuleOp mod,
                                               const PipeGraph &pipeGraph) {
  PipeCapacityAnalysisResult result;
  if (!pipeGraph.hasLaunchGrid()) {
    LLVM_DEBUG(llvm::dbgs()
               << "PipeCapacity: skip module without ttl.launch_grid\n");
    return result;
  }

  LLVM_DEBUG(llvm::dbgs() << "PipeCapacity: "
                          << pipeGraph.getReceiverDFBNodes().size()
                          << " receiver DFB node(s), "
                          << pipeGraph.getPipeReceiverEndpoints().size()
                          << " receiver endpoint(s)\n");

  for (const PipeReceiverEndpoint &receiverEndpoint :
       pipeGraph.getPipeReceiverEndpoints()) {
    PipeCapacityEndpointFacts endpointFacts =
        getCapacityEndpointFacts(pipeGraph, receiverEndpoint);
    debugCandidateEndpoint(endpointFacts);
    const PipeTransferNode &transferNode =
        pipeGraph.getPipeTransferNode(endpointFacts.transferNode);

    // The current capacity protocol proves one receiver's progress per
    // endpoint. Collective progress requires a separate proof that a fast
    // receiver cannot release capacity still owned by another receiver.
    if (isCollectiveTransfer(transferNode.transferContract)) {
      debugRejectEndpoint(
          endpointFacts,
          "collective capacity-counter synchronization requires per-receiver "
          "release accounting");
      continue;
    }

    ArrayRef<PipeReceiverEndpointId> writerEndpoints =
        pipeGraph.getReceiverDFBWriterEndpoints(endpointFacts.receiverDFBNode);
    if (writerEndpoints.size() != 1) {
      debugRejectEndpoint(endpointFacts,
                          "receiver DFB has " +
                              llvm::Twine(writerEndpoints.size()) +
                              " writer endpoint(s)");
      continue;
    }

    const PipeReceiverDFBNode &receiverDFBNode =
        pipeGraph.getReceiverDFBNode(endpointFacts.receiverDFBNode);
    if (!receiverDFBNode.hasProvenPipeOnlyProducerStream) {
      debugRejectEndpoint(endpointFacts,
                          "receiver DFB producer stream is not proven "
                          "pipe-only");
      continue;
    }

    // Capacity adds one unit per pop, so each matching reserve must also
    // consume one block. Enforce that invariant here even though pop validation
    // currently rejects wider spans.
    if (receiverEndpoint.receiverDFBInfo.receiverSlotSpanBlocks != 1) {
      debugRejectEndpoint(
          endpointFacts,
          "receiver reserve spans " +
              llvm::Twine(
                  receiverEndpoint.receiverDFBInfo.receiverSlotSpanBlocks) +
              " DFB blocks; capacity accounting assumes one");
      continue;
    }

    if (!collectAndCheckPosts(mod, endpointFacts, pipeGraph)) {
      continue;
    }

    if (!collectAndCheckSends(mod, endpointFacts, pipeGraph,
                              endpointFacts.sends)) {
      continue;
    }

    if (!collectAndCheckPops(mod, endpointFacts, pipeGraph,
                             endpointFacts.pops)) {
      continue;
    }

    debugAcceptEndpoint(endpointFacts, endpointFacts.sends.size(),
                        endpointFacts.pops.size());
    result.addEndpointFacts(std::move(endpointFacts));
  }

  return result;
}

} // namespace mlir::tt::ttl
