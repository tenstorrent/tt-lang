// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "PipeCapacityAnalysis.h"

#include "DFBAcquireReleaseAnalysis.h"
#include "PipeTransportDFBAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/Debug.h"

#include <cstddef>
#include <optional>
#include <string>

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
                                int64_t popCount) {
  LLVM_DEBUG({
    llvm::dbgs() << "PipeCapacity: accept ";
    endpointFacts.print(llvm::dbgs());
    llvm::dbgs() << ": sends=1 pops=" << popCount << "\n";
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
      receiverEndpoint.receiverDFBInfo.receiverSlotSpanBlocks,
      false,
      PipeTransferSendOp(),
      {},
  };
}

static bool checkPosts(const PipeCapacityEndpointFacts &endpointFacts,
                       const PipeGraph &pipeGraph) {
  LaunchNodeDomain receiverDomain =
      getSingleLaunchNodeDomain({endpointFacts.receiverDFB.receiver.x,
                                 endpointFacts.receiverDFB.receiver.y});
  const PipeTransferNode &transferNode =
      pipeGraph.getPipeTransferNode(endpointFacts.transferNode);
  bool sawPost = false;
  bool valid = true;
  for (Operation *postOperation : transferNode.receiverPostOps) {
    auto postOp = llvm::cast<PipeTransferPostOp>(postOperation);
    LaunchNodeDomain postDomain =
        pipeGraph.getOperationLaunchDomain(postOp.getOperation());
    if (!launchNodeDomainsOverlap(postDomain, receiverDomain)) {
      continue;
    }
    if (!isExactlyDomain(postOp, receiverDomain, pipeGraph) ||
        !isNocKernelThread(postOp)) {
      debugRejectEndpoint(endpointFacts,
                          "post is not in the receiver NOC domain");
      valid = false;
      continue;
    }
    if (!isReceiverDFBView(postOp.getDst(), endpointFacts.receiverDFB)) {
      debugRejectEndpoint(endpointFacts,
                          "post destination is not the receiver DFB");
      valid = false;
      continue;
    }
    sawPost = true;
  }
  if (valid && !sawPost) {
    debugRejectEndpoint(endpointFacts, "no matching receiver posts");
    valid = false;
  }
  return valid;
}

static FailureOr<PipeTransferSendOp>
checkSend(const PipeCapacityEndpointFacts &endpointFacts,
          const PipeGraph &pipeGraph) {
  const PipeCapacityReleaseTarget &target = endpointFacts.releaseTarget;
  LaunchNodeDomain sourceDomain =
      getSingleLaunchNodeDomain({target.logicalX, target.logicalY});
  const PipeTransferNode &transferNode =
      pipeGraph.getPipeTransferNode(endpointFacts.transferNode);
  auto sendOp = llvm::dyn_cast<PipeTransferSendOp>(transferNode.sendOp);
  if (!sendOp) {
    debugRejectEndpoint(endpointFacts, "no matching sends");
    return failure();
  }
  if (!isExactlyDomain(sendOp, sourceDomain, pipeGraph) ||
      !isNocKernelThread(sendOp)) {
    debugRejectEndpoint(endpointFacts, "send is not in the sender NOC domain");
    return failure();
  }
  return sendOp;
}

/// Validate one receiver release and record it for capacity lowering.
static bool checkAndCollectPop(CBPopOp popOp,
                               const PipeCapacityEndpointFacts &endpointFacts,
                               const PipeGraph &pipeGraph,
                               SmallVectorImpl<CBPopOp> &pops) {
  LaunchNodeDomain receiverDomain =
      getSingleLaunchNodeDomain({endpointFacts.receiverDFB.receiver.x,
                                 endpointFacts.receiverDFB.receiver.y});
  LaunchNodeDomain popDomain =
      pipeGraph.getOperationLaunchDomain(popOp.getOperation());
  if (!(popDomain == receiverDomain) || !isNocKernelThread(popOp)) {
    debugRejectEndpoint(endpointFacts, "pop is not in the receiver NOC domain");
    return false;
  }
  std::optional<int64_t> maybeReleasedBlocks =
      getDFBTransactionBlockCount(popOp);
  if (!maybeReleasedBlocks ||
      *maybeReleasedBlocks != endpointFacts.receiverBlocksPerTransfer) {
    debugRejectEndpoint(endpointFacts,
                        "pop does not release the transfer's DFB block span");
    return false;
  }
  ArrayRef<Operation *> owners =
      pipeGraph.getDFBAcquireReleaseIndex(popOp).getReleaseIntervalOwners(
          popOp);
  if (owners.size() != 1 || !isa<CBWaitOp>(owners.front())) {
    debugRejectEndpoint(endpointFacts,
                        "pop is not owned by a matching receiver wait");
    return false;
  }
  CBWaitOp waitOp = cast<CBWaitOp>(owners.front());
  LaunchNodeDomain waitDomain =
      pipeGraph.getOperationLaunchDomain(waitOp.getOperation());
  if (!(waitDomain == receiverDomain) || !isNocKernelThread(waitOp)) {
    debugRejectEndpoint(endpointFacts,
                        "wait is not in the receiver NOC domain");
    return false;
  }
  std::optional<int64_t> maybeWaitedBlocks =
      getDFBTransactionBlockCount(waitOp);
  if (!maybeWaitedBlocks || *maybeWaitedBlocks != *maybeReleasedBlocks) {
    debugRejectEndpoint(endpointFacts,
                        "wait and pop release different DFB block counts");
    return false;
  }
  pops.push_back(popOp);
  return true;
}

/// Collect releases from transport ownership or the physical receiver DFB.
static bool collectAndCheckPops(ArrayRef<CBPopOp> candidatePops,
                                const PipeCapacityEndpointFacts &endpointFacts,
                                const PipeGraph &pipeGraph,
                                ArrayRef<CBPopOp> ownedPops,
                                SmallVectorImpl<CBPopOp> &pops) {
  bool valid = true;
  if (!ownedPops.empty()) {
    for (CBPopOp popOp : ownedPops) {
      valid &= checkAndCollectPop(popOp, endpointFacts, pipeGraph, pops);
    }
  } else {
    LaunchNodeDomain receiverDomain =
        getSingleLaunchNodeDomain({endpointFacts.receiverDFB.receiver.x,
                                   endpointFacts.receiverDFB.receiver.y});
    for (CBPopOp popOp : candidatePops) {
      LaunchNodeDomain popDomain =
          pipeGraph.getOperationLaunchDomain(popOp.getOperation());
      if (!launchNodeDomainsOverlap(popDomain, receiverDomain)) {
        continue;
      }
      valid &= checkAndCollectPop(popOp, endpointFacts, pipeGraph, pops);
    }
  }
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

  llvm::DenseMap<int64_t, SmallVector<CBPopOp>> popsByDFBIndex;
  mod.walk([&](CBPopOp popOp) {
    std::optional<int64_t> maybeDFBIndex = getCBIndex(popOp.getCb());
    if (maybeDFBIndex) {
      popsByDFBIndex[*maybeDFBIndex].push_back(popOp);
    }
  });

  for (const PipeReceiverEndpoint &receiverEndpoint :
       pipeGraph.getPipeReceiverEndpoints()) {
    PipeCapacityEndpointFacts endpointFacts =
        getCapacityEndpointFacts(pipeGraph, receiverEndpoint);
    debugCandidateEndpoint(endpointFacts);
    const PipeTransferNode &transferNode =
        pipeGraph.getPipeTransferNode(endpointFacts.transferNode);
    std::string ownershipFailure;
    FailureOr<PipeTransportDFBOwnership> transportOwnership =
        analyzePipeTransportDFBOwnership(transferNode, pipeGraph,
                                         ownershipFailure);
    bool transportOwnsStorage =
        succeeded(transportOwnership) &&
        transportOwnership->endpoint == receiverEndpoint.id;
    if (transportOwnsStorage) {
      std::optional<int64_t> initialCapacity = llvm::checkedMul(
          transferNode.blockSpan, transferNode.destinationGroupDepth);
      if (!initialCapacity) {
        debugRejectEndpoint(endpointFacts,
                            "transport capacity exceeds int64_t");
        continue;
      }
      endpointFacts.initialCapacity = *initialCapacity;
      endpointFacts.receiverBlocksPerTransfer = transferNode.blockSpan;
      endpointFacts.transportOwnsStorage = true;
    }

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

    if (!transportOwnsStorage) {
      ArrayRef<PipeReceiverEndpointId> writerEndpoints =
          pipeGraph.getReceiverDFBWriterEndpoints(
              endpointFacts.receiverDFBNode);
      if (writerEndpoints.size() != 1) {
        std::string reason = "receiver DFB has " +
                             std::to_string(writerEndpoints.size()) +
                             " writer endpoint(s)";
        debugRejectEndpoint(endpointFacts, reason);
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
    }

    if (!checkPosts(endpointFacts, pipeGraph)) {
      continue;
    }

    FailureOr<PipeTransferSendOp> maybeSendOp =
        checkSend(endpointFacts, pipeGraph);
    if (failed(maybeSendOp)) {
      continue;
    }
    endpointFacts.send = *maybeSendOp;

    auto candidatePopsIt =
        popsByDFBIndex.find(endpointFacts.receiverDFB.dfbIndex);
    ArrayRef<CBPopOp> candidatePops;
    if (candidatePopsIt != popsByDFBIndex.end()) {
      candidatePops = candidatePopsIt->second;
    }
    ArrayRef<CBPopOp> ownedPops =
        transportOwnsStorage
            ? ArrayRef<CBPopOp>(transportOwnership->destination.pops)
            : ArrayRef<CBPopOp>();
    if (!collectAndCheckPops(candidatePops, endpointFacts, pipeGraph, ownedPops,
                             endpointFacts.pops)) {
      continue;
    }

    debugAcceptEndpoint(endpointFacts, endpointFacts.pops.size());
    result.addEndpointFacts(std::move(endpointFacts));
  }

  return result;
}

} // namespace mlir::tt::ttl
