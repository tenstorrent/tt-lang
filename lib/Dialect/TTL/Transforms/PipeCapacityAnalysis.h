// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Pipe Capacity Analysis
//===----------------------------------------------------------------------===//
//
// This file declares the analysis and lowering plan for PipeNet transfers that
// can replace receiver-post synchronization with sender capacity counters.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPECAPACITYANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPECAPACITYANALYSIS_H

// PipeGraph contains one transfer node per send and its corresponding receiver
// posts, one endpoint per receiver, and one physical DFB node per receiver
// coordinate and finalized DFB index. Transfer nodes remain distinct even when
// they have the same PipeKey.
//
// The safety invariant is graph-local: a receiver dataflow buffer pop may
// release sender capacity only when its receiver dataflow buffer node has
// exactly one writer endpoint and a proven pipe-only producer stream. Every
// producer-side DFB advance must be owned by a receiver reserve whose matching
// pipe posts complete before the advance. Capacity analysis separately requires
// every receiver pop to have a matching `ttl.cb_wait`. With one writer
// endpoint, each valid one-block pop frees one capacity unit for that
// endpoint's sender. With zero or multiple writer endpoints, the pop identifies
// only the DFB, so its sender is ambiguous.
//
// Fabric transfers use routing-plane flow control and are excluded from
// capacity-counter selection.
//
// This analysis must run after `ttl-insert-cb-sync` and
// `ttl-finalize-dfb-indices`. The proof depends on finalized receiver dataflow
// buffer ids and on concrete `ttl.cb_pop` ops, including pops inserted by the
// compiler for Python `with` regions.
//
// Pseudocode:
//
//   for endpoint in pipeGraph.getPipeReceiverEndpoints():
//     node = pipeGraph.getReceiverDFBNode(endpoint.receiverDFBNode)
//     require node.writerEndpoints.size() == 1
//     require every endpoint post to target the receiver DFB from the receiver
//             NOC thread
//     require the receiver DFB node to have a proven pipe-only producer stream
//     require every send to run on the sender NOC thread
//     require every receiver-overlapping pop of the DFB to be owned by a
//             receiver-domain wait and free one block
//
//   for transferNode in pipeGraph.getPipeTransferNodes():
//     require the transfer to use the intra-device point-to-point NoC transport
//     require every receiver endpoint to have proven lowerable capacity facts
//     initialize each receiver endpoint's sender-local capacity counter to that
//             receiver dataflow buffer's block_count
//     reuse counter storage across different source nodes only when their
//             initial block counts match
//     record one capacity acquire per endpoint for each send
//     record one capacity release to the sender for each endpoint pop
//     mark the transfer's send and receiver posts as using the capacity
//     protocol
//
// If an endpoint requirement is not proven, that endpoint has no capacity
// fact. If any endpoint of a transfer node is missing a proven lowerable fact,
// the analysis records no facts for that transfer. TTL-to-TTKernel lowering
// uses proven facts to replace receiver-post sender readiness with sender-local
// capacity counters.

#include "PipeCounter.h"
#include "PipeGraph.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttl {

struct PipeResourcePlan;

/// Source node that receives a capacity-release increment over the NoC.
struct PipeCapacityReleaseTarget {
  int64_t logicalX = 0;
  int64_t logicalY = 0;
};

/// Capacity consumed by one sender before issuing a payload write.
struct PipeCapacityAcquireInfo {
  PipeCounterInfo counter;
  int64_t count = 1;
};

/// Capacity returned to a sender after one receiver DFB pop.
struct PipeCapacityReleaseInfo {
  PipeCapacityReleaseTarget target;
  PipeCounterInfo counter;
  int64_t count = 1;
};

/// Initial shared capacity and storage selected for one sender counter.
struct PipeCapacityInitInfo {
  PipeCounterInfo counter;
  int64_t initialCapacity = 0;
};

/// Lowering plan for transfers proven safe to use capacity synchronization.
class PipeCapacityPlan {
public:
  /// Return the capacity consumed before each execution of `op`.
  ArrayRef<PipeCapacityAcquireInfo> lookupAcquires(PipeTransferSendOp op) const;

  /// Return the capacity released immediately after `op`.
  ArrayRef<PipeCapacityReleaseInfo> lookupReleases(CBPopOp op) const;

  /// Return whether `op` belongs to a capacity-synchronized transfer.
  bool usesCapacityProtocol(PipeTransferSendOp op) const;

  /// Return whether `op` belongs to a capacity-synchronized transfer.
  bool usesCapacityProtocol(PipeTransferPostOp op) const;

  /// Return the shared-counter initializations grouped by sender function.
  const llvm::MapVector<func::FuncOp, SmallVector<PipeCapacityInitInfo>> &
  getInitializations() const {
    return initializations;
  }

  /// Return whether no transfer has been assigned capacity resources.
  bool empty() const {
    return acquires.empty() && releases.empty() && initializations.empty();
  }

  /// Return whether two plans selected the same sends and receiver posts.
  bool hasSameSelectedTransfers(const PipeCapacityPlan &other) const;

  /// Record one sender acquire for `op`.
  void addAcquire(PipeTransferSendOp op, PipeCapacityAcquireInfo info);

  /// Record one receiver release for `op`.
  void addRelease(CBPopOp op, PipeCapacityReleaseInfo info);

  /// Record the initial shared count for `counter` in `func`.
  void addInitialization(func::FuncOp func, PipeCapacityInitInfo info);

  /// Select capacity synchronization for `op`.
  void markCapacityTransfer(PipeTransferSendOp op);

  /// Select capacity synchronization for `op`.
  void markCapacityTransfer(PipeTransferPostOp op);

  /// Continue allocation after the completion and readiness counters.
  void initializeCounterAllocation(PipeCounterAllocationCounts counts);

  /// Allocate storage for one proven sender-capacity counter.
  PipeCounterInfo allocateCounter();

  /// Return the combined completion, readiness, and capacity totals.
  PipeCounterAllocationCounts getCounterAllocationCounts() const {
    return counterAllocator.getCounts();
  }

private:
  llvm::MapVector<Operation *, SmallVector<PipeCapacityAcquireInfo>> acquires;
  llvm::MapVector<Operation *, SmallVector<PipeCapacityReleaseInfo>> releases;
  llvm::MapVector<func::FuncOp, SmallVector<PipeCapacityInitInfo>>
      initializations;
  llvm::SmallPtrSet<Operation *, 16> capacityTransferOps;
  PipeCounterAllocator counterAllocator;
};

/// Build capacity synchronization facts for every proven transfer in `mod`.
void buildPipeCapacityPlan(ModuleOp mod, const PipeGraph &pipeGraph,
                           const PipeResourcePlan &resources,
                           PipeCapacityPlan &plan);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPECAPACITYANALYSIS_H
