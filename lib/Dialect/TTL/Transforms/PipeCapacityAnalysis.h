// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Pipe Capacity Analysis
//===----------------------------------------------------------------------===//
//
// This file declares the analysis facts required to prove that receiver DFB
// releases can replenish sender capacity counters.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPECAPACITYANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPECAPACITYANALYSIS_H

// PipeGraph contains one transfer node per send and its corresponding receiver
// posts, one endpoint per receiver, and one logical DFB lifecycle node per
// receiver coordinate. Each lifecycle node also records its physical index.
// Transfer nodes remain distinct even when they have the same PipeKey.
//
// The safety invariant is graph-local. For ordinary DFB storage, a receiver
// pop may release sender capacity only when its receiver DFB node has exactly
// one writer endpoint and a proven pipe-only producer stream. Every
// producer-side DFB advance must be owned by a receiver reserve whose matching
// pipe posts complete before the advance. Capacity analysis separately
// requires every receiver pop to have a matching `ttl.cb_wait`. With one writer
// endpoint, each valid pop frees the transfer's receiver block span for that
// endpoint's sender. With zero or multiple writer endpoints, the pop identifies
// only the DFB, so its sender is ambiguous.
//
// A grouped point-to-point transport may replace a complete source and
// destination DFB lifecycle with private transport storage. In that case, the
// ownership proof identifies the exact destination pop that releases capacity,
// so other users of the physical DFB do not make the release ambiguous.
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
//     require endpoint.transferNode to be point-to-point
//     if the transport owns the endpoint storage:
//       use the owned destination pop and the transport ring depth
//     else:
//       require node.writerEndpoints.size() == 1
//       require the receiver DFB node to have a proven pipe-only producer
//       stream
//     require every endpoint post to execute on the receiver NOC thread
//     require every send to run on the sender NOC thread
//     require every receiver-overlapping pop of the DFB to be owned by a
//             receiver-domain wait and free endpoint.receiverSlotSpanBlocks
//             blocks
//
// If an endpoint requirement is not proven, that endpoint has no capacity
// fact. Protocol selection and counter allocation consume these facts but are
// not part of the analysis.

#include "PipeGraph.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>

namespace mlir::tt::ttl {

/// Source node that receives a capacity-release increment over the NoC.
struct PipeCapacityReleaseTarget {
  int64_t logicalX = 0;
  int64_t logicalY = 0;
};

/// Proven operations that consume and release one receiver endpoint's DFB
/// capacity.
struct PipeCapacityEndpointFacts {
  PipeTransferNodeId transferNode = 0;
  PipeReceiverEndpointId endpoint = 0;
  PipeReceiverDFBNodeId receiverDFBNode = 0;
  PipeReceiverDFBKey receiverDFB;
  PipeCapacityReleaseTarget releaseTarget;
  int64_t initialCapacity = 0;
  int64_t receiverBlocksPerTransfer = 1;
  /// The grouped transport replaces this endpoint's original DFB storage.
  bool transportOwnsStorage = false;
  PipeTransferSendOp send;
  SmallVector<CBPopOp> pops;

  /// Print the endpoint and its initial capacity for diagnostics.
  void print(llvm::raw_ostream &os) const;
};

/// Capacity facts proven independently of resource and address selection.
class PipeCapacityAnalysisResult {
public:
  /// Return whether capacity accounting was proven for `endpoint`.
  bool hasEndpointFacts(PipeReceiverEndpointId endpoint) const;

  /// Return the proven facts for `endpoint`.
  const PipeCapacityEndpointFacts &
  getEndpointFacts(PipeReceiverEndpointId endpoint) const;

private:
  friend PipeCapacityAnalysisResult analyzePipeCapacity(const PipeGraph &);

  void addEndpointFacts(PipeCapacityEndpointFacts facts);

  SmallVector<PipeCapacityEndpointFacts> endpointFacts;
  llvm::DenseMap<PipeReceiverEndpointId, std::size_t> factsIndexByEndpoint;
};

/// Prove capacity accounting facts for the graph's receiver endpoints.
PipeCapacityAnalysisResult analyzePipeCapacity(const PipeGraph &pipeGraph);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPECAPACITYANALYSIS_H
