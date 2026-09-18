// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Pipe Transport DFB Analysis
//===----------------------------------------------------------------------===//
//
// This file declares the proof that one DFB lifecycle is private to one pipe
// transport stream.
//
//===----------------------------------------------------------------------===//

#ifndef TTLANG_DIALECT_TTL_TRANSFORMS_PIPETRANSPORTDFBANALYSIS_H
#define TTLANG_DIALECT_TTL_TRANSFORMS_PIPETRANSPORTDFBANALYSIS_H

#include "PipeGraph.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::tt::ttl {

/// Identifies how one transport stream uses a DFB.
enum class PipeTransportDFBRole {
  Source,
  Destination,
};

/// Complete reserve/push and wait/pop lifecycle for one transport DFB.
struct PipeTransportDFBUse {
  Value dfb;
  BindCBOp bind;
  PipeTransportDFBRole role = PipeTransportDFBRole::Source;
  PipeTransferNodeId transferNode = 0;
  SmallVector<CBReserveOp> reserves;
  SmallVector<CBPushOp> pushes;
  SmallVector<CBWaitOp> waits;
  SmallVector<CBPopOp> pops;
  SmallVector<AttachCBOp> attaches;
  CopyOp tensorCopy;
  TensorSliceOp tensorSlice;
};

/// Source and destination DFB lifecycles replaceable by one transport stream.
struct PipeTransportDFBOwnership {
  PipeTransferNodeId transferNode = 0;
  PipeReceiverEndpointId endpoint = 0;
  scf::ForOp loop;
  PipeTransportDFBUse source;
  PipeTransportDFBUse destination;
};

/// Return whether every use of `dfb` is nested in `loop`.
bool hasOnlyPipeTransportLoopUses(scf::ForOp loop, Value dfb);

/// Return whether acquired views are private to one transport role.
bool hasPrivatePipeTransportDFBViews(const PipeTransportDFBUse &dfbUse,
                                     const PipeGraph &pipeGraph);

/// Prove that `dfb` has one complete lifecycle owned by `transferNode`.
///
/// The lifecycle inside `loop` must contain one reserve/push pair, one wait/pop
/// pair, and one contiguous tensor copy in the direction required by `role`.
/// The release-owner proof must associate each release with the corresponding
/// acquire. Uses outside `loop` are ignored so a grouped transfer and its
/// scalar residual can use independent storage. Call
/// `hasOnlyPipeTransportLoopUses` when transforming the DFB declaration itself.
/// On failure, `reason` describes why the lifecycle is unsupported.
FailureOr<PipeTransportDFBUse>
analyzePipeTransportDFBUse(scf::ForOp loop, Value dfb,
                           PipeTransportDFBRole role,
                           PipeTransferNodeId transferNode,
                           const PipeGraph &pipeGraph, std::string &reason);

/// Prove that one grouped point-to-point transfer owns both DFB lifecycles.
///
/// The source and destination operations must execute in the same static loop,
/// use distinct nodes, and expose no acquired view outside their transport
/// roles. Uses in a scalar residual loop are permitted because transport-owned
/// storage does not advance the original DFB state.
FailureOr<PipeTransportDFBOwnership>
analyzePipeTransportDFBOwnership(const PipeTransferNode &transferNode,
                                 const PipeGraph &pipeGraph,
                                 std::string &reason);

} // namespace mlir::tt::ttl

#endif // TTLANG_DIALECT_TTL_TRANSFORMS_PIPETRANSPORTDFBANALYSIS_H
