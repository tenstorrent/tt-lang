// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Verify that PipeNet-coupled operations execute only on launch nodes whose
// PipeNet roles permit them. The analysis is a `DenseForwardDataFlowAnalysis`
// whose lattice is the set of launch coordinates that can reach each program
// point. Predicate-bearing region ops (`scf.if`, `affine.if`, `ttl.if_src`,
// `ttl.if_dst`, `ttl.pipenet_foreach_src`, `ttl.pipenet_foreach_dst`,
// `ttl.pipenet_scope`) narrow that set on region entry; pipe-coupled ops are
// checked against the narrowed set.
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "ttlang/Analysis/ValueOriginAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstddef>
#include <functional>
#include <optional>
#include <tuple>

#define DEBUG_TYPE "ttl-verify-pipenet-guards"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLVERIFYPIPENETGUARDS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

constexpr std::size_t kMaxPipeScheduleCycleNotes = 8;

//===----------------------------------------------------------------------===//
// Module state collected before the analysis runs and updated during it.
//===----------------------------------------------------------------------===//

/// A dataflow buffer wait and the launch-node domain where it executes.
struct WaitUse {
  CBWaitOp op;
  LaunchNodeDomain domain;
  int64_t dfbId;
};

/// Return true if `copyOp` publishes a destination dataflow buffer slot for a
/// pipe receive.
bool isPipeReceiveCopy(CopyOp copyOp) {
  return mlir::isa<PipeType, SelectedPipeDstType>(copyOp.getSrc().getType()) &&
         getAttachedCB(copyOp.getDst());
}

/// Return the unique pipe receive copied by a wait, or no value for a non-pipe
/// wait. Fail when different possible sources require different events.
FailureOr<std::optional<CopyOp>>
findDefiningPipeReceiveCopy(ValueOriginAnalysis &analysis, Value value) {
  return analysis.getOrigins(value).uniqueMapped<std::optional<CopyOp>>(
      [](Value origin) -> FailureOr<std::optional<CopyOp>> {
        if (auto copyOp = origin.getDefiningOp<CopyOp>()) {
          return isPipeReceiveCopy(copyOp) ? std::optional<CopyOp>(copyOp)
                                           : std::optional<CopyOp>();
        }
        if (origin.getDefiningOp<PipeTransferSendOp>()) {
          return std::optional<CopyOp>();
        }
        return failure();
      });
}

/// Pipe synchronization event used by the wait-for graph verifier.
enum class PipeEventKind { Send, ReceivePost, ReceiveWait };

/// One pipe synchronization event on the launch-node domain where it executes.
struct PipeEvent {
  Operation *op = nullptr;
  PipeType pipeType;
  PipeEventKind kind;
  LaunchNodeDomain domain;
  Operation *selectedForeachOp = nullptr;
  int64_t selectedRecordIndex = -1;
};

struct SelectedPipeRecords {
  PipeNetRecordsAttr records;
  PipeRole role = PipeRole::Active;
  Operation *foreachOp = nullptr;
};

std::optional<SelectedPipeRecords> getSelectedPipeRecords(Value pipe) {
  pipe = traceUnrealizedCasts(pipe);
  if (auto selectedSrc = pipe.getDefiningOp<SelectPipeSrcOp>()) {
    return SelectedPipeRecords{selectedSrc.getRecords(), PipeRole::Source,
                               nullptr};
  }
  if (auto selectedDst = pipe.getDefiningOp<SelectPipeDstOp>()) {
    return SelectedPipeRecords{selectedDst.getRecords(), PipeRole::Destination,
                               nullptr};
  }
  auto blockArg = mlir::dyn_cast<BlockArgument>(pipe);
  if (!blockArg || blockArg.getArgNumber() != 0) {
    return std::nullopt;
  }
  Operation *owner = blockArg.getOwner()->getParentOp();
  if (auto foreachSrc = mlir::dyn_cast<PipeNetForeachSrcOp>(owner)) {
    return SelectedPipeRecords{foreachSrc.getRecords(), PipeRole::Source,
                               owner};
  }
  if (auto foreachDst = mlir::dyn_cast<PipeNetForeachDstOp>(owner)) {
    return SelectedPipeRecords{foreachDst.getRecords(), PipeRole::Destination,
                               owner};
  }
  return std::nullopt;
}

std::optional<SelectedPipeRecords> getSelectedSourceRecords(Value pipe) {
  std::optional<SelectedPipeRecords> selected = getSelectedPipeRecords(pipe);
  if (selected && selected->role == PipeRole::Source) {
    return selected;
  }
  return std::nullopt;
}

std::optional<SelectedPipeRecords> getSelectedDestinationRecords(Value pipe) {
  std::optional<SelectedPipeRecords> selected = getSelectedPipeRecords(pipe);
  if (selected && selected->role == PipeRole::Destination) {
    return selected;
  }
  return std::nullopt;
}

struct ModuleState;
void checkKnownSubset(Operation *op, const LaunchNodeDomain &current,
                      const LaunchNodeDomain &allowed,
                      Operation *unanalyzableOp, Twine primaryMessage,
                      ArrayRef<std::pair<int64_t, PipeRole>> roles,
                      ModuleState &state);

struct ModuleState : LaunchNodeDomainState {
  explicit ModuleState(ModuleOp module) : valueOrigins(module) {}

  ValueOriginAnalysis valueOrigins;
  llvm::DenseMap<int64_t, LaunchNodeDomain> dfbProducerDomains;
  SmallVector<WaitUse> waitUses;
  SmallVector<PipeEvent> pipeEvents;
  llvm::DenseMap<Operation *, SmallVector<std::size_t>> pipeEventIndices;

  void replacePipeEvents(Operation *op, SmallVector<PipeEvent> events) {
    if (events.empty()) {
      return;
    }
    auto it = pipeEventIndices.find(op);
    if (it == pipeEventIndices.end()) {
      SmallVector<std::size_t> indices;
      indices.reserve(events.size());
      for (PipeEvent &event : events) {
        indices.push_back(pipeEvents.size());
        pipeEvents.push_back(event);
      }
      pipeEventIndices.try_emplace(op, std::move(indices));
      return;
    }

    assert(it->second.size() == events.size() &&
           "pipe event count for an op changed during analysis");
    for (auto [index, event] : llvm::zip_equal(it->second, events)) {
      pipeEvents[index] = event;
    }
  }

  void appendSelectedPipeEvents(Operation *op, PipeNetRecordsAttr records,
                                PipeEventKind kind,
                                const LaunchNodeDomain &domain,
                                Operation *foreachOp,
                                SmallVectorImpl<PipeEvent> &events) {
    PipeRole role =
        kind == PipeEventKind::Send ? PipeRole::Source : PipeRole::Destination;
    for (auto [recordIndex, record] : llvm::enumerate(records.getPipes())) {
      PipeType pipeType = PipeType::get(
          records.getContext(), record.getSrcX(), record.getSrcY(),
          record.getDstStartX(), record.getDstStartY(), record.getDstEndX(),
          record.getDstEndY(), records.getPipeNetId());
      LaunchNodeDomain roleDomain =
          role == PipeRole::Source
              ? getPipeRecordSourceLaunchNodeDomain(record)
              : getPipeRecordDestinationLaunchNodeDomain(record);
      events.push_back(PipeEvent{op, pipeType, kind,
                                 domain.intersectWith(roleDomain), foreachOp,
                                 static_cast<int64_t>(recordIndex)});
    }
  }

  SmallVector<PipeEvent> getPipeCopyEvents(CopyOp copyOp,
                                           const LaunchNodeDomain &domain) {
    SmallVector<PipeEvent> events;
    Operation *op = copyOp.getOperation();
    if (auto pipeType = mlir::dyn_cast<PipeType>(copyOp.getDst().getType())) {
      events.push_back(PipeEvent{
          op, pipeType, PipeEventKind::Send,
          domain.intersectWith(getPipeSourceLaunchNodeDomain(pipeType)),
          nullptr, -1});
      return events;
    }
    if (std::optional<SelectedPipeRecords> selected =
            getSelectedSourceRecords(copyOp.getDst())) {
      appendSelectedPipeEvents(op, selected->records, PipeEventKind::Send,
                               domain, selected->foreachOp, events);
      return events;
    }
    if (auto pipeType = mlir::dyn_cast<PipeType>(copyOp.getSrc().getType())) {
      if (isPipeReceiveCopy(copyOp)) {
        events.push_back(PipeEvent{
            op, pipeType, PipeEventKind::ReceivePost,
            domain.intersectWith(getPipeDestinationLaunchNodeDomain(pipeType)),
            nullptr, -1});
      }
      return events;
    }
    if (std::optional<SelectedPipeRecords> selected =
            getSelectedDestinationRecords(copyOp.getSrc())) {
      if (isPipeReceiveCopy(copyOp)) {
        appendSelectedPipeEvents(op, selected->records,
                                 PipeEventKind::ReceivePost, domain,
                                 selected->foreachOp, events);
      }
    }
    return events;
  }

  void recordPipeEvent(CopyOp copyOp, const LaunchNodeDomain &domain) {
    replacePipeEvents(copyOp.getOperation(), getPipeCopyEvents(copyOp, domain));
  }

  /// Record a receive completion wait and verify that it is
  /// destination-guarded.
  void recordPipeWaitEvent(WaitOp waitOp, const LaunchNodeDomain &domain,
                           Operation *unanalyzableOp) {
    FailureOr<std::optional<CopyOp>> maybeCopyOp =
        findDefiningPipeReceiveCopy(valueOrigins, waitOp.getXf());
    if (failed(maybeCopyOp)) {
      waitOp.emitOpError()
          << "requires either every possible source to be the same pipe "
             "receive ttl.copy or no source to be a pipe receive";
      sawError = true;
      return;
    }
    if (!maybeCopyOp->has_value()) {
      return;
    }
    CopyOp copyOp = **maybeCopyOp;

    SmallVector<PipeEvent> events;
    Operation *op = waitOp.getOperation();
    if (auto pipeType = mlir::dyn_cast<PipeType>(copyOp.getSrc().getType())) {
      int64_t netId = pipeType.getPipeNetId();
      std::string name = netName(netId);
      std::string msg;
      llvm::raw_string_ostream(msg)
          << "this `ttl.wait` waits for a pipe receive on launched nodes "
             "that are not destinations of PipeNet "
          << name << "; keep the wait under the same `if " << name
          << ".is_dst(): ...` or `" << name
          << ".if_dst(...)` guard as the receive copy";
      checkKnownSubset(
          waitOp, domain, getPipeDestinationLaunchNodeDomain(pipeType),
          unanalyzableOp, msg, {{netId, PipeRole::Destination}}, *this);
      if (sawError) {
        return;
      }
      events.push_back(PipeEvent{
          op, pipeType, PipeEventKind::ReceiveWait,
          domain.intersectWith(getPipeDestinationLaunchNodeDomain(pipeType)),
          nullptr, -1});
    } else if (std::optional<SelectedPipeRecords> selected =
                   getSelectedDestinationRecords(copyOp.getSrc())) {
      int64_t netId = selected->records.getPipeNetId();
      std::string name = netName(netId);
      std::string msg;
      llvm::raw_string_ostream(msg)
          << "this `ttl.wait` waits for a pipe receive on launched nodes "
             "that are not destinations of PipeNet "
          << name << "; keep the wait under the same `if " << name
          << ".is_dst(): ...` or `" << name
          << ".if_dst(...)` guard as the receive copy";
      checkKnownSubset(waitOp, domain,
                       getPipeRecordsRoleLaunchNodeDomain(
                           selected->records, PipeRole::Destination),
                       unanalyzableOp, msg, {{netId, PipeRole::Destination}},
                       *this);
      if (sawError) {
        return;
      }
      appendSelectedPipeEvents(op, selected->records,
                               PipeEventKind::ReceiveWait, domain,
                               selected->foreachOp, events);
    }

    replacePipeEvents(op, std::move(events));
  }
};

//===----------------------------------------------------------------------===//
// Diagnostic helpers.
//===----------------------------------------------------------------------===//

// Render the verifier's role domain back as a runtime predicate string.
// Examples:
//   net_0.is_src()                    (one net, one role)
//   net_0.is_active()                 (one net, src and dst both seen)
//   net_0.is_dst() or net_1.is_src()  (different nets)
//
// Input roles are only `Source` or `Destination` (from `pipenet_scope`);
// `is_active` is synthesized when a net has both.
std::string formatGuardExpression(ArrayRef<std::pair<int64_t, PipeRole>> roles,
                                  const ModuleState &state) {
  SmallVector<int64_t> orderedIds;
  llvm::DenseMap<int64_t, std::pair<bool, bool>> rolesByNet; // (hasSrc, hasDst)
  for (auto [id, role] : roles) {
    auto [it, inserted] = rolesByNet.try_emplace(id, std::pair{false, false});
    if (inserted) {
      orderedIds.push_back(id);
    }
    if (role == PipeRole::Source) {
      it->second.first = true;
    } else {
      it->second.second = true;
    }
  }

  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  bool first = true;
  for (int64_t id : orderedIds) {
    auto [hasSrc, hasDst] = rolesByNet[id];
    if (!first) {
      os << " or ";
    }
    first = false;
    StringRef method =
        (hasSrc && hasDst) ? "is_active" : (hasSrc ? "is_src" : "is_dst");
    os << state.netName(id) << "." << method << "()";
  }
  return buffer;
}

// Emit an op error when `current` is not a subset of `allowed`. Attaches an
// example offending coord, the unanalyzable predicate location (if any), and
// declaration notes for each named PipeNet role.
void checkKnownSubset(Operation *op, const LaunchNodeDomain &current,
                      const LaunchNodeDomain &allowed,
                      Operation *unanalyzableOp, Twine primaryMessage,
                      ArrayRef<std::pair<int64_t, PipeRole>> roles,
                      ModuleState &state) {
  if (!current.known) {
    auto diag = op->emitOpError()
                << "could not statically analyze the PipeNet guard "
                   "around this op; rewrite using `net.is_src()` / "
                   "`net.is_dst()` / `net.is_active()`, or compare "
                   "`ttl.node(dims=2)` coordinates against integer "
                   "constants";
    if (unanalyzableOp) {
      diag.attachNote(unanalyzableOp->getLoc())
          << "this expression is not statically analyzable";
    }
    state.sawError = true;
    return;
  }
  if (current.isSubsetOf(allowed)) {
    return;
  }
  LaunchNodeDomain extra = current.subtract(allowed);
  auto diag = op->emitOpError() << primaryMessage;
  if (extra.known && !extra.nodes.empty()) {
    LaunchNodeCoord example = *extra.nodes.begin();
    diag.attachNote() << "example node where the guard does not hold: "
                      << "core_x=" << example.x << ", core_y=" << example.y;
  }
  for (auto &p : roles) {
    auto it = state.pipeNetLocs.find(p.first);
    if (it == state.pipeNetLocs.end() || it->second.empty()) {
      continue;
    }
    diag.attachNote(it->second.front())
        << "PipeNet " << state.netName(p.first) << " declared here";
  }
  state.sawError = true;
}

// Diagnose a `ttl.copy` whose endpoint is a pipe but whose enclosing domain
// extends outside the pipe's source/destination set.
void verifyCopy(CopyOp copyOp, const LaunchNodeDomain &current,
                Operation *unanalyzable, ModuleState &state) {
  if (auto dstPipeType = mlir::dyn_cast<PipeType>(copyOp.getDst().getType())) {
    int64_t netId = dstPipeType.getPipeNetId();
    std::string name = state.netName(netId);
    std::string msg;
    llvm::raw_string_ostream(msg)
        << "this `ttl.copy(buffer, pipe)` sends data on PipeNet " << name
        << " from a node that is not a source of any pipe in that net; "
           "wrap the copy in `"
        << name << ".if_src(...)` or guard with `if " << name
        << ".is_src(): ...`";
    checkKnownSubset(copyOp, current,
                     getPipeSourceLaunchNodeDomain(dstPipeType), unanalyzable,
                     msg, {{netId, PipeRole::Source}}, state);
    return;
  }
  if (std::optional<SelectedPipeRecords> selected =
          getSelectedSourceRecords(copyOp.getDst())) {
    int64_t netId = selected->records.getPipeNetId();
    std::string name = state.netName(netId);
    std::string msg;
    llvm::raw_string_ostream(msg)
        << "this `ttl.copy(buffer, pipe)` sends data on PipeNet " << name
        << " from a node that is not a source of any pipe in that net; "
           "wrap the copy in `"
        << name << ".if_src(...)` or guard with `if " << name
        << ".is_src(): ...`";
    checkKnownSubset(
        copyOp, current,
        getPipeRecordsRoleLaunchNodeDomain(selected->records, PipeRole::Source),
        unanalyzable, msg, {{netId, PipeRole::Source}}, state);
    return;
  }
  if (auto srcPipeType = mlir::dyn_cast<PipeType>(copyOp.getSrc().getType())) {
    int64_t netId = srcPipeType.getPipeNetId();
    std::string name = state.netName(netId);
    std::string msg;
    llvm::raw_string_ostream(msg)
        << "this `ttl.copy(pipe, buffer)` receives data from PipeNet " << name
        << " on a node that is not a destination of any pipe in that "
           "net; wrap the copy in `"
        << name << ".if_dst(...)` or guard with `if " << name
        << ".is_dst(): ...`";
    checkKnownSubset(
        copyOp, current, getPipeDestinationLaunchNodeDomain(srcPipeType),
        unanalyzable, msg, {{netId, PipeRole::Destination}}, state);
    return;
  }
  if (std::optional<SelectedPipeRecords> selected =
          getSelectedDestinationRecords(copyOp.getSrc())) {
    int64_t netId = selected->records.getPipeNetId();
    std::string name = state.netName(netId);
    std::string msg;
    llvm::raw_string_ostream(msg)
        << "this `ttl.copy(pipe, buffer)` receives data from PipeNet " << name
        << " on a node that is not a destination of any pipe in that "
           "net; wrap the copy in `"
        << name << ".if_dst(...)` or guard with `if " << name
        << ".is_dst(): ...`";
    checkKnownSubset(copyOp, current,
                     getPipeRecordsRoleLaunchNodeDomain(selected->records,
                                                        PipeRole::Destination),
                     unanalyzable, msg, {{netId, PipeRole::Destination}},
                     state);
  }
}

/// Verify that a `ttl.pipenet_scope` body only executes on nodes participating
/// in at least one selected PipeNet role.
void verifyPipeNetScope(PipeNetScopeOp scopeOp, const LaunchNodeDomain &domain,
                        const PipeNetScopeLaunchNodeDomains &scope,
                        ModuleState &state) {
  std::string msg;
  {
    llvm::raw_string_ostream os(msg);
    SmallVector<int64_t> uniqueIds;
    for (auto &role : scope.roles) {
      if (!llvm::is_contained(uniqueIds, role.first)) {
        uniqueIds.push_back(role.first);
      }
    }
    os << "this region exchanges data on PipeNet";
    if (uniqueIds.size() != 1) {
      os << "s";
    }
    os << " ";
    llvm::interleaveComma(uniqueIds, os,
                          [&](int64_t id) { os << state.netName(id); });
    os << " on launched nodes that are not part of "
       << (uniqueIds.size() == 1 ? "that net" : "those nets")
       << "; wrap the surrounding work in `if "
       << formatGuardExpression(scope.roles, state)
       << ": ...` so non-participating nodes skip it";
  }
  checkKnownSubset(scopeOp, domain, scope.domain,
                   /*unanalyzableOp=*/nullptr, msg, scope.roles, state);
}

/// Dispatch the generic launch-domain callback to the checks that care about
/// a specific operation kind.
void recordGuardOperation(Operation *op, const LaunchNodeDomain &domain,
                          Operation *unanalyzableOp, ModuleState &state) {
  TypeSwitch<Operation *>(op)
      .Case<CopyOp>([&](CopyOp copy) {
        verifyCopy(copy, domain, unanalyzableOp, state);
        state.recordPipeEvent(copy, domain);
      })
      .Case<WaitOp>([&](WaitOp wait) {
        state.recordPipeWaitEvent(wait, domain, unanalyzableOp);
      })
      .Case<CBPushOp>([&](CBPushOp push) {
        FailureOr<int64_t> dfbId = getDFBId(push.getCb());
        assert(succeeded(dfbId) && "DFB identities were verified");
        state.dfbProducerDomains[*dfbId] =
            state.dfbProducerDomains[*dfbId].unionWith(domain);
      })
      .Case<CBWaitOp>([&](CBWaitOp wait) {
        FailureOr<int64_t> dfbId = getDFBId(wait.getCb());
        assert(succeeded(dfbId) && "DFB identities were verified");
        state.waitUses.push_back({wait, domain, *dfbId});
      });
}

// Cross-check each recorded `cb_wait` against the producer domain collected
// for the same dataflow buffer. Errors when the wait's lattice domain is not
// covered by any producer (deadlock-prone IR).
void verifyCBWaits(ModuleState &state) {
  for (WaitUse &use : state.waitUses) {
    auto it = state.dfbProducerDomains.find(use.dfbId);
    if (it == state.dfbProducerDomains.end()) {
      use.op.emitOpError()
          << "this `cb_wait` reads from a dataflow buffer that no other "
             "thread fills; check that another `@ttl.compute()` or "
             "`@ttl.datamovement()` thread reserves and pushes the same "
             "buffer";
      state.sawError = true;
      continue;
    }
    checkKnownSubset(use.op, use.domain, it->second,
                     /*unanalyzableOp=*/nullptr,
                     "this `cb_wait` runs on launched nodes where no "
                     "thread pushes data to the buffer (would deadlock); "
                     "guard the wait with the same `if net.is_active(): "
                     "...` predicate the producer uses",
                     /*roles=*/{}, state);
  }
}

enum class PipeScheduleNodeKind { Send, ReceivePost, ReceiveWait };

enum class PipeScheduleEdgeKind {
  ProgramOrder,
  ReceivePostEnablesSend,
  SendCompletesReceive
};

using PipeScheduleNodeId = std::size_t;

/// Directed wait-for edge in the pipe schedule graph.
struct PipeScheduleEdge {
  PipeScheduleNodeId successor;
  PipeScheduleEdgeKind kind;
};

/// Pipe synchronization event specialized to one launch node.
struct PipeScheduleNode {
  Operation *op;
  PipeType pipeType;
  LaunchNodeCoord coord;
  PipeScheduleNodeKind kind;
  std::optional<std::uint64_t> executionCountDivisor;
  SmallVector<PipeScheduleEdge> successors;
};

/// Retain every send for one logical pipe in IR order so receiver posts can be
/// paired with the corresponding send.
struct PipeSendOccurrences {
  PipeType pipeType;
  SmallVector<PipeScheduleNodeId> nodes;
};

using PipeIdentity =
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

using PipeCoordIdentity =
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
               int64_t, int64_t>;

/// Return a stable identity for one pipe endpoint relation.
PipeIdentity getPipeIdentity(PipeType pipeType) {
  return {pipeType.getPipeNetId(), pipeType.getSrcX(),
          pipeType.getSrcY(),      pipeType.getDstStartX(),
          pipeType.getDstEndX(),   pipeType.getDstStartY(),
          pipeType.getDstEndY()};
}

/// Return a stable identity for one pipe endpoint relation at one launch node.
PipeCoordIdentity getPipeCoordIdentity(PipeType pipeType,
                                       LaunchNodeCoord coord) {
  auto [pipeNetId, srcX, srcY, dstStartX, dstEndX, dstStartY, dstEndY] =
      getPipeIdentity(pipeType);
  return {pipeNetId, srcX,    srcY,    dstStartX, dstEndX,
          dstStartY, dstEndY, coord.x, coord.y};
}

/// Use distinct nodes so identical record coordinates still retain record
/// order in the wait-for graph.
PipeScheduleNodeId
addPipeScheduleNode(SmallVectorImpl<PipeScheduleNode> &nodes, Operation *op,
                    PipeType pipeType, LaunchNodeCoord coord,
                    PipeScheduleNodeKind kind,
                    std::optional<std::uint64_t> executionCountDivisor) {
  PipeScheduleNodeId nodeId = nodes.size();
  nodes.push_back({op, pipeType, coord, kind, executionCountDivisor, {}});
  return nodeId;
}

/// Add a directed graph edge unless the same typed edge already exists.
void addPipeScheduleEdge(SmallVectorImpl<PipeScheduleNode> &nodes,
                         PipeScheduleNodeId predecessor,
                         PipeScheduleNodeId successor,
                         PipeScheduleEdgeKind kind) {
  SmallVectorImpl<PipeScheduleEdge> &successors = nodes[predecessor].successors;
  if (!llvm::any_of(successors, [&](const PipeScheduleEdge &edge) {
        return edge.successor == successor && edge.kind == kind;
      })) {
    successors.push_back({successor, kind});
  }
}

/// Find any directed cycle in the pipe schedule graph.
std::optional<SmallVector<PipeScheduleNodeId>>
findPipeScheduleCycle(ArrayRef<PipeScheduleNode> nodes) {
  SmallVector<PipeScheduleNodeId> stack;
  SmallVector<PipeScheduleNodeId> cycle;
  SmallVector<uint8_t> colors(nodes.size(), 0);

  std::function<bool(PipeScheduleNodeId)> visit =
      [&](PipeScheduleNodeId nodeId) {
        colors[nodeId] = 1;
        stack.push_back(nodeId);
        for (const PipeScheduleEdge &edge : nodes[nodeId].successors) {
          PipeScheduleNodeId successor = edge.successor;
          if (colors[successor] == 0) {
            if (visit(successor)) {
              return true;
            }
            continue;
          }
          if (colors[successor] != 1) {
            continue;
          }
          auto cycleStart = llvm::find(stack, successor);
          cycle.append(cycleStart, stack.end());
          cycle.push_back(successor);
          return true;
        }
        stack.pop_back();
        colors[nodeId] = 2;
        return false;
      };

  for (PipeScheduleNodeId nodeId = 0, count = nodes.size(); nodeId < count;
       ++nodeId) {
    if (colors[nodeId] == 0 && visit(nodeId)) {
      return cycle;
    }
  }
  return std::nullopt;
}

/// Return the first edge kind between two schedule nodes, if present.
std::optional<PipeScheduleEdgeKind>
getPipeScheduleEdgeKind(ArrayRef<PipeScheduleNode> nodes,
                        PipeScheduleNodeId predecessor,
                        PipeScheduleNodeId successor) {
  for (const PipeScheduleEdge &edge : nodes[predecessor].successors) {
    if (edge.successor == successor) {
      return edge.kind;
    }
  }
  return std::nullopt;
}

/// Return true if a reported cycle contains the requested typed edge.
bool cycleContainsEdge(ArrayRef<PipeScheduleNode> nodes,
                       ArrayRef<PipeScheduleNodeId> cycle,
                       PipeScheduleNodeId predecessor,
                       PipeScheduleNodeId successor,
                       PipeScheduleEdgeKind kind) {
  for (std::size_t idx = 0, count = cycle.size() - 1; idx < count; ++idx) {
    if (cycle[idx] != predecessor || cycle[idx + 1] != successor) {
      continue;
    }
    std::optional<PipeScheduleEdgeKind> maybeActualKind =
        getPipeScheduleEdgeKind(nodes, predecessor, successor);
    if (maybeActualKind && *maybeActualKind == kind) {
      return true;
    }
  }
  return false;
}

/// Return true if a section of a reported cycle is entirely program order.
bool cycleHasProgramOrderPath(ArrayRef<PipeScheduleNode> nodes,
                              ArrayRef<PipeScheduleNodeId> cycle,
                              std::size_t startCycleIndex,
                              std::size_t endCycleIndex) {
  assert(startCycleIndex < endCycleIndex &&
         "expected a forward range within the reported cycle");
  for (std::size_t idx = startCycleIndex; idx < endCycleIndex; ++idx) {
    std::optional<PipeScheduleEdgeKind> maybeEdgeKind =
        getPipeScheduleEdgeKind(nodes, cycle[idx], cycle[idx + 1]);
    if (!maybeEdgeKind ||
        *maybeEdgeKind != PipeScheduleEdgeKind::ProgramOrder) {
      return false;
    }
  }
  return true;
}

/// Render a schedule node as a diagnostic phrase.
std::string describePipeScheduleNode(const PipeScheduleNode &node) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  switch (node.kind) {
  case PipeScheduleNodeKind::Send:
    os << "send";
    break;
  case PipeScheduleNodeKind::ReceivePost:
    os << "receiver post";
    break;
  case PipeScheduleNodeKind::ReceiveWait:
    os << "receive completion";
    break;
  }
  os << " at core_x=" << node.coord.x << ", core_y=" << node.coord.y;
  return buffer;
}

/// Render a wait-for edge as a diagnostic explanation.
std::string describePipeScheduleEdge(const PipeScheduleNode &predecessor,
                                     const PipeScheduleNode &successor,
                                     PipeScheduleEdgeKind kind) {
  std::string buffer;
  llvm::raw_string_ostream os(buffer);
  switch (kind) {
  case PipeScheduleEdgeKind::ProgramOrder:
    os << "program order requires " << describePipeScheduleNode(successor)
       << " after " << describePipeScheduleNode(predecessor);
    break;
  case PipeScheduleEdgeKind::ReceivePostEnablesSend:
    os << "sender waits for " << describePipeScheduleNode(predecessor)
       << " before " << describePipeScheduleNode(successor);
    break;
  case PipeScheduleEdgeKind::SendCompletesReceive:
    os << describePipeScheduleNode(successor) << " waits for "
       << describePipeScheduleNode(predecessor) << " to transfer data";
    break;
  }
  return buffer;
}

/// Identify the common single-thread bug where a receive wait is ordered before
/// the send that can complete it.
std::optional<std::pair<PipeScheduleNodeId, PipeScheduleNodeId>>
findReceiveWaitBeforeCompletingSend(ArrayRef<PipeScheduleNode> nodes,
                                    ArrayRef<PipeScheduleNodeId> cycle) {
  std::size_t cycleNodeCount = cycle.size() - 1;
  for (std::size_t waitIdx = 0; waitIdx < cycleNodeCount; ++waitIdx) {
    PipeScheduleNodeId waitNodeId = cycle[waitIdx];
    const PipeScheduleNode &waitNode = nodes[waitNodeId];
    if (waitNode.kind != PipeScheduleNodeKind::ReceiveWait) {
      continue;
    }
    for (std::size_t sendIdx = waitIdx + 1; sendIdx < cycle.size(); ++sendIdx) {
      PipeScheduleNodeId sendNodeId = cycle[sendIdx];
      const PipeScheduleNode &sendNode = nodes[sendNodeId];
      if (sendNode.kind != PipeScheduleNodeKind::Send) {
        continue;
      }
      if (!cycleHasProgramOrderPath(nodes, cycle, waitIdx, sendIdx)) {
        continue;
      }
      if (cycleContainsEdge(nodes, cycle, sendNodeId, waitNodeId,
                            PipeScheduleEdgeKind::SendCompletesReceive)) {
        return std::make_pair(waitNodeId, sendNodeId);
      }
    }
  }
  return std::nullopt;
}

/// Identify the common single-thread bug where a send is ordered before the
/// receive post that enables it.
std::optional<std::pair<PipeScheduleNodeId, PipeScheduleNodeId>>
findSendBeforeReceivePost(ArrayRef<PipeScheduleNode> nodes,
                          ArrayRef<PipeScheduleNodeId> cycle) {
  std::size_t cycleNodeCount = cycle.size() - 1;
  for (std::size_t sendIdx = 0; sendIdx < cycleNodeCount; ++sendIdx) {
    PipeScheduleNodeId sendNodeId = cycle[sendIdx];
    const PipeScheduleNode &sendNode = nodes[sendNodeId];
    if (sendNode.kind != PipeScheduleNodeKind::Send) {
      continue;
    }
    for (std::size_t postIdx = sendIdx + 1; postIdx < cycle.size(); ++postIdx) {
      PipeScheduleNodeId postNodeId = cycle[postIdx];
      const PipeScheduleNode &postNode = nodes[postNodeId];
      if (postNode.kind != PipeScheduleNodeKind::ReceivePost) {
        continue;
      }
      if (!cycleHasProgramOrderPath(nodes, cycle, sendIdx, postIdx)) {
        continue;
      }
      if (cycleContainsEdge(nodes, cycle, postNodeId, sendNodeId,
                            PipeScheduleEdgeKind::ReceivePostEnablesSend)) {
        return std::make_pair(sendNodeId, postNodeId);
      }
    }
  }
  return std::nullopt;
}

/// Attach a bounded set of edge notes for a reported schedule cycle.
void emitPipeScheduleCycleNotes(InFlightDiagnostic &diag,
                                ArrayRef<PipeScheduleNode> nodes,
                                ArrayRef<PipeScheduleNodeId> cycle) {
  for (std::size_t idx = 0, count = cycle.size() - 1; idx < count; ++idx) {
    PipeScheduleNodeId predecessorId = cycle[idx];
    PipeScheduleNodeId successorId = cycle[idx + 1];
    std::optional<PipeScheduleEdgeKind> maybeEdgeKind =
        getPipeScheduleEdgeKind(nodes, predecessorId, successorId);
    if (!maybeEdgeKind) {
      continue;
    }
    const PipeScheduleNode &predecessor = nodes[predecessorId];
    const PipeScheduleNode &successor = nodes[successorId];
    diag.attachNote(successor.op->getLoc())
        << describePipeScheduleEdge(predecessor, successor, *maybeEdgeKind);
    if (idx + 1 >= kMaxPipeScheduleCycleNotes) {
      break;
    }
  }
}

/// Emit the most specific diagnostic available for a pipe schedule cycle.
void emitPipeScheduleCycleDiagnostic(ArrayRef<PipeScheduleNode> nodes,
                                     ArrayRef<PipeScheduleNodeId> cycle,
                                     ModuleState &state) {
  if (auto waitBeforeSend = findReceiveWaitBeforeCompletingSend(nodes, cycle)) {
    const PipeScheduleNode &waitNode = nodes[waitBeforeSend->first];
    const PipeScheduleNode &sendNode = nodes[waitBeforeSend->second];
    auto diag = waitNode.op->emitOpError()
                << "receive wait occurs before the send that completes it on "
                   "PipeNet "
                << state.netName(waitNode.pipeType.getPipeNetId());
    diag.attachNote(waitNode.op->getLoc())
        << "this wait blocks until the sender transfers into the posted "
           "destination dataflow buffer slot";
    diag.attachNote(sendNode.op->getLoc())
        << "this send is ordered after the wait in the same data-movement "
           "thread";
    diag.attachNote(waitNode.op->getLoc())
        << "move the receive wait after the send, or place send and receive in "
           "separate data-movement threads";
    state.sawError = true;
    return;
  }

  if (auto sendBeforePost = findSendBeforeReceivePost(nodes, cycle)) {
    const PipeScheduleNode &sendNode = nodes[sendBeforePost->first];
    const PipeScheduleNode &postNode = nodes[sendBeforePost->second];
    auto diag = sendNode.op->emitOpError()
                << "pipe send occurs before the receiver posts a dataflow "
                   "buffer reservation on PipeNet "
                << state.netName(sendNode.pipeType.getPipeNetId());
    diag.attachNote(sendNode.op->getLoc())
        << "this send waits for each destination to post "
           "`ttl.copy(pipe, dst)`";
    diag.attachNote(postNode.op->getLoc())
        << "this receiver post is ordered after the send in the same "
           "data-movement thread";
    diag.attachNote(sendNode.op->getLoc())
        << "move `ttl.copy(pipe, dst)` before the dependent send, or place "
           "send "
           "and receive in separate data-movement threads";
    state.sawError = true;
    return;
  }

  PipeScheduleNode node = nodes[cycle.front()];
  auto diag = node.op->emitOpError()
              << "pipe schedule contains a wait-for cycle on PipeNet "
              << state.netName(node.pipeType.getPipeNetId())
              << "; post the receive before the dependent send, or place the "
                 "send and receive in separate data-movement threads";

  emitPipeScheduleCycleNotes(diag, nodes, cycle);
  state.sawError = true;
}

/// Normalize a foreach operation's total execution count to one selected
/// record occurrence.
static std::optional<std::uint64_t>
getPipeScheduleNodeExecutionCount(const PipeScheduleNode &node,
                                  ModuleState &state) {
  std::optional<std::uint64_t> totalCount =
      getExactExecutionCountAtLaunchNode(node.op, node.coord, state);
  if (!totalCount) {
    return totalCount;
  }
  if (!node.executionCountDivisor) {
    return std::nullopt;
  }
  if (*node.executionCountDivisor == 1) {
    return totalCount;
  }

  if (*totalCount % *node.executionCountDivisor != 0) {
    return std::nullopt;
  }
  // ExecutionCountAnalysis counts every matching foreach record. This node
  // represents one callback execution for one matching record combination.
  return *totalCount / *node.executionCountDivisor;
}

/// Pair predecessor and successor operations at the same IR-order position.
/// Each pair may execute repeatedly, so its operations must execute equally
/// often under equivalent conditions.
LogicalResult addPipeOccurrenceEdges(SmallVectorImpl<PipeScheduleNode> &nodes,
                                     ArrayRef<PipeScheduleNodeId> predecessors,
                                     ArrayRef<PipeScheduleNodeId> successors,
                                     PipeScheduleEdgeKind kind,
                                     StringRef predecessorName,
                                     StringRef successorName,
                                     LaunchNodeCoord receiverCoord,
                                     ModuleState &state) {
  assert(!predecessors.empty() && !successors.empty() &&
         "schedule correspondence requires both event kinds");
  if (predecessors.size() != successors.size()) {
    bool hasExtraPredecessor = predecessors.size() > successors.size();
    PipeScheduleNodeId unmatchedNode = hasExtraPredecessor
                                           ? predecessors[successors.size()]
                                           : successors[predecessors.size()];
    StringRef unmatchedName =
        hasExtraPredecessor ? predecessorName : successorName;
    StringRef missingName =
        hasExtraPredecessor ? successorName : predecessorName;
    auto diag =
        nodes[unmatchedNode].op->emitOpError()
        << "PipeNet "
        << state.netName(nodes[unmatchedNode].pipeType.getPipeNetId())
        << " requires one " << predecessorName << " operation for each "
        << successorName << " operation at receiver core_x=" << receiverCoord.x
        << ", core_y=" << receiverCoord.y << "; found " << predecessors.size()
        << " " << predecessorName << " operation(s) and " << successors.size()
        << " " << successorName << " operation(s)";
    diag.attachNote(nodes[unmatchedNode].op->getLoc())
        << "this " << unmatchedName << " operation has no corresponding "
        << missingName << " operation";
    state.sawError = true;
    return failure();
  }
  for (auto [predecessor, successor] : llvm::zip(predecessors, successors)) {
    bool haveEqualExecutionCounts = false;
    if (nodes[predecessor].executionCountDivisor != 1 ||
        nodes[successor].executionCountDivisor != 1) {
      std::optional<std::uint64_t> predecessorCount =
          getPipeScheduleNodeExecutionCount(nodes[predecessor], state);
      std::optional<std::uint64_t> successorCount =
          getPipeScheduleNodeExecutionCount(nodes[successor], state);
      haveEqualExecutionCounts = predecessorCount && successorCount &&
                                 *predecessorCount == *successorCount;
    } else {
      haveEqualExecutionCounts = proveEqualExecutionCountAtLaunchNodes(
          nodes[predecessor].op, nodes[predecessor].coord, nodes[successor].op,
          nodes[successor].coord, state);
    }
    if (!haveEqualExecutionCounts) {
      auto diag = nodes[successor].op->emitOpError()
                  << "cannot prove a one-to-one synchronization schedule on "
                     "PipeNet "
                  << state.netName(nodes[successor].pipeType.getPipeNetId())
                  << " for receiver core_x=" << receiverCoord.x
                  << ", core_y=" << receiverCoord.y << "; " << predecessorName
                  << " and " << successorName
                  << " occurrences do not have matching proven execution "
                     "counts and conditions";
      diag.attachNote(nodes[predecessor].op->getLoc())
          << "matching " << predecessorName << " occurrence is here";
      state.sawError = true;
      return failure();
    }
    addPipeScheduleEdge(nodes, predecessor, successor, kind);
  }
  return success();
}

using ActiveForeachRecord = std::pair<Operation *, int64_t>;

static std::optional<int64_t>
getActiveForeachRecordIndex(ArrayRef<ActiveForeachRecord> activeRecords,
                            Operation *foreachOp) {
  auto activeIt = llvm::find_if(llvm::reverse(activeRecords),
                                [&](const ActiveForeachRecord &active) {
                                  return active.first == foreachOp;
                                });
  if (activeIt == activeRecords.rend()) {
    return std::nullopt;
  }
  return activeIt->second;
}

/// Visit pipe events in the order they execute at `coord`.
///
/// A PipeNet foreach body executes once to completion for each matching record.
/// Recursing through each record before continuing after the foreach preserves
/// that order in the wait-for graph.
static void walkPipeEventsInProgramOrder(
    Operation *op, LaunchNodeCoord coord, const ModuleState &state,
    SmallVectorImpl<ActiveForeachRecord> &activeRecords,
    std::optional<std::uint64_t> executionCountDivisor,
    llvm::function_ref<void(const PipeEvent &, std::optional<std::uint64_t>)>
        visitEvent) {
  PipeNetRecordsAttr foreachRecords;
  PipeRole foreachRole = PipeRole::Active;
  if (auto foreachSrc = mlir::dyn_cast<PipeNetForeachSrcOp>(op)) {
    foreachRecords = foreachSrc.getRecords();
    foreachRole = PipeRole::Source;
  } else if (auto foreachDst = mlir::dyn_cast<PipeNetForeachDstOp>(op)) {
    foreachRecords = foreachDst.getRecords();
    foreachRole = PipeRole::Destination;
  }

  if (foreachRecords) {
    SmallVector<int64_t> matchingRecordIndices;
    for (auto [recordIndex, record] :
         llvm::enumerate(foreachRecords.getPipes())) {
      LaunchNodeDomain recordDomain =
          foreachRole == PipeRole::Source
              ? getPipeRecordSourceLaunchNodeDomain(record)
              : getPipeRecordDestinationLaunchNodeDomain(record);
      if (knownLaunchNodeDomainContains(recordDomain, coord)) {
        matchingRecordIndices.push_back(static_cast<int64_t>(recordIndex));
      }
    }

    std::optional<std::uint64_t> nestedExecutionCountDivisor =
        executionCountDivisor;
    if (executionCountDivisor && !matchingRecordIndices.empty()) {
      nestedExecutionCountDivisor = llvm::checkedMulUnsigned(
          *executionCountDivisor,
          static_cast<std::uint64_t>(matchingRecordIndices.size()));
    }

    for (int64_t recordIndex : matchingRecordIndices) {
      activeRecords.emplace_back(op, recordIndex);
      for (Region &region : op->getRegions()) {
        for (Block &block : region) {
          for (Operation &nestedOp : block) {
            walkPipeEventsInProgramOrder(&nestedOp, coord, state, activeRecords,
                                         nestedExecutionCountDivisor,
                                         visitEvent);
          }
        }
      }
      activeRecords.pop_back();
    }
    return;
  }

  auto eventIt = state.pipeEventIndices.find(op);
  if (eventIt != state.pipeEventIndices.end()) {
    for (std::size_t eventIndex : eventIt->second) {
      const PipeEvent &event = state.pipeEvents[eventIndex];
      if (!knownLaunchNodeDomainContains(event.domain, coord)) {
        continue;
      }
      if (event.selectedForeachOp) {
        std::optional<int64_t> activeRecordIndex =
            getActiveForeachRecordIndex(activeRecords, event.selectedForeachOp);
        assert(activeRecordIndex &&
               "selected pipe event must be nested in its foreach operation");
        if (*activeRecordIndex != event.selectedRecordIndex) {
          continue;
        }
      }
      visitEvent(event, executionCountDivisor);
    }
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        walkPipeEventsInProgramOrder(&nestedOp, coord, state, activeRecords,
                                     executionCountDivisor, visitEvent);
      }
    }
  }
}

// Verify synchronization dependencies implied by pipe operations. Receive-side
// ttl.copy makes a reserved DFB slot available to the sender, while ttl.wait on
// its handle waits for payload arrival. Modeling availability and completion as
// separate events preserves asynchronous semantics and detects wait-for cycles.
void verifyPipeScheduleCycles(ModuleOp module, ModuleState &state) {
  SmallVector<PipeScheduleNode> nodes;
  // Preserve program order so cycle selection and diagnostics are
  // deterministic.
  llvm::MapVector<PipeIdentity, PipeSendOccurrences> sendOccurrences;
  llvm::DenseMap<PipeCoordIdentity, SmallVector<PipeScheduleNodeId>>
      receivePostNodes;
  llvm::DenseMap<PipeCoordIdentity, SmallVector<PipeScheduleNodeId>>
      receiveWaitNodes;

  module.walk([&](func::FuncOp funcOp) {
    for (LaunchNodeCoord coord : state.baseDomain.nodes) {
      SmallVector<ActiveForeachRecord> activeRecords;
      std::optional<PipeScheduleNodeId> lastNode;
      walkPipeEventsInProgramOrder(
          funcOp, coord, state, activeRecords,
          /*executionCountDivisor=*/1,
          [&](const PipeEvent &event,
              std::optional<std::uint64_t> executionCountDivisor) {
            PipeScheduleNodeKind nodeKind;
            if (event.kind == PipeEventKind::Send) {
              nodeKind = PipeScheduleNodeKind::Send;
            } else if (event.kind == PipeEventKind::ReceivePost) {
              nodeKind = PipeScheduleNodeKind::ReceivePost;
            } else {
              nodeKind = PipeScheduleNodeKind::ReceiveWait;
            }

            PipeScheduleNodeId nodeId =
                addPipeScheduleNode(nodes, event.op, event.pipeType, coord,
                                    nodeKind, executionCountDivisor);

            if (event.kind == PipeEventKind::Send) {
              PipeIdentity identity = getPipeIdentity(event.pipeType);
              auto sendIt =
                  sendOccurrences
                      .try_emplace(identity,
                                   PipeSendOccurrences{event.pipeType, {}})
                      .first;
              sendIt->second.nodes.push_back(nodeId);
            } else if (event.kind == PipeEventKind::ReceivePost) {
              receivePostNodes[getPipeCoordIdentity(event.pipeType, coord)]
                  .push_back(nodeId);
            } else {
              receiveWaitNodes[getPipeCoordIdentity(event.pipeType, coord)]
                  .push_back(nodeId);
            }

            if (lastNode) {
              addPipeScheduleEdge(nodes, *lastNode, nodeId,
                                  PipeScheduleEdgeKind::ProgramOrder);
            }
            lastNode = nodeId;
          });
    }
  });

  for (const auto &[pipeIdentity, sends] : sendOccurrences) {
    LaunchNodeDomain destinations =
        getPipeDestinationLaunchNodeDomain(sends.pipeType);
    for (LaunchNodeCoord coord : destinations.nodes) {
      PipeCoordIdentity identity = getPipeCoordIdentity(sends.pipeType, coord);
      auto postIt = receivePostNodes.find(identity);
      if (postIt != receivePostNodes.end()) {
        if (failed(addPipeOccurrenceEdges(
                nodes, postIt->second, sends.nodes,
                PipeScheduleEdgeKind::ReceivePostEnablesSend, "receiver post",
                "send", coord, state))) {
          continue;
        }
      }
      auto waitIt = receiveWaitNodes.find(identity);
      if (waitIt != receiveWaitNodes.end()) {
        if (failed(addPipeOccurrenceEdges(
                nodes, sends.nodes, waitIt->second,
                PipeScheduleEdgeKind::SendCompletesReceive, "send",
                "receive wait", coord, state))) {
          continue;
        }
      }
    }
  }

  if (std::optional<SmallVector<PipeScheduleNodeId>> maybeCycle =
          findPipeScheduleCycle(nodes)) {
    emitPipeScheduleCycleDiagnostic(nodes, *maybeCycle, state);
  }
}

// Reject predicates and scopes that reference no static or record-table
// declaration, since an empty role set would otherwise accept any nested work.
void validatePipeNetReferences(ModuleOp module, ModuleState &state) {
  module.walk([&](Operation *op) {
    auto report = [&](int64_t netId) {
      op->emitOpError() << "references unknown PipeNet " << state.netName(netId)
                        << " (id " << netId
                        << "); no pipe or PipeNet record table declares this "
                           "net";
      state.sawError = true;
    };
    if (auto pred = mlir::dyn_cast<PipeNetPredicateOpInterface>(op)) {
      if (!state.pipeNetLocs.count(pred.getReferencedPipeNetId())) {
        report(pred.getReferencedPipeNetId());
      }
      return;
    }
    if (auto scopeOp = mlir::dyn_cast<PipeNetScopeOp>(op)) {
      SmallVector<int64_t> ids;
      if (readPipeNetScopeIds(scopeOp, ids)) {
        for (int64_t id : ids) {
          if (!state.pipeNetLocs.count(id)) {
            report(id);
          }
        }
      }
    }
  });
}

struct TTLVerifyPipeNetGuardsPass
    : impl::TTLVerifyPipeNetGuardsBase<TTLVerifyPipeNetGuardsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    ModuleState state(module);
    state.initialize(module);
    if (state.hasPipes() && !state.hasLaunchGrid) {
      module.emitError()
          << "ttl-verify-pipenet-guards requires a `ttl.launch_grid` "
             "module attribute (an i64 array of length 2 with positive "
             "entries)";
      signalPassFailure();
      return;
    }
    if (!state.hasPipes()) {
      return;
    }
    if (failed(verifyResolvedDFBIdentities(module, getArgument()))) {
      signalPassFailure();
      return;
    }

    validatePipeNetReferences(module, state);
    if (state.sawError) {
      signalPassFailure();
      return;
    }

    // Kernel-thread `func.func`s are runtime-invoked entry points with no
    // callers (so they are analysis roots and get `setToEntryState`); helpers
    // they call have the caller's narrowed lattice flow through `func.call`.
    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    LaunchNodeDomainAnalysisOptions options;
    options.operationCallback = [&](Operation *op,
                                    const LaunchNodeDomain &domain,
                                    Operation *unanalyzableOp) {
      recordGuardOperation(op, domain, unanalyzableOp, state);
    };
    options.pipeNetScopeCallback =
        [&](PipeNetScopeOp scopeOp, const LaunchNodeDomain &domain,
            Operation * /*unanalyzableOp*/,
            const PipeNetScopeLaunchNodeDomains &scope) {
          verifyPipeNetScope(scopeOp, domain, scope, state);
        };
    solver.load<LaunchNodeDomainAnalysis>(state, options);
    if (failed(solver.initializeAndRun(module))) {
      signalPassFailure();
      return;
    }

    verifyCBWaits(state);
    if (!state.sawError) {
      verifyPipeScheduleCycles(module, state);
    }

    if (state.sawError) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
