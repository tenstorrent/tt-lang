// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Verify that PipeNet-coupled operations execute only on launch nodes whose
// PipeNet roles permit them. The analysis is a `DenseForwardDataFlowAnalysis`
// whose lattice is the set of launch coordinates that can reach each program
// point. Predicate-bearing region ops (`scf.if`, `affine.if`, `ttl.if_src`,
// `ttl.if_dst`, `ttl.pipenet_scope`) narrow that set on region entry; pipe-
// coupled ops are checked against the narrowed set.
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <optional>
#include <tuple>

#define DEBUG_TYPE "ttl-verify-pipenet-guards"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLVERIFYPIPENETGUARDS
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

constexpr unsigned kMaxPipeScheduleCycleNotes = 8;

//===----------------------------------------------------------------------===//
// Module state collected before the analysis runs and updated during it.
//===----------------------------------------------------------------------===//

/// A dataflow buffer wait and the launch-node domain where it executes.
struct WaitUse {
  CBWaitOp op;
  LaunchNodeDomain domain;
  int64_t cbIndex;
};

/// Return true if `copyOp` publishes a destination dataflow buffer slot for a
/// pipe receive.
bool isPipeReceiveCopy(CopyOp copyOp) {
  return mlir::isa<PipeType>(copyOp.getSrc().getType()) &&
         getAttachedCB(copyOp.getDst());
}

/// Trace a `ttl.wait` transfer handle back to its receive-side pipe copy.
std::optional<CopyOp> findDefiningPipeReceiveCopy(Value value) {
  llvm::SmallPtrSet<Value, 16> seen;
  return traceTransferHandleSource<std::optional<CopyOp>>(
      value,
      [](Value source) {
        auto copyOp = source.getDefiningOp<CopyOp>();
        if (copyOp && isPipeReceiveCopy(copyOp)) {
          return std::optional<CopyOp>(copyOp);
        }
        return std::optional<CopyOp>();
      },
      seen);
}

/// Pipe synchronization event used by the wait-for graph verifier.
enum class PipeEventKind { Send, ReceivePost, ReceiveWait };

/// One pipe synchronization event on the launch-node domain where it executes.
struct PipeEvent {
  Operation *op = nullptr;
  PipeType pipeType;
  PipeEventKind kind;
  LaunchNodeDomain domain;
};

struct ModuleState;
void checkKnownSubset(Operation *op, const LaunchNodeDomain &current,
                      const LaunchNodeDomain &allowed,
                      Operation *unanalyzableOp, Twine primaryMessage,
                      ArrayRef<std::pair<int64_t, PipeRole>> roles,
                      ModuleState &state);

struct ModuleState : LaunchNodeDomainState {
  llvm::DenseMap<int64_t, LaunchNodeDomain> cbProducerDomains;
  SmallVector<WaitUse> waitUses;
  SmallVector<PipeEvent> pipeEvents;
  llvm::DenseMap<Operation *, unsigned> pipeEventIndices;

  /// Record pipe sends and receive posts from `ttl.copy` operations.
  void recordPipeEvent(CopyOp copyOp, const LaunchNodeDomain &domain) {
    PipeEvent event;
    event.op = copyOp.getOperation();
    if (auto pipeType = mlir::dyn_cast<PipeType>(copyOp.getDst().getType())) {
      event.pipeType = pipeType;
      event.kind = PipeEventKind::Send;
      event.domain =
          domain.intersectWith(getPipeSourceLaunchNodeDomain(pipeType));
    } else if (auto pipeType =
                   mlir::dyn_cast<PipeType>(copyOp.getSrc().getType())) {
      if (!isPipeReceiveCopy(copyOp)) {
        return;
      }
      event.pipeType = pipeType;
      event.kind = PipeEventKind::ReceivePost;
      event.domain =
          domain.intersectWith(getPipeDestinationLaunchNodeDomain(pipeType));
    } else {
      return;
    }

    auto [it, inserted] =
        pipeEventIndices.try_emplace(copyOp.getOperation(), pipeEvents.size());
    if (inserted) {
      pipeEvents.push_back(event);
      return;
    }
    pipeEvents[it->second] = event;
  }

  /// Record a receive completion wait and verify that it is
  /// destination-guarded.
  void recordPipeWaitEvent(WaitOp waitOp, const LaunchNodeDomain &domain,
                           Operation *unanalyzableOp) {
    std::optional<CopyOp> copyOp = findDefiningPipeReceiveCopy(waitOp.getXf());
    if (!copyOp.has_value()) {
      return;
    }
    auto pipeType = mlir::cast<PipeType>(copyOp->getSrc().getType());

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

    PipeEvent event;
    event.op = waitOp.getOperation();
    event.pipeType = pipeType;
    event.kind = PipeEventKind::ReceiveWait;
    event.domain =
        domain.intersectWith(getPipeDestinationLaunchNodeDomain(pipeType));

    auto [it, inserted] =
        pipeEventIndices.try_emplace(waitOp.getOperation(), pipeEvents.size());
    if (inserted) {
      pipeEvents.push_back(event);
      return;
    }
    pipeEvents[it->second] = event;
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
        if (auto cbIndex = getCBIndex(push.getCb())) {
          state.cbProducerDomains[*cbIndex] =
              state.cbProducerDomains[*cbIndex].unionWith(domain);
        }
      })
      .Case<CBWaitOp>([&](CBWaitOp wait) {
        if (auto cbIndex = getCBIndex(wait.getCb())) {
          state.waitUses.push_back({wait, domain, *cbIndex});
        }
      });
}

// Cross-check each recorded `cb_wait` against the producer domain collected
// for the same dataflow buffer. Errors when the wait's lattice domain is not
// covered by any producer (deadlock-prone IR).
void verifyCBWaits(ModuleState &state) {
  for (WaitUse &use : state.waitUses) {
    auto it = state.cbProducerDomains.find(use.cbIndex);
    if (it == state.cbProducerDomains.end()) {
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

/// Directed wait-for edge in the pipe schedule graph.
struct PipeScheduleEdge {
  unsigned successor;
  PipeScheduleEdgeKind kind;
};

/// Pipe synchronization event specialized to one launch node.
struct PipeScheduleNode {
  Operation *op;
  PipeType pipeType;
  LaunchNodeCoord coord;
  PipeScheduleNodeKind kind;
  SmallVector<PipeScheduleEdge> successors;
};

using PipeIdentity =
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>;

using PipeCoordIdentity =
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
               int64_t, int64_t>;

using PipeNodeIdentity = std::tuple<Operation *, int64_t, int64_t, int64_t>;

using ProgramPointIdentity = std::tuple<Operation *, int64_t, int64_t>;

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

/// Return a stable identity for one schedule node in the wait-for graph.
PipeNodeIdentity getPipeNodeIdentity(Operation *op, LaunchNodeCoord coord,
                                     PipeScheduleNodeKind kind) {
  return {op, coord.x, coord.y, static_cast<int64_t>(kind)};
}

/// Add or reuse the graph node for one pipe synchronization event.
unsigned
addPipeScheduleNode(SmallVectorImpl<PipeScheduleNode> &nodes,
                    llvm::DenseMap<PipeNodeIdentity, unsigned> &nodeIds,
                    Operation *op, PipeType pipeType, LaunchNodeCoord coord,
                    PipeScheduleNodeKind kind) {
  PipeNodeIdentity identity = getPipeNodeIdentity(op, coord, kind);
  auto [it, inserted] = nodeIds.try_emplace(identity, nodes.size());
  if (inserted) {
    nodes.push_back({op, pipeType, coord, kind, {}});
  }
  return it->second;
}

/// Add a directed graph edge unless the same typed edge already exists.
void addPipeScheduleEdge(SmallVectorImpl<PipeScheduleNode> &nodes,
                         unsigned predecessor, unsigned successor,
                         PipeScheduleEdgeKind kind) {
  SmallVectorImpl<PipeScheduleEdge> &successors = nodes[predecessor].successors;
  if (!llvm::any_of(successors, [&](const PipeScheduleEdge &edge) {
        return edge.successor == successor && edge.kind == kind;
      })) {
    successors.push_back({successor, kind});
  }
}

/// Find any directed cycle in the pipe schedule graph.
std::optional<SmallVector<unsigned>>
findPipeScheduleCycle(ArrayRef<PipeScheduleNode> nodes) {
  SmallVector<unsigned> stack;
  SmallVector<unsigned> cycle;
  SmallVector<uint8_t> colors(nodes.size(), 0);

  std::function<bool(unsigned)> visit = [&](unsigned nodeId) {
    colors[nodeId] = 1;
    stack.push_back(nodeId);
    for (const PipeScheduleEdge &edge : nodes[nodeId].successors) {
      unsigned successor = edge.successor;
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

  for (unsigned nodeId = 0, count = nodes.size(); nodeId < count; ++nodeId) {
    if (colors[nodeId] == 0 && visit(nodeId)) {
      return cycle;
    }
  }
  return std::nullopt;
}

/// Return the first edge kind between two schedule nodes, if present.
std::optional<PipeScheduleEdgeKind>
getPipeScheduleEdgeKind(ArrayRef<PipeScheduleNode> nodes, unsigned predecessor,
                        unsigned successor) {
  for (const PipeScheduleEdge &edge : nodes[predecessor].successors) {
    if (edge.successor == successor) {
      return edge.kind;
    }
  }
  return std::nullopt;
}

/// Return true if a reported cycle contains the requested typed edge.
bool cycleContainsEdge(ArrayRef<PipeScheduleNode> nodes,
                       ArrayRef<unsigned> cycle, unsigned predecessor,
                       unsigned successor, PipeScheduleEdgeKind kind) {
  for (unsigned idx = 0, count = cycle.size() - 1; idx < count; ++idx) {
    if (cycle[idx] != predecessor || cycle[idx + 1] != successor) {
      continue;
    }
    std::optional<PipeScheduleEdgeKind> actualKind =
        getPipeScheduleEdgeKind(nodes, predecessor, successor);
    if (actualKind && *actualKind == kind) {
      return true;
    }
  }
  return false;
}

/// Return true if a section of a reported cycle is entirely program order.
bool cycleHasProgramOrderPath(ArrayRef<PipeScheduleNode> nodes,
                              ArrayRef<unsigned> cycle,
                              unsigned startCycleIndex,
                              unsigned endCycleIndex) {
  assert(startCycleIndex < endCycleIndex &&
         "expected a forward range within the reported cycle");
  for (unsigned idx = startCycleIndex; idx < endCycleIndex; ++idx) {
    std::optional<PipeScheduleEdgeKind> edgeKind =
        getPipeScheduleEdgeKind(nodes, cycle[idx], cycle[idx + 1]);
    if (!edgeKind || *edgeKind != PipeScheduleEdgeKind::ProgramOrder) {
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
    os << "destination address publication";
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
std::optional<std::pair<unsigned, unsigned>>
findReceiveWaitBeforeCompletingSend(ArrayRef<PipeScheduleNode> nodes,
                                    ArrayRef<unsigned> cycle) {
  unsigned cycleNodeCount = cycle.size() - 1;
  for (unsigned waitIdx = 0; waitIdx < cycleNodeCount; ++waitIdx) {
    unsigned waitNodeId = cycle[waitIdx];
    const PipeScheduleNode &waitNode = nodes[waitNodeId];
    if (waitNode.kind != PipeScheduleNodeKind::ReceiveWait) {
      continue;
    }
    for (unsigned sendIdx = waitIdx + 1; sendIdx < cycle.size(); ++sendIdx) {
      unsigned sendNodeId = cycle[sendIdx];
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
std::optional<std::pair<unsigned, unsigned>>
findSendBeforeReceivePost(ArrayRef<PipeScheduleNode> nodes,
                          ArrayRef<unsigned> cycle) {
  unsigned cycleNodeCount = cycle.size() - 1;
  for (unsigned sendIdx = 0; sendIdx < cycleNodeCount; ++sendIdx) {
    unsigned sendNodeId = cycle[sendIdx];
    const PipeScheduleNode &sendNode = nodes[sendNodeId];
    if (sendNode.kind != PipeScheduleNodeKind::Send) {
      continue;
    }
    for (unsigned postIdx = sendIdx + 1; postIdx < cycle.size(); ++postIdx) {
      unsigned postNodeId = cycle[postIdx];
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
                                ArrayRef<unsigned> cycle) {
  for (unsigned idx = 0, count = cycle.size() - 1; idx < count; ++idx) {
    unsigned predecessorId = cycle[idx];
    unsigned successorId = cycle[idx + 1];
    std::optional<PipeScheduleEdgeKind> edgeKind =
        getPipeScheduleEdgeKind(nodes, predecessorId, successorId);
    if (!edgeKind) {
      continue;
    }
    const PipeScheduleNode &predecessor = nodes[predecessorId];
    const PipeScheduleNode &successor = nodes[successorId];
    diag.attachNote(successor.op->getLoc())
        << describePipeScheduleEdge(predecessor, successor, *edgeKind);
    if (idx + 1 >= kMaxPipeScheduleCycleNotes) {
      break;
    }
  }
}

/// Emit the most specific diagnostic available for a pipe schedule cycle.
void emitPipeScheduleCycleDiagnostic(ArrayRef<PipeScheduleNode> nodes,
                                     ArrayRef<unsigned> cycle,
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
                << "pipe send occurs before the receiver publishes a "
                   "destination address on PipeNet "
                << state.netName(sendNode.pipeType.getPipeNetId());
    diag.attachNote(sendNode.op->getLoc())
        << "this send waits for each destination to execute "
           "`ttl.copy(pipe, dst)`";
    diag.attachNote(postNode.op->getLoc())
        << "this destination address publication is ordered after the send in "
           "the "
           "same data-movement thread";
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

// Verify the hidden pipe synchronization introduced by receiver-advertised pipe
// lowering. Receive-side ttl.copy publishes the address; ttl.wait on that
// handle waits for completion. Modeling those as distinct events preserves
// async copy semantics while rejecting wait-for cycles.
void verifyPipeScheduleCycles(ModuleOp module, ModuleState &state) {
  SmallVector<PipeScheduleNode> nodes;
  SmallVector<std::pair<unsigned, PipeType>> sendNodes;
  llvm::DenseMap<PipeNodeIdentity, unsigned> nodeIds;
  llvm::DenseMap<PipeCoordIdentity, SmallVector<unsigned>> receivePostNodes;
  llvm::DenseMap<PipeCoordIdentity, SmallVector<unsigned>> receiveWaitNodes;
  llvm::DenseMap<ProgramPointIdentity, unsigned> lastCompletionNodes;

  module.walk([&](Operation *op) {
    auto eventIt = state.pipeEventIndices.find(op);
    if (eventIt == state.pipeEventIndices.end()) {
      return;
    }
    PipeEvent event = state.pipeEvents[eventIt->second];
    if (!event.domain.known || event.domain.nodes.empty()) {
      return;
    }

    auto funcOp = op->getParentOfType<func::FuncOp>();
    if (!funcOp) {
      return;
    }

    for (LaunchNodeCoord coord : event.domain.nodes) {
      PipeScheduleNodeKind nodeKind;
      if (event.kind == PipeEventKind::Send) {
        nodeKind = PipeScheduleNodeKind::Send;
      } else if (event.kind == PipeEventKind::ReceivePost) {
        nodeKind = PipeScheduleNodeKind::ReceivePost;
      } else {
        nodeKind = PipeScheduleNodeKind::ReceiveWait;
      }

      unsigned nodeId = addPipeScheduleNode(nodes, nodeIds, op, event.pipeType,
                                            coord, nodeKind);

      if (event.kind == PipeEventKind::Send) {
        sendNodes.push_back({nodeId, event.pipeType});
      } else if (event.kind == PipeEventKind::ReceivePost) {
        receivePostNodes[getPipeCoordIdentity(event.pipeType, coord)].push_back(
            nodeId);
      } else {
        receiveWaitNodes[getPipeCoordIdentity(event.pipeType, coord)].push_back(
            nodeId);
      }

      ProgramPointIdentity programPoint{funcOp.getOperation(), coord.x,
                                        coord.y};
      auto lastIt = lastCompletionNodes.find(programPoint);
      if (lastIt != lastCompletionNodes.end()) {
        addPipeScheduleEdge(nodes, lastIt->second, nodeId,
                            PipeScheduleEdgeKind::ProgramOrder);
      }
      lastCompletionNodes[programPoint] = nodeId;
    }
  });

  for (auto [sendNode, pipeType] : sendNodes) {
    LaunchNodeDomain destinations =
        getPipeDestinationLaunchNodeDomain(pipeType);
    for (LaunchNodeCoord coord : destinations.nodes) {
      PipeCoordIdentity identity = getPipeCoordIdentity(pipeType, coord);
      auto postIt = receivePostNodes.find(identity);
      if (postIt != receivePostNodes.end()) {
        for (unsigned receivePostNode : postIt->second) {
          addPipeScheduleEdge(nodes, receivePostNode, sendNode,
                              PipeScheduleEdgeKind::ReceivePostEnablesSend);
        }
      }
      auto waitIt = receiveWaitNodes.find(identity);
      if (waitIt != receiveWaitNodes.end()) {
        for (unsigned receiveWaitNode : waitIt->second) {
          addPipeScheduleEdge(nodes, sendNode, receiveWaitNode,
                              PipeScheduleEdgeKind::SendCompletesReceive);
        }
      }
    }
  }

  if (std::optional<SmallVector<unsigned>> cycle =
          findPipeScheduleCycle(nodes)) {
    emitPipeScheduleCycleDiagnostic(nodes, *cycle, state);
  }
}

// Walk the module and report any `pipenet_scope` or PipeNetPredicate that
// references a PipeNet id not declared by some `ttl.create_pipe`.
void validatePipeNetReferences(ModuleOp module, ModuleState &state) {
  module.walk([&](Operation *op) {
    auto report = [&](int64_t netId) {
      op->emitOpError() << "references unknown PipeNet " << state.netName(netId)
                        << " (id " << netId
                        << "); no `ttl.create_pipe` declares this net";
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

    ModuleState state;
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
