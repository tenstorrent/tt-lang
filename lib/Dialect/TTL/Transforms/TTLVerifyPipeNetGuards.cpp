// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Verify the launch-node domains and synchronization schedules of PipeNet
// operations. Launch-node dataflow determines where each operation can execute.
// The guard pass checks those domains against PipeNet roles. The schedule pass
// uses the same domains to prove event correspondence and reject wait-for
// cycles.
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "ttlang/Analysis/ValueOriginAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <tuple>

#define DEBUG_TYPE "ttl-verify-pipenet"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLVERIFYPIPENETGUARDS
#define GEN_PASS_DEF_TTLVERIFYPIPENETSCHEDULE
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

constexpr std::size_t kMaxPipeScheduleCycleNotes = 8;
constexpr std::size_t kMaxPipeScheduleNodes = 4096;

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

/// Return the diagnostic name for a pipe synchronization event.
StringRef getPipeEventName(PipeEventKind kind) {
  switch (kind) {
  case PipeEventKind::Send:
    return "send";
  case PipeEventKind::ReceivePost:
    return "receiver post";
  case PipeEventKind::ReceiveWait:
    return "receive wait";
  }
  llvm_unreachable("unknown pipe event kind");
}

/// One pipe synchronization event on the launch-node domain where it executes.
struct PipeEvent {
  Operation *op = nullptr;
  PipeType pipeType;
  PipeEventKind kind;
  LaunchNodeDomain domain;
  Operation *unanalyzableOp = nullptr;
};

struct ModuleState;
void checkKnownSubset(Operation *op, const LaunchNodeDomain &current,
                      const LaunchNodeDomain &allowed,
                      Operation *unanalyzableOp, Twine primaryMessage,
                      ArrayRef<std::pair<int64_t, PipeRole>> roles,
                      ModuleState &state);

struct ModuleState : LaunchNodeDomainState {
  explicit ModuleState(ValueOriginAnalysis &valueOrigins)
      : valueOrigins(valueOrigins) {}

  ValueOriginAnalysis &valueOrigins;
  llvm::DenseMap<int64_t, LaunchNodeDomain> cbProducerDomains;
  SmallVector<WaitUse> waitUses;
  SmallVector<PipeEvent> pipeEvents;
  llvm::DenseMap<Operation *, std::size_t> pipeEventIndices;

  /// Return the receive copied by a wait and emit the common ambiguity error.
  FailureOr<std::optional<CopyOp>> getPipeReceiveCopy(WaitOp waitOp) {
    FailureOr<std::optional<CopyOp>> maybeCopyOp =
        findDefiningPipeReceiveCopy(valueOrigins, waitOp.getXf());
    if (failed(maybeCopyOp)) {
      waitOp.emitOpError()
          << "requires either every possible source to be the same pipe "
             "receive ttl.copy or no source to be a pipe receive";
      sawError = true;
    }
    return maybeCopyOp;
  }

  /// Record pipe sends and receive posts from `ttl.copy` operations.
  void recordPipeEvent(CopyOp copyOp, const LaunchNodeDomain &domain,
                       Operation *unanalyzableOp) {
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
    event.unanalyzableOp = unanalyzableOp;

    auto [it, inserted] =
        pipeEventIndices.try_emplace(copyOp.getOperation(), pipeEvents.size());
    if (inserted) {
      pipeEvents.push_back(event);
      return;
    }
    pipeEvents[it->second] = event;
  }

  /// Record a receive completion wait for schedule verification.
  void recordPipeWaitEvent(WaitOp waitOp, const LaunchNodeDomain &domain,
                           Operation *unanalyzableOp) {
    FailureOr<std::optional<CopyOp>> maybeCopyOp = getPipeReceiveCopy(waitOp);
    if (failed(maybeCopyOp)) {
      return;
    }
    if (!maybeCopyOp->has_value()) {
      return;
    }
    CopyOp copyOp = **maybeCopyOp;
    auto pipeType = mlir::cast<PipeType>(copyOp.getSrc().getType());

    PipeEvent event;
    event.op = waitOp.getOperation();
    event.pipeType = pipeType;
    event.kind = PipeEventKind::ReceiveWait;
    event.domain =
        domain.intersectWith(getPipeDestinationLaunchNodeDomain(pipeType));
    event.unanalyzableOp = unanalyzableOp;

    auto [it, inserted] =
        pipeEventIndices.try_emplace(waitOp.getOperation(), pipeEvents.size());
    if (inserted) {
      pipeEvents.push_back(event);
      return;
    }
    pipeEvents[it->second] = event;
  }
};

/// Final launch-node facts for one operation.
struct PipeNetOperationDomainInfo {
  LaunchNodeDomain domain;
  Operation *unanalyzableOp = nullptr;
};

/// Final launch-node facts for one PipeNet scope.
struct PipeNetScopeDomainInfo {
  LaunchNodeDomain domain;
  PipeNetScopeLaunchNodeDomains scope;
};

/// Cache launch-node dataflow results shared by the two read-only verifiers.
class PipeNetLaunchNodeDomainAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PipeNetLaunchNodeDomainAnalysis)

  /// Compute and retain the final launch-node dataflow facts for `root`.
  explicit PipeNetLaunchNodeDomainAnalysis(Operation *root) {
    auto module = mlir::cast<ModuleOp>(root);
    LaunchNodeDomainState state;
    state.initialize(module);

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    LaunchNodeDomainAnalysisOptions options;
    options.operationCallback = [&](Operation *op,
                                    const LaunchNodeDomain &domain,
                                    Operation *unanalyzableOp) {
      operationDomains[op] = {domain, unanalyzableOp};
    };
    options.pipeNetScopeCallback =
        [&](PipeNetScopeOp scopeOp, const LaunchNodeDomain &domain,
            Operation * /*unanalyzableOp*/,
            const PipeNetScopeLaunchNodeDomains &scope) {
          scopeDomains[scopeOp.getOperation()] = {domain, scope};
        };
    solver.load<LaunchNodeDomainAnalysis>(state, options);
    valid = succeeded(solver.initializeAndRun(module)) && !state.sawError;
  }

  /// Return whether dataflow completed without errors.
  bool isValid() const { return valid; }

  /// Return final launch-node facts for `op`, if it was analyzed.
  const PipeNetOperationDomainInfo *getOperationInfo(Operation *op) const {
    auto infoIt = operationDomains.find(op);
    return infoIt == operationDomains.end() ? nullptr : &infoIt->second;
  }

  /// Return final launch-node facts for `scopeOp`, if it was analyzed.
  const PipeNetScopeDomainInfo *getScopeInfo(PipeNetScopeOp scopeOp) const {
    auto infoIt = scopeDomains.find(scopeOp.getOperation());
    return infoIt == scopeDomains.end() ? nullptr : &infoIt->second;
  }

private:
  llvm::DenseMap<Operation *, PipeNetOperationDomainInfo> operationDomains;
  llvm::DenseMap<Operation *, PipeNetScopeDomainInfo> scopeDomains;
  /// Whether analysis completed without an unsupported or invalid construct.
  bool valid = false;
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

/// Verify that a receive completion wait executes only at destinations of its
/// defining pipe receive.
void verifyPipeWaitGuard(WaitOp waitOp, const LaunchNodeDomain &domain,
                         Operation *unanalyzableOp, ModuleState &state) {
  FailureOr<std::optional<CopyOp>> maybeCopyOp =
      state.getPipeReceiveCopy(waitOp);
  if (failed(maybeCopyOp)) {
    return;
  }
  if (!maybeCopyOp->has_value()) {
    return;
  }

  auto pipeType = mlir::cast<PipeType>((**maybeCopyOp).getSrc().getType());
  int64_t netId = pipeType.getPipeNetId();
  std::string name = state.netName(netId);
  std::string message;
  llvm::raw_string_ostream(message)
      << "this `ttl.wait` waits for a pipe receive on launched nodes "
         "that are not destinations of PipeNet "
      << name << "; keep the wait under the same `if " << name
      << ".is_dst(): ...` or `" << name
      << ".if_dst(...)` guard as the receive copy";
  checkKnownSubset(waitOp, domain, getPipeDestinationLaunchNodeDomain(pipeType),
                   unanalyzableOp, message, {{netId, PipeRole::Destination}},
                   state);
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
      .Case<CopyOp>(
          [&](CopyOp copy) { verifyCopy(copy, domain, unanalyzableOp, state); })
      .Case<WaitOp>([&](WaitOp wait) {
        verifyPipeWaitGuard(wait, domain, unanalyzableOp, state);
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

/// Record the synchronization events used by the schedule verifier.
void recordScheduleOperation(Operation *op, const LaunchNodeDomain &domain,
                             Operation *unanalyzableOp, ModuleState &state) {
  TypeSwitch<Operation *>(op)
      .Case<CopyOp>([&](CopyOp copy) {
        state.recordPipeEvent(copy, domain, unanalyzableOp);
      })
      .Case<WaitOp>([&](WaitOp wait) {
        state.recordPipeWaitEvent(wait, domain, unanalyzableOp);
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

/// One direct call and the function entered by that call.
struct PipeCallSite {
  func::CallOp call;
  func::FuncOp callee;
};

/// Pipe synchronization event specialized to one launch node.
struct PipeScheduleNode {
  Operation *op;
  PipeType pipeType;
  LaunchNodeCoord coord;
  PipeEventKind kind;
  func::FuncOp kernelFunction;
  SmallVector<PipeCallSite> callSites;
  SmallVector<PipeScheduleEdge> successors;
};

/// Retain the sends for one logical pipe in deterministic traversal order.
struct PipeOccurrences {
  PipeType pipeType;
  SmallVector<PipeScheduleNodeId> sends;
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

/// Add one graph node for a synchronization event at one call-site occurrence.
PipeScheduleNodeId addPipeScheduleNode(SmallVectorImpl<PipeScheduleNode> &nodes,
                                       const PipeEvent &event,
                                       LaunchNodeCoord coord,
                                       func::FuncOp kernelFunction,
                                       ArrayRef<PipeCallSite> callSites) {
  PipeScheduleNodeId nodeId = nodes.size();
  nodes.push_back({event.op,
                   event.pipeType,
                   coord,
                   event.kind,
                   kernelFunction,
                   SmallVector<PipeCallSite>(callSites),
                   {}});
  return nodeId;
}

/// Reject event definitions whose relative order would depend on independent
/// kernel threads.
LogicalResult
verifySingleKernelFunction(ArrayRef<PipeScheduleNode> nodes,
                           ArrayRef<PipeScheduleNodeId> existingNodes,
                           const PipeEvent &event, func::FuncOp kernelFunction,
                           ModuleState &state) {
  if (existingNodes.empty()) {
    return success();
  }
  const PipeScheduleNode &firstNode = nodes[existingNodes.front()];
  if (firstNode.kernelFunction == kernelFunction) {
    return success();
  }
  auto diag = event.op->emitOpError()
              << "cannot verify PipeNet synchronization because "
              << getPipeEventName(event.kind)
              << " definitions for the same pipe endpoint occur in multiple "
                 "kernel-thread functions";
  func::FuncOp firstKernelFunction = firstNode.kernelFunction;
  diag.attachNote(firstNode.op->getLoc())
      << "the first " << getPipeEventName(event.kind) << " definition is in @"
      << firstKernelFunction.getSymName();
  state.sawError = true;
  return failure();
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
  struct DFSFrame {
    PipeScheduleNodeId nodeId;
    std::size_t nextSuccessor = 0;
  };

  SmallVector<PipeScheduleNodeId> activeNodes;
  SmallVector<DFSFrame> frames;
  SmallVector<std::uint8_t> colors(nodes.size(), 0);

  for (PipeScheduleNodeId nodeId = 0, count = nodes.size(); nodeId < count;
       ++nodeId) {
    if (colors[nodeId] != 0) {
      continue;
    }
    colors[nodeId] = 1;
    activeNodes.push_back(nodeId);
    frames.push_back({nodeId, 0});
    while (!frames.empty()) {
      DFSFrame &frame = frames.back();
      ArrayRef<PipeScheduleEdge> successors = nodes[frame.nodeId].successors;
      if (frame.nextSuccessor == successors.size()) {
        colors[frame.nodeId] = 2;
        frames.pop_back();
        activeNodes.pop_back();
        continue;
      }

      PipeScheduleNodeId successor =
          successors[frame.nextSuccessor++].successor;
      if (colors[successor] == 0) {
        colors[successor] = 1;
        activeNodes.push_back(successor);
        frames.push_back({successor, 0});
        continue;
      }
      if (colors[successor] != 1) {
        continue;
      }

      auto cycleStart = llvm::find(activeNodes, successor);
      SmallVector<PipeScheduleNodeId> cycle(cycleStart, activeNodes.end());
      cycle.push_back(successor);
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
  case PipeEventKind::Send:
    os << "send";
    break;
  case PipeEventKind::ReceivePost:
    os << "receiver post";
    break;
  case PipeEventKind::ReceiveWait:
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
    if (waitNode.kind != PipeEventKind::ReceiveWait) {
      continue;
    }
    for (std::size_t sendIdx = waitIdx + 1; sendIdx < cycle.size(); ++sendIdx) {
      PipeScheduleNodeId sendNodeId = cycle[sendIdx];
      const PipeScheduleNode &sendNode = nodes[sendNodeId];
      if (sendNode.kind != PipeEventKind::Send) {
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
    if (sendNode.kind != PipeEventKind::Send) {
      continue;
    }
    for (std::size_t postIdx = sendIdx + 1; postIdx < cycle.size(); ++postIdx) {
      PipeScheduleNodeId postNodeId = cycle[postIdx];
      const PipeScheduleNode &postNode = nodes[postNodeId];
      if (postNode.kind != PipeEventKind::ReceivePost) {
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
  std::size_t noteCount = 0;
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
    ++noteCount;
    if (noteCount >= kMaxPipeScheduleCycleNotes) {
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

  const PipeScheduleNode &node = nodes[cycle.front()];
  auto diag = node.op->emitOpError()
              << "pipe schedule contains a wait-for cycle on PipeNet "
              << state.netName(node.pipeType.getPipeNetId())
              << "; post the receive before the dependent send, or place the "
                 "send and receive in separate data-movement threads";

  emitPipeScheduleCycleNotes(diag, nodes, cycle);
  state.sawError = true;
}

/// Constant and unresolved factors in one schedule occurrence count.
struct PipeExecutionCountExpression {
  std::uint64_t constantFactor = 1;
  SmallVector<Operation *> unresolvedOps;
};

/// Separate proven constant factors from operations whose execution counts are
/// symbolic. Multiplication composes caller invocation counts with the local
/// count of an event inside a helper.
std::optional<PipeExecutionCountExpression>
getPipeExecutionCountExpression(const PipeScheduleNode &node,
                                ModuleState &state) {
  PipeExecutionCountExpression expression;
  auto collectFactor = [&](Operation *op) -> LogicalResult {
    std::optional<std::uint64_t> maybeCount =
        getExactExecutionCountAtLaunchNode(op, node.coord, state);
    if (!maybeCount) {
      expression.unresolvedOps.push_back(op);
      return success();
    }
    std::optional<std::uint64_t> maybeProduct =
        llvm::checkedMulUnsigned(expression.constantFactor, *maybeCount);
    if (!maybeProduct) {
      return failure();
    }
    expression.constantFactor = *maybeProduct;
    return success();
  };
  for (const PipeCallSite &callSite : node.callSites) {
    func::CallOp call = callSite.call;
    if (failed(collectFactor(call.getOperation()))) {
      return std::nullopt;
    }
  }
  if (failed(collectFactor(node.op))) {
    return std::nullopt;
  }
  return expression;
}

/// Return the operand supplied to a helper argument at this occurrence's call
/// site.
std::optional<Value> resolveFunctionArgument(BlockArgument argument,
                                             ArrayRef<PipeCallSite> callSites) {
  auto function =
      dyn_cast_if_present<func::FuncOp>(argument.getOwner()->getParentOp());
  if (!function || argument.getOwner() != &function.getBody().front()) {
    return std::nullopt;
  }
  auto callSiteIt = llvm::find_if(llvm::reverse(callSites),
                                  [&](const PipeCallSite &callSite) {
                                    return callSite.callee == function;
                                  });
  if (callSiteIt == callSites.rend()) {
    return std::nullopt;
  }
  func::CallOp call = callSiteIt->call;
  if (argument.getArgNumber() >= call.getNumOperands()) {
    return std::nullopt;
  }
  return call.getOperand(argument.getArgNumber());
}

/// Prove equal dynamic counts for two call-site-specific schedule nodes.
bool proveEqualPipeScheduleNodeCounts(const PipeScheduleNode &lhs,
                                      const PipeScheduleNode &rhs,
                                      ModuleState &state) {
  std::optional<PipeExecutionCountExpression> maybeLhs =
      getPipeExecutionCountExpression(lhs, state);
  std::optional<PipeExecutionCountExpression> maybeRhs =
      getPipeExecutionCountExpression(rhs, state);
  if (!maybeLhs || !maybeRhs ||
      maybeLhs->constantFactor != maybeRhs->constantFactor ||
      maybeLhs->unresolvedOps.size() != maybeRhs->unresolvedOps.size()) {
    return false;
  }
  return llvm::all_of(
      llvm::zip(maybeLhs->unresolvedOps, maybeRhs->unresolvedOps),
      [&](auto pair) {
        auto resolveLhsFunctionArgument = [&](BlockArgument argument) {
          return resolveFunctionArgument(argument, lhs.callSites);
        };
        auto resolveRhsFunctionArgument = [&](BlockArgument argument) {
          return resolveFunctionArgument(argument, rhs.callSites);
        };
        return proveEqualUnresolvedExecutionCountAtLaunchNodes(
            std::get<0>(pair), lhs.coord, std::get<1>(pair), rhs.coord, state,
            resolveLhsFunctionArgument, resolveRhsFunctionArgument);
      });
}

/// Return whether the static predecessor and successor counts cannot pair.
bool haveInvalidPipeOccurrenceCount(ArrayRef<PipeScheduleNodeId> predecessors,
                                    ArrayRef<PipeScheduleNodeId> successors,
                                    bool requireEqualOccurrences) {
  return requireEqualOccurrences ? predecessors.size() != successors.size()
                                 : predecessors.size() < successors.size();
}

/// Emit one diagnostic for a static event-definition mismatch across one or
/// more receivers of the same pipe.
void emitPipeOccurrenceCountError(ArrayRef<PipeScheduleNode> nodes,
                                  ArrayRef<PipeScheduleNodeId> predecessors,
                                  ArrayRef<PipeScheduleNodeId> successors,
                                  StringRef predecessorName,
                                  StringRef successorName,
                                  ArrayRef<LaunchNodeCoord> receiverCoords,
                                  ModuleState &state) {
  assert(!receiverCoords.empty() && "expected an affected receiver");
  bool hasExtraPredecessor = predecessors.size() > successors.size();
  PipeScheduleNodeId unmatchedNode = hasExtraPredecessor
                                         ? predecessors[successors.size()]
                                         : successors[predecessors.size()];
  StringRef unmatchedName =
      hasExtraPredecessor ? predecessorName : successorName;
  StringRef missingName = hasExtraPredecessor ? successorName : predecessorName;
  LaunchNodeCoord firstCoord = receiverCoords.front();
  auto diag = nodes[unmatchedNode].op->emitOpError()
              << "PipeNet "
              << state.netName(nodes[unmatchedNode].pipeType.getPipeNetId())
              << " requires one static " << predecessorName
              << " definition for each static " << successorName
              << " definition at receiver core_x=" << firstCoord.x
              << ", core_y=" << firstCoord.y << "; found "
              << predecessors.size() << " static " << predecessorName
              << " definition(s) and " << successors.size() << " static "
              << successorName << " definition(s)";
  diag.attachNote(nodes[unmatchedNode].op->getLoc())
      << "this " << unmatchedName << " has no corresponding " << missingName;
  std::size_t noteCount = 1;
  for (LaunchNodeCoord coord : receiverCoords.drop_front()) {
    if (noteCount >= kMaxPipeScheduleCycleNotes) {
      break;
    }
    diag.attachNote(nodes[unmatchedNode].op->getLoc())
        << "the same mismatch applies at receiver core_x=" << coord.x
        << ", core_y=" << coord.y;
    ++noteCount;
  }
  state.sawError = true;
}

/// Pair predecessor and successor operations at the same traversal position.
/// Repeated pairs must have equal execution counts under equivalent control
/// conditions.
LogicalResult addPipeOccurrenceEdges(SmallVectorImpl<PipeScheduleNode> &nodes,
                                     ArrayRef<PipeScheduleNodeId> predecessors,
                                     ArrayRef<PipeScheduleNodeId> successors,
                                     PipeScheduleEdgeKind kind,
                                     StringRef predecessorName,
                                     StringRef successorName,
                                     LaunchNodeCoord receiverCoord,
                                     ModuleState &state,
                                     bool requireEqualOccurrences = true) {
  assert(!haveInvalidPipeOccurrenceCount(predecessors, successors,
                                         requireEqualOccurrences) &&
         "static occurrence counts must be validated before pairing");
  for (auto [predecessor, successor] : llvm::zip(predecessors, successors)) {
    if (!proveEqualPipeScheduleNodeCounts(nodes[predecessor], nodes[successor],
                                          state)) {
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

/// Return true when an operation or one of its enclosing calls is proven not
/// to execute at `coord`.
bool hasZeroExecutionCount(ArrayRef<PipeCallSite> callSites, Operation *op,
                           LaunchNodeCoord coord, ModuleState &state) {
  if (llvm::any_of(callSites, [&](const PipeCallSite &callSite) {
        std::optional<std::uint64_t> maybeCount =
            getExactExecutionCountAtLaunchNode(callSite.call, coord, state);
        return maybeCount && *maybeCount == 0;
      })) {
    return true;
  }
  std::optional<std::uint64_t> maybeCount =
      getExactExecutionCountAtLaunchNode(op, coord, state);
  return maybeCount && *maybeCount == 0;
}

/// Return the functions that contain a pipe event directly or through calls to
/// another function in the module.
llvm::DenseSet<Operation *>
getFunctionsWithPipeEvents(ModuleOp module, const ModuleState &state,
                           SymbolTableCollection &symbolTables) {
  llvm::DenseSet<Operation *> functions;
  for (const PipeEvent &event : state.pipeEvents) {
    if (auto function = event.op->getParentOfType<func::FuncOp>()) {
      functions.insert(function.getOperation());
    }
  }

  bool changed;
  do {
    changed = false;
    for (func::FuncOp function : module.getOps<func::FuncOp>()) {
      if (functions.contains(function.getOperation())) {
        continue;
      }
      function.walk([&](func::CallOp callOp) {
        func::FuncOp callee =
            symbolTables.lookupNearestSymbolFrom<func::FuncOp>(
                callOp, callOp.getCalleeAttr());
        if (!callee || !functions.contains(callee.getOperation())) {
          return WalkResult::advance();
        }
        functions.insert(function.getOperation());
        changed = true;
        return WalkResult::interrupt();
      });
    }
  } while (changed);
  return functions;
}

/// Return functions reachable from kernel-thread entry points through direct
/// calls.
llvm::DenseSet<Operation *>
getFunctionsReachableFromKernelThreads(ModuleOp module,
                                       SymbolTableCollection &symbolTables) {
  llvm::DenseSet<Operation *> reachableFunctions;
  SmallVector<func::FuncOp> worklist;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (function->hasAttr(kKernelThreadAttrName) &&
        reachableFunctions.insert(function.getOperation()).second) {
      worklist.push_back(function);
    }
  }
  for (std::size_t worklistIndex = 0; worklistIndex < worklist.size();
       ++worklistIndex) {
    worklist[worklistIndex].walk([&](func::CallOp callOp) {
      func::FuncOp callee = symbolTables.lookupNearestSymbolFrom<func::FuncOp>(
          callOp, callOp.getCalleeAttr());
      if (callee && reachableFunctions.insert(callee.getOperation()).second) {
        worklist.push_back(callee);
      }
    });
  }
  return reachableFunctions;
}

/// Reject pipe events with no kernel-thread entry point and direct-call chain.
LogicalResult verifyPipeEventFunctionsReachable(
    const ModuleState &state,
    const llvm::DenseSet<Operation *> &reachableFunctions) {
  llvm::DenseSet<Operation *> diagnosedFunctions;
  LogicalResult result = success();
  for (const PipeEvent &event : state.pipeEvents) {
    func::FuncOp function = event.op->getParentOfType<func::FuncOp>();
    if (!function || reachableFunctions.contains(function.getOperation()) ||
        !diagnosedFunctions.insert(function.getOperation()).second) {
      continue;
    }
    event.op->emitOpError()
        << "cannot verify PipeNet synchronization in @" << function.getSymName()
        << " because it is not reachable from a kernel-thread function "
           "through direct calls";
    result = failure();
  }
  return result;
}

/// Reject control flow whose execution order cannot be represented by a
/// linear sequence of synchronization events.
LogicalResult verifyPipeEventFunctionsHaveOneBlock(
    ModuleOp module,
    const llvm::DenseSet<Operation *> &functionsWithPipeEvents) {
  LogicalResult result = success();
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (!functionsWithPipeEvents.contains(function.getOperation()) ||
        function.getBody().hasOneBlock()) {
      continue;
    }
    function.emitOpError()
        << "cannot verify PipeNet synchronization in multi-block function @"
        << function.getSymName()
        << "; schedule verification requires a single-block function body";
    result = failure();
  }
  return result;
}

/// Reject events whose launch-node set is not known exactly. Omitting such an
/// event from coordinate-specific schedules could accept an invalid program.
LogicalResult verifyPipeEventDomainsKnown(const ModuleState &state) {
  LogicalResult result = success();
  for (const PipeEvent &event : state.pipeEvents) {
    if (event.domain.known) {
      continue;
    }
    auto diag = event.op->emitOpError()
                << "cannot verify PipeNet synchronization because this "
                << getPipeEventName(event.kind)
                << " has an unknown launch-node domain";
    if (event.unanalyzableOp) {
      diag.attachNote(event.unanalyzableOp->getLoc())
          << "this coordinate-dependent condition cannot be evaluated "
             "statically";
    }
    result = failure();
  }
  return result;
}

/// Visit pipe events in the order executed by one kernel thread. Direct helper
/// calls are expanded at each call site so their events retain caller order and
/// invocation multiplicity.
WalkResult walkPipeEventsInProgramOrder(
    Operation *op, LaunchNodeCoord coord, ModuleState &state,
    const llvm::DenseSet<Operation *> &functionsWithPipeEvents,
    SymbolTableCollection &symbolTables,
    SmallVectorImpl<func::FuncOp> &activeFunctions,
    SmallVectorImpl<PipeCallSite> &callSites,
    llvm::DenseSet<Operation *> &diagnosedRecursiveCalls,
    llvm::function_ref<WalkResult(const PipeEvent &, ArrayRef<PipeCallSite>)>
        visitEvent) {
  auto eventIt = state.pipeEventIndices.find(op);
  if (eventIt != state.pipeEventIndices.end()) {
    const PipeEvent &event = state.pipeEvents[eventIt->second];
    if (knownLaunchNodeDomainContains(event.domain, coord)) {
      if (!hasZeroExecutionCount(callSites, event.op, coord, state) &&
          visitEvent(event, callSites).wasInterrupted()) {
        return WalkResult::interrupt();
      }
    }
  }

  if (auto callOp = mlir::dyn_cast<func::CallOp>(op)) {
    func::FuncOp callee = symbolTables.lookupNearestSymbolFrom<func::FuncOp>(
        callOp, callOp.getCalleeAttr());
    if (!callee || !functionsWithPipeEvents.contains(callee.getOperation()) ||
        hasZeroExecutionCount({}, callOp.getOperation(), coord, state)) {
      return WalkResult::advance();
    }
    if (llvm::is_contained(activeFunctions, callee)) {
      if (diagnosedRecursiveCalls.insert(callOp.getOperation()).second) {
        callOp.emitOpError()
            << "cannot verify PipeNet synchronization through a recursive "
               "call to @"
            << callee.getSymName();
        state.sawError = true;
      }
      return WalkResult::advance();
    }

    activeFunctions.push_back(callee);
    callSites.push_back({callOp, callee});
    llvm::scope_exit restoreCallStack([&] {
      callSites.pop_back();
      activeFunctions.pop_back();
    });
    for (Block &block : callee.getBody()) {
      for (Operation &nestedOp : block) {
        if (walkPipeEventsInProgramOrder(
                &nestedOp, coord, state, functionsWithPipeEvents, symbolTables,
                activeFunctions, callSites, diagnosedRecursiveCalls, visitEvent)
                .wasInterrupted()) {
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (walkPipeEventsInProgramOrder(
                &nestedOp, coord, state, functionsWithPipeEvents, symbolTables,
                activeFunctions, callSites, diagnosedRecursiveCalls, visitEvent)
                .wasInterrupted()) {
          return WalkResult::interrupt();
        }
      }
    }
  }
  return WalkResult::advance();
}

// Verify synchronization dependencies implied by pipe operations. Receive-side
// ttl.copy makes a reserved DFB slot available to the sender, while ttl.wait on
// its handle waits for payload arrival. Modeling availability and completion as
// separate events preserves asynchronous semantics and detects wait-for cycles.
void verifyPipeScheduleCycles(ModuleOp module, ModuleState &state) {
  SmallVector<PipeScheduleNode> nodes;
  llvm::MapVector<PipeIdentity, PipeOccurrences> pipeOccurrences;
  llvm::DenseMap<PipeCoordIdentity, SmallVector<PipeScheduleNodeId>>
      receivePostNodes;
  llvm::DenseMap<PipeCoordIdentity, SmallVector<PipeScheduleNodeId>>
      receiveWaitNodes;
  SymbolTableCollection symbolTables;
  llvm::DenseSet<Operation *> functionsWithPipeEvents =
      getFunctionsWithPipeEvents(module, state, symbolTables);
  llvm::DenseSet<Operation *> reachableFunctions =
      getFunctionsReachableFromKernelThreads(module, symbolTables);
  if (failed(verifyPipeEventFunctionsReachable(state, reachableFunctions)) ||
      failed(verifyPipeEventFunctionsHaveOneBlock(module,
                                                  functionsWithPipeEvents)) ||
      failed(verifyPipeEventDomainsKnown(state))) {
    state.sawError = true;
    return;
  }
  llvm::DenseSet<Operation *> diagnosedRecursiveCalls;
  LaunchNodeDomain eventDomain;
  for (const PipeEvent &event : state.pipeEvents) {
    eventDomain = eventDomain.unionWith(event.domain);
  }
  const std::set<LaunchNodeCoord> &scheduleCoords =
      eventDomain.known ? eventDomain.nodes : state.baseDomain.nodes;

  // Kernel-thread functions execute independently. Expand their direct helper
  // calls separately at each launch node to retain the order and multiplicity
  // of their pipe events.
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (!function->hasAttr(kKernelThreadAttrName)) {
      continue;
    }
    for (LaunchNodeCoord coord : scheduleCoords) {
      SmallVector<func::FuncOp> activeFunctions{function};
      SmallVector<PipeCallSite> callSites;
      std::optional<PipeScheduleNodeId> lastNode;
      for (Block &block : function.getBody()) {
        for (Operation &op : block) {
          WalkResult walkResult = walkPipeEventsInProgramOrder(
              &op, coord, state, functionsWithPipeEvents, symbolTables,
              activeFunctions, callSites, diagnosedRecursiveCalls,
              [&](const PipeEvent &event,
                  ArrayRef<PipeCallSite> activeCallSites) {
                if (nodes.size() >= kMaxPipeScheduleNodes) {
                  event.op->emitOpError()
                      << "cannot verify PipeNet synchronization because the "
                         "schedule contains more than "
                      << kMaxPipeScheduleNodes
                      << " pipe events after specializing launch nodes and "
                         "expanding helper calls";
                  state.sawError = true;
                  return WalkResult::interrupt();
                }
                PipeIdentity pipeIdentity = getPipeIdentity(event.pipeType);
                auto pipeIt =
                    pipeOccurrences
                        .try_emplace(pipeIdentity,
                                     PipeOccurrences{event.pipeType, {}})
                        .first;
                SmallVector<PipeScheduleNodeId> *matchingNodes = nullptr;
                if (event.kind == PipeEventKind::Send) {
                  matchingNodes = &pipeIt->second.sends;
                } else if (event.kind == PipeEventKind::ReceivePost) {
                  matchingNodes = &receivePostNodes[getPipeCoordIdentity(
                      event.pipeType, coord)];
                } else {
                  matchingNodes = &receiveWaitNodes[getPipeCoordIdentity(
                      event.pipeType, coord)];
                }
                if (failed(verifySingleKernelFunction(
                        nodes, *matchingNodes, event, function, state))) {
                  return WalkResult::interrupt();
                }
                PipeScheduleNodeId nodeId = addPipeScheduleNode(
                    nodes, event, coord, function, activeCallSites);
                matchingNodes->push_back(nodeId);
                if (lastNode) {
                  addPipeScheduleEdge(nodes, *lastNode, nodeId,
                                      PipeScheduleEdgeKind::ProgramOrder);
                }
                lastNode = nodeId;
                return WalkResult::advance();
              });
          if (walkResult.wasInterrupted()) {
            return;
          }
        }
      }
    }
  }

  for (const auto &[pipeIdentity, occurrences] : pipeOccurrences) {
    LaunchNodeDomain destinations =
        getPipeDestinationLaunchNodeDomain(occurrences.pipeType);
    auto getReceiverNodes = [](auto &nodesByCoord,
                               const PipeCoordIdentity &identity) {
      auto nodeIt = nodesByCoord.find(identity);
      return nodeIt == nodesByCoord.end()
                 ? ArrayRef<PipeScheduleNodeId>()
                 : ArrayRef<PipeScheduleNodeId>(nodeIt->second);
    };

    SmallVector<LaunchNodeCoord> postMismatchCoords;
    if (!occurrences.sends.empty()) {
      for (LaunchNodeCoord coord : destinations.nodes) {
        ArrayRef<PipeScheduleNodeId> posts =
            getReceiverNodes(receivePostNodes,
                             getPipeCoordIdentity(occurrences.pipeType, coord));
        if (haveInvalidPipeOccurrenceCount(posts, occurrences.sends,
                                           /*requireEqualOccurrences=*/true)) {
          postMismatchCoords.push_back(coord);
        }
      }
    }
    if (!postMismatchCoords.empty()) {
      ArrayRef<PipeScheduleNodeId> posts = getReceiverNodes(
          receivePostNodes, getPipeCoordIdentity(occurrences.pipeType,
                                                 postMismatchCoords.front()));
      emitPipeOccurrenceCountError(nodes, posts, occurrences.sends,
                                   "receiver post", "send", postMismatchCoords,
                                   state);
      continue;
    }

    SmallVector<LaunchNodeCoord> waitMismatchCoords;
    for (LaunchNodeCoord coord : destinations.nodes) {
      ArrayRef<PipeScheduleNodeId> waits = getReceiverNodes(
          receiveWaitNodes, getPipeCoordIdentity(occurrences.pipeType, coord));
      if (haveInvalidPipeOccurrenceCount(occurrences.sends, waits,
                                         /*requireEqualOccurrences=*/false)) {
        waitMismatchCoords.push_back(coord);
      }
    }
    if (!waitMismatchCoords.empty()) {
      ArrayRef<PipeScheduleNodeId> waits = getReceiverNodes(
          receiveWaitNodes, getPipeCoordIdentity(occurrences.pipeType,
                                                 waitMismatchCoords.front()));
      emitPipeOccurrenceCountError(nodes, occurrences.sends, waits, "send",
                                   "receive wait", waitMismatchCoords, state);
      continue;
    }

    for (LaunchNodeCoord coord : destinations.nodes) {
      PipeCoordIdentity identity =
          getPipeCoordIdentity(occurrences.pipeType, coord);
      if (!occurrences.sends.empty()) {
        ArrayRef<PipeScheduleNodeId> posts =
            getReceiverNodes(receivePostNodes, identity);
        if (failed(addPipeOccurrenceEdges(
                nodes, posts, occurrences.sends,
                PipeScheduleEdgeKind::ReceivePostEnablesSend, "receiver post",
                "send", coord, state))) {
          break;
        }
      }
      ArrayRef<PipeScheduleNodeId> waits =
          getReceiverNodes(receiveWaitNodes, identity);
      if (!waits.empty()) {
        if (failed(addPipeOccurrenceEdges(
                nodes, occurrences.sends, waits,
                PipeScheduleEdgeKind::SendCompletesReceive, "send",
                "receive wait", coord, state,
                /*requireEqualOccurrences=*/false))) {
          break;
        }
      }
    }
  }

  if (std::optional<SmallVector<PipeScheduleNodeId>> maybeCycle =
          findPipeScheduleCycle(nodes)) {
    emitPipeScheduleCycleDiagnostic(nodes, *maybeCycle, state);
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

/// Verify that every declared pipe endpoint belongs to the module launch grid.
LogicalResult validatePipeEndpoints(ModuleOp module, const ModuleState &state) {
  LogicalResult result = success();
  module.walk([&](CreatePipeOp pipe) {
    PipeType pipeType = mlir::cast<PipeType>(pipe.getResult().getType());
    LaunchNodeCoord source{pipeType.getSrcX(), pipeType.getSrcY()};
    if (!knownLaunchNodeDomainContains(state.baseDomain, source)) {
      pipe.emitOpError() << "declares source core_x=" << source.x
                         << ", core_y=" << source.y
                         << " outside the module `ttl.launch_grid`";
      result = failure();
      return;
    }
    if (!getPipeDestinationLaunchNodeDomain(pipeType).isSubsetOf(
            state.baseDomain)) {
      pipe.emitOpError() << "declares destination range core_x="
                         << pipeType.getDstStartX() << ".."
                         << pipeType.getDstEndX()
                         << ", core_y=" << pipeType.getDstStartY() << ".."
                         << pipeType.getDstEndY()
                         << " outside the module `ttl.launch_grid`";
      result = failure();
    }
  });
  return result;
}

/// Initialize the shared PipeNet metadata required by both verification passes.
LogicalResult initializePipeNetVerification(ModuleOp module, ModuleState &state,
                                            StringRef passName) {
  state.initialize(module);
  validatePipeNetReferences(module, state);
  if (state.sawError) {
    return failure();
  }
  if (!state.hasPipes()) {
    return success();
  }
  if (!state.hasLaunchGrid) {
    module.emitError() << passName
                       << " requires a `ttl.launch_grid` module attribute "
                          "(an i64 array of length 2 with positive entries)";
    return failure();
  }
  return validatePipeEndpoints(module, state);
}

struct TTLVerifyPipeNetGuardsPass
    : impl::TTLVerifyPipeNetGuardsBase<TTLVerifyPipeNetGuardsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    ValueOriginAnalysis &valueOrigins = getAnalysis<ValueOriginAnalysis>();
    ModuleState state(valueOrigins);
    if (failed(initializePipeNetVerification(module, state,
                                             "ttl-verify-pipenet-guards"))) {
      signalPassFailure();
      return;
    }
    if (!state.hasPipes()) {
      markAllAnalysesPreserved();
      return;
    }

    const PipeNetLaunchNodeDomainAnalysis &launchDomains =
        getAnalysis<PipeNetLaunchNodeDomainAnalysis>();
    if (!launchDomains.isValid()) {
      signalPassFailure();
      return;
    }
    module.walk([&](Operation *op) {
      if (const PipeNetOperationDomainInfo *info =
              launchDomains.getOperationInfo(op)) {
        recordGuardOperation(op, info->domain, info->unanalyzableOp, state);
      }
      if (auto scopeOp = mlir::dyn_cast<PipeNetScopeOp>(op)) {
        if (const PipeNetScopeDomainInfo *info =
                launchDomains.getScopeInfo(scopeOp)) {
          verifyPipeNetScope(scopeOp, info->domain, info->scope, state);
        }
      }
    });

    verifyCBWaits(state);
    if (state.sawError) {
      signalPassFailure();
      return;
    }
    markAllAnalysesPreserved();
  }
};

struct TTLVerifyPipeNetSchedulePass
    : impl::TTLVerifyPipeNetScheduleBase<TTLVerifyPipeNetSchedulePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    ValueOriginAnalysis &valueOrigins = getAnalysis<ValueOriginAnalysis>();
    ModuleState state(valueOrigins);
    if (failed(initializePipeNetVerification(module, state,
                                             "ttl-verify-pipenet-schedule"))) {
      signalPassFailure();
      return;
    }
    if (!state.hasPipes()) {
      markAllAnalysesPreserved();
      return;
    }

    const PipeNetLaunchNodeDomainAnalysis &launchDomains =
        getAnalysis<PipeNetLaunchNodeDomainAnalysis>();
    if (!launchDomains.isValid()) {
      signalPassFailure();
      return;
    }
    module.walk([&](Operation *op) {
      if (const PipeNetOperationDomainInfo *info =
              launchDomains.getOperationInfo(op)) {
        recordScheduleOperation(op, info->domain, info->unanalyzableOp, state);
      }
    });

    if (!state.sawError) {
      verifyPipeScheduleCycles(module, state);
    }
    if (state.sawError) {
      signalPassFailure();
      return;
    }
    markAllAnalysesPreserved();
  }
};

} // namespace

} // namespace mlir::tt::ttl
