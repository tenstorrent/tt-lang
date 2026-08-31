// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

//===----------------------------------------------------------------------===//
// Verify the launch-node domains and synchronization schedules of PipeNet
// operations. Launch-node dataflow determines where each operation can execute.
// The guard pass checks those domains against PipeNet roles, including
// table-selected source and destination records. The schedule pass expands
// direct calls and record callbacks to prove event correspondence and reject
// wait-for cycles in execution order.
//===----------------------------------------------------------------------===//

#include "DFBVerification.h"
#include "PipeGraph.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "ttlang/Analysis/ExecutionCountAnalysis.h"
#include "ttlang/Analysis/ValueOriginAnalysis.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/IR/TTLOps.h"
#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Transforms/DFBLogicalIdentityAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/PipeNetExecutionUtils.h"
#include "ttlang/Dialect/TTL/Transforms/PipeTransferAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/TransferProvenance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <tuple>

#define DEBUG_TYPE "ttl-verify-pipenet"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLVERIFYPIPENETGUARDS
#define GEN_PASS_DEF_TTLVERIFYPIPENETSCHEDULE
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

constexpr std::size_t kMaxPipeScheduleDiagnosticNotes = 8;
constexpr std::size_t kMaxPipeScheduleNodesPerLaunchNode = 4096;

//===----------------------------------------------------------------------===//
// Module state collected before the analysis runs and updated during it.
//===----------------------------------------------------------------------===//

/// A dataflow buffer wait and the launch-node domain where it executes.
struct WaitUse {
  CBWaitOp op;
  LaunchNodeDomain domain;
  int64_t dfbId;
};

/// Pipe synchronization event used by the wait-for graph verifier.
enum class PipeEventKind { Send, ReceivePost, ReceiveWait, ReceiveWaitAny };

/// Return the diagnostic name for a pipe synchronization event.
StringRef getPipeEventName(PipeEventKind kind) {
  switch (kind) {
  case PipeEventKind::Send:
    return "send";
  case PipeEventKind::ReceivePost:
    return "receiver post";
  case PipeEventKind::ReceiveWait:
    return "receive wait";
  case PipeEventKind::ReceiveWaitAny:
    return "receive wait-any";
  }
  llvm_unreachable("unknown pipe event kind");
}

/// One completion alternative of a receive wait-any event.
struct PipeWaitAnyAlternative {
  PipeType pipeType;
  Operation *receivePost = nullptr;
};

/// One pipe synchronization event on the launch-node domain where it executes.
struct PipeEvent {
  Operation *op = nullptr;
  Value pipe;
  PipeType pipeType;
  DeviceTransferAttr deviceTransfer;
  PipeEventKind kind;
  // Exact analyzed domain, or unknown when a predicate cannot be evaluated.
  LaunchNodeDomain domain;
  // Known conservative domain used to enumerate possible schedule events.
  LaunchNodeDomain scheduleDomain;
  Operation *unanalyzableOp = nullptr;
  /// Receive post whose token is observed by a receive-wait event.
  Operation *receivePost = nullptr;
  Operation *selectedForeachOp = nullptr;
  int64_t selectedRecordIndex = -1;
  SmallVector<PipeWaitAnyAlternative> waitAnyAlternatives = {};
};

PipeEvent makePipeEvent(Operation *op, Value pipe, PipeType pipeType,
                        DeviceTransferAttr deviceTransfer, PipeEventKind kind,
                        const LaunchNodeDomain &domain,
                        const LaunchNodeDomain &roleDomain,
                        Operation *unanalyzableOp, Operation *receivePost,
                        Operation *selectedForeachOp,
                        int64_t selectedRecordIndex) {
  LaunchNodeDomain exactDomain = domain.intersectWith(roleDomain);
  LaunchNodeDomain scheduleDomain = exactDomain;
  if (!scheduleDomain.known) {
    if (const std::set<LaunchNodeCoord> *upperBound =
            exactDomain.getUpperBoundNodes()) {
      scheduleDomain = LaunchNodeDomain{};
      scheduleDomain.nodes = *upperBound;
    }
  }
  return PipeEvent{op,
                   pipe,
                   pipeType,
                   deviceTransfer,
                   kind,
                   std::move(exactDomain),
                   std::move(scheduleDomain),
                   unanalyzableOp,
                   receivePost,
                   selectedForeachOp,
                   selectedRecordIndex};
}

/// Return the source or destination execution location for `event`.
FailureOr<LaunchExecutionLocation>
getPipeEventExecutionLocation(const PipeEvent &event, LaunchNodeCoord node) {
  PipeRole role = event.kind == PipeEventKind::Send ? PipeRole::Source
                                                    : PipeRole::Destination;
  return getPipeExecutionLocation(node, event.deviceTransfer, role);
}

struct ModuleState;
void checkKnownSubset(Operation *op, const LaunchNodeDomain &current,
                      const LaunchNodeDomain &allowed,
                      Operation *unanalyzableOp, Twine primaryMessage,
                      ArrayRef<std::pair<int64_t, PipeRole>> roles,
                      ModuleState &state);

/// Mutable facts recorded during one verifier pass.
struct ModuleState {
  /// Constructs state for schedule verification.
  ModuleState(const PipeTransferIndex &transfers,
              const LaunchNodeDomainState &launchDomains,
              ValueOriginAnalysis &valueOrigins)
      : transfers(transfers), launchDomains(launchDomains),
        valueOrigins(valueOrigins) {}

  /// Constructs state for guard verification with resolved DFB identities.
  ModuleState(const PipeTransferIndex &transfers,
              const LaunchNodeDomainState &launchDomains,
              ValueOriginAnalysis &valueOrigins,
              const DFBLogicalIdentityAnalysis &dfbIdentities)
      : transfers(transfers), launchDomains(launchDomains),
        valueOrigins(valueOrigins), dfbIdentities(&dfbIdentities) {}

  const PipeTransferIndex &transfers;
  const LaunchNodeDomainState &launchDomains;
  ValueOriginAnalysis &valueOrigins;
  /// Logical identities required by guard verification.
  const DFBLogicalIdentityAnalysis *dfbIdentities = nullptr;
  bool sawError = false;
  llvm::DenseMap<int64_t, LaunchNodeDomain> dfbProducerDomains;
  SmallVector<WaitUse> waitUses;
  SmallVector<PipeEvent> pipeEvents;
  llvm::DenseMap<Operation *, SmallVector<std::size_t>> pipeEventIndices;

  /// Diagnose malformed selected-pipe IR when this pass runs directly.
  void reportInvalidSelectedPipeDefinition(CopyOp copyOp) {
    copyOp.emitOpError()
        << "selected pipe operand must be defined by ttl.select_pipe_src, "
           "ttl.select_pipe_dst, ttl.pipenet_foreach_src, or "
           "ttl.pipenet_foreach_dst";
    sawError = true;
  }

  /// Replace the events for one operation after a dataflow update.
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

  /// Append one event for each selected record represented by `op`.
  void appendSelectedPipeEvents(Operation *op, Value pipe,
                                PipeNetRecordsAttr records, PipeEventKind kind,
                                const LaunchNodeDomain &domain,
                                Operation *unanalyzableOp,
                                Operation *receivePost, Operation *foreachOp,
                                SmallVectorImpl<PipeEvent> &events) {
    PipeRole role =
        kind == PipeEventKind::Send ? PipeRole::Source : PipeRole::Destination;
    for (auto [recordIndex, record] : llvm::enumerate(records.getPipes())) {
      PipeType pipeType = getPipeTypeFromRecord(records.getContext(), record,
                                                records.getPipeNetId());
      LaunchNodeDomain roleDomain =
          getPipeRecordRoleLaunchNodeDomain(record, role);
      events.push_back(
          makePipeEvent(op, pipe, pipeType, record.getDeviceTransfer(), kind,
                        domain, roleDomain, unanalyzableOp, receivePost,
                        foreachOp, static_cast<int64_t>(recordIndex)));
    }
  }

  /// Return the direct or selected-record events represented by `copyOp`.
  SmallVector<PipeEvent> getPipeCopyEvents(CopyOp copyOp,
                                           const LaunchNodeDomain &domain,
                                           Operation *unanalyzableOp) {
    SmallVector<PipeEvent> events;
    Operation *op = copyOp.getOperation();
    if (auto pipeType = mlir::dyn_cast<PipeType>(copyOp.getDst().getType())) {
      events.push_back(
          makePipeEvent(op, copyOp.getDst(), pipeType, DeviceTransferAttr(),
                        PipeEventKind::Send, domain,
                        getPipeSourceLaunchNodeDomain(pipeType), unanalyzableOp,
                        /*receivePost=*/nullptr, /*selectedForeachOp=*/nullptr,
                        /*selectedRecordIndex=*/-1));
      return events;
    }
    FailureOr<SelectedPipeRecords> selectedDst =
        getSelectedPipeRecords(copyOp.getDst());
    if (succeeded(selectedDst)) {
      appendSelectedPipeEvents(op, copyOp.getDst(), selectedDst->records,
                               PipeEventKind::Send, domain, unanalyzableOp,
                               /*receivePost=*/nullptr,
                               selectedDst->maybeForeachOp, events);
      return events;
    }
    if (mlir::isa<SelectedPipeSrcType, SelectedPipeDstType>(
            copyOp.getDst().getType())) {
      reportInvalidSelectedPipeDefinition(copyOp);
      return events;
    }
    if (auto pipeType = mlir::dyn_cast<PipeType>(copyOp.getSrc().getType())) {
      if (isPipeReceiveCopy(copyOp)) {
        events.push_back(makePipeEvent(
            op, copyOp.getSrc(), pipeType, DeviceTransferAttr(),
            PipeEventKind::ReceivePost, domain,
            getPipeDestinationLaunchNodeDomain(pipeType,
                                               launchDomains.baseDomain),
            unanalyzableOp, /*receivePost=*/nullptr,
            /*selectedForeachOp=*/nullptr, /*selectedRecordIndex=*/-1));
      }
      return events;
    }
    FailureOr<SelectedPipeRecords> selectedSrc =
        getSelectedPipeRecords(copyOp.getSrc());
    if (succeeded(selectedSrc)) {
      if (isPipeReceiveCopy(copyOp)) {
        appendSelectedPipeEvents(op, copyOp.getSrc(), selectedSrc->records,
                                 PipeEventKind::ReceivePost, domain,
                                 unanalyzableOp, /*receivePost=*/nullptr,
                                 selectedSrc->maybeForeachOp, events);
      }
    } else if (mlir::isa<SelectedPipeSrcType, SelectedPipeDstType>(
                   copyOp.getSrc().getType())) {
      reportInvalidSelectedPipeDefinition(copyOp);
    }
    return events;
  }

  /// Record pipe sends and receive posts from `ttl.copy` operations.
  void recordPipeEvent(CopyOp copyOp, const LaunchNodeDomain &domain,
                       Operation *unanalyzableOp) {
    replacePipeEvents(copyOp.getOperation(),
                      getPipeCopyEvents(copyOp, domain, unanalyzableOp));
  }

  /// Record a receive completion wait for schedule verification.
  void recordPipeWaitEvent(WaitOp waitOp, const LaunchNodeDomain &domain,
                           Operation *unanalyzableOp) {
    std::optional<CopyOp> maybeCopyOp = transfers.getReceivePost(waitOp);
    if (!maybeCopyOp) {
      return;
    }
    CopyOp copyOp = *maybeCopyOp;
    SmallVector<PipeEvent> events;
    Operation *op = waitOp.getOperation();
    if (auto pipeType = mlir::dyn_cast<PipeType>(copyOp.getSrc().getType())) {
      events.push_back(makePipeEvent(
          op, copyOp.getSrc(), pipeType, DeviceTransferAttr(),
          PipeEventKind::ReceiveWait, domain,
          getPipeDestinationLaunchNodeDomain(pipeType,
                                             launchDomains.baseDomain),
          unanalyzableOp, copyOp.getOperation(),
          /*selectedForeachOp=*/nullptr, /*selectedRecordIndex=*/-1));
    } else {
      FailureOr<SelectedPipeRecords> selected =
          getSelectedPipeRecords(copyOp.getSrc());
      if (failed(selected)) {
        reportInvalidSelectedPipeDefinition(copyOp);
        replacePipeEvents(op, {});
        return;
      }
      appendSelectedPipeEvents(op, copyOp.getSrc(), selected->records,
                               PipeEventKind::ReceiveWait, domain,
                               unanalyzableOp, copyOp.getOperation(),
                               selected->maybeForeachOp, events);
    }
    replacePipeEvents(op, std::move(events));
  }

  /// Record one disjunctive receive-completion event.
  void recordPipeWaitAnyEvent(WaitAnyOp waitOp, const LaunchNodeDomain &domain,
                              Operation *unanalyzableOp) {
    SmallVector<PipeWaitAnyAlternative> alternatives;
    LaunchNodeDomain allowedDomain = launchDomains.baseDomain;
    for (ArrayRef<Operation *> possiblePosts :
         transfers.getReceivePosts(waitOp)) {
      for (Operation *post : possiblePosts) {
        CopyOp copyOp = cast<CopyOp>(post);
        if (auto pipeType = dyn_cast<PipeType>(copyOp.getSrc().getType())) {
          alternatives.push_back({pipeType, post});
          allowedDomain =
              allowedDomain.intersectWith(getPipeDestinationLaunchNodeDomain(
                  pipeType, launchDomains.baseDomain));
          continue;
        }
        FailureOr<SelectedPipeRecords> selected =
            getSelectedPipeRecords(copyOp.getSrc());
        if (failed(selected)) {
          reportInvalidSelectedPipeDefinition(copyOp);
          return;
        }
        allowedDomain =
            allowedDomain.intersectWith(getPipeRecordsRoleLaunchNodeDomain(
                selected->records, PipeRole::Destination));
        for (PipeRecordAttr record : selected->records.getPipes()) {
          alternatives.push_back(
              {getPipeTypeFromRecord(selected->records.getContext(), record,
                                     selected->records.getPipeNetId()),
               post});
        }
      }
    }
    assert(!alternatives.empty() && "wait-any requires a candidate");
    CopyOp representativePost = cast<CopyOp>(alternatives.front().receivePost);
    PipeEvent event = makePipeEvent(
        waitOp.getOperation(), representativePost.getSrc(),
        alternatives.front().pipeType, DeviceTransferAttr(),
        PipeEventKind::ReceiveWaitAny, domain, allowedDomain, unanalyzableOp,
        /*receivePost=*/nullptr, /*selectedForeachOp=*/nullptr,
        /*selectedRecordIndex=*/-1);
    event.waitAnyAlternatives = std::move(alternatives);
    replacePipeEvents(waitOp.getOperation(), {std::move(event)});
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
  Operation *unanalyzableOp = nullptr;
  PipeNetScopeLaunchNodeDomains scope;
};

/// Cache launch-node dataflow results shared by the two read-only verifiers.
class PipeNetLaunchNodeDomainAnalysis {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PipeNetLaunchNodeDomainAnalysis)

  /// Compute and retain the final launch-node dataflow facts for `root`.
  explicit PipeNetLaunchNodeDomainAnalysis(Operation *root) {
    auto module = mlir::cast<ModuleOp>(root);
    state.initialize(module);
    if (!state.hasPipes()) {
      valid = true;
      return;
    }
    if (!state.hasLaunchGrid) {
      return;
    }

    DataFlowSolver solver;
    dataflow::loadBaselineAnalyses(solver);
    LaunchNodeDomainAnalysisOptions options;
    options.computeRegionDomain =
        [&](Operation *op, unsigned) -> std::optional<LaunchNodeDomain> {
      auto ifOp = dyn_cast<scf::IfOp>(op);
      if (ifOp && getReadyReceiveSelection(ifOp.getCondition())) {
        return state.baseDomain;
      }
      return std::nullopt;
    };
    options.operationCallback = [&](Operation *op,
                                    const LaunchNodeDomain &domain,
                                    Operation *unanalyzableOp) {
      operationDomains[op] = {domain, unanalyzableOp};
    };
    options.pipeNetScopeCallback =
        [&](PipeNetScopeOp scopeOp, const LaunchNodeDomain &domain,
            Operation *unanalyzableOp,
            const PipeNetScopeLaunchNodeDomains &scope) {
          scopeDomains[scopeOp.getOperation()] = {domain, unanalyzableOp,
                                                  scope};
        };
    solver.load<LaunchNodeDomainAnalysis>(state, options);
    valid = succeeded(solver.initializeAndRun(module)) && !state.sawError;
  }

  /// Return whether dataflow completed without errors.
  bool isValid() const { return valid; }

  /// Return the module metadata and cached execution-count analyses.
  const LaunchNodeDomainState &getState() const { return state; }

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
  LaunchNodeDomainState state;
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
    os << state.launchDomains.netName(id) << "." << method << "()";
  }
  return buffer;
}

/// Verify that a receive completion wait executes only at destinations of its
/// defining pipe receive.
void verifyPipeWaitGuard(WaitOp waitOp, const LaunchNodeDomain &domain,
                         Operation *unanalyzableOp, ModuleState &state) {
  std::optional<CopyOp> maybeCopyOp = state.transfers.getReceivePost(waitOp);
  if (!maybeCopyOp) {
    return;
  }

  CopyOp copyOp = *maybeCopyOp;
  int64_t netId;
  LaunchNodeDomain allowedDomain;
  if (auto pipeType = mlir::dyn_cast<PipeType>(copyOp.getSrc().getType())) {
    netId = pipeType.getPipeNetId();
    allowedDomain = getPipeDestinationLaunchNodeDomain(
        pipeType, state.launchDomains.baseDomain);
  } else {
    FailureOr<SelectedPipeRecords> selected =
        getSelectedPipeRecords(copyOp.getSrc());
    if (failed(selected)) {
      state.reportInvalidSelectedPipeDefinition(copyOp);
      return;
    }
    netId = selected->records.getPipeNetId();
    allowedDomain = getPipeRecordsRoleLaunchNodeDomain(selected->records,
                                                       PipeRole::Destination);
  }
  std::string name = state.launchDomains.netName(netId);
  std::string message;
  llvm::raw_string_ostream(message)
      << "this `ttl.wait` waits for a pipe receive on launched nodes "
         "that are not destinations of PipeNet "
      << name << "; keep the wait under the same `if " << name
      << ".is_dst(): ...` or `" << name
      << ".if_dst(...)` guard as the receive copy";
  checkKnownSubset(waitOp, domain, allowedDomain, unanalyzableOp, message,
                   {{netId, PipeRole::Destination}}, state);
}

// Emit an op error when `current` is not a subset of `allowed`. Attaches an
// example offending coord, the unanalyzable predicate location (if any), and
// declaration notes for each named PipeNet role.
void checkKnownSubset(Operation *op, const LaunchNodeDomain &current,
                      const LaunchNodeDomain &allowed,
                      Operation *unanalyzableOp, Twine primaryMessage,
                      ArrayRef<std::pair<int64_t, PipeRole>> roles,
                      ModuleState &state) {
  if (current.isUpperBoundSubsetOf(allowed)) {
    return;
  }
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
  LaunchNodeDomain extra = current.subtract(allowed);
  auto diag = op->emitOpError() << primaryMessage;
  if (extra.known && !extra.nodes.empty()) {
    LaunchNodeCoord example = *extra.nodes.begin();
    diag.attachNote() << "example node where the guard does not hold: "
                      << "core_x=" << example.x << ", core_y=" << example.y;
  }
  for (auto &p : roles) {
    auto it = state.launchDomains.pipeNetLocs.find(p.first);
    if (it == state.launchDomains.pipeNetLocs.end() || it->second.empty()) {
      continue;
    }
    diag.attachNote(it->second.front())
        << "PipeNet " << state.launchDomains.netName(p.first)
        << " declared here";
  }
  state.sawError = true;
}

// Diagnose a `ttl.copy` whose endpoint is a pipe but whose enclosing domain
// extends outside the pipe's source/destination set.
void verifyCopy(CopyOp copyOp, const LaunchNodeDomain &current,
                Operation *unanalyzable, ModuleState &state) {
  if (auto dstPipeType = mlir::dyn_cast<PipeType>(copyOp.getDst().getType())) {
    int64_t netId = dstPipeType.getPipeNetId();
    std::string name = state.launchDomains.netName(netId);
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
  FailureOr<SelectedPipeRecords> selectedDst =
      getSelectedPipeRecords(copyOp.getDst());
  if (succeeded(selectedDst)) {
    int64_t netId = selectedDst->records.getPipeNetId();
    std::string name = state.launchDomains.netName(netId);
    std::string msg;
    llvm::raw_string_ostream(msg)
        << "this `ttl.copy(buffer, pipe)` sends data on PipeNet " << name
        << " from a node that is not a source of any pipe in that net; "
           "wrap the copy in `"
        << name << ".if_src(...)` or guard with `if " << name
        << ".is_src(): ...`";
    checkKnownSubset(copyOp, current,
                     getPipeRecordsRoleLaunchNodeDomain(selectedDst->records,
                                                        PipeRole::Source),
                     unanalyzable, msg, {{netId, PipeRole::Source}}, state);
    return;
  }
  if (auto srcPipeType = mlir::dyn_cast<PipeType>(copyOp.getSrc().getType())) {
    int64_t netId = srcPipeType.getPipeNetId();
    std::string name = state.launchDomains.netName(netId);
    std::string msg;
    llvm::raw_string_ostream(msg)
        << "this `ttl.copy(pipe, buffer)` receives data from PipeNet " << name
        << " on a node that is not a destination of any pipe in that "
           "net; wrap the copy in `"
        << name << ".if_dst(...)` or guard with `if " << name
        << ".is_dst(): ...`";
    checkKnownSubset(copyOp, current,
                     getPipeDestinationLaunchNodeDomain(
                         srcPipeType, state.launchDomains.baseDomain),
                     unanalyzable, msg, {{netId, PipeRole::Destination}},
                     state);
    return;
  }
  FailureOr<SelectedPipeRecords> selectedSrc =
      getSelectedPipeRecords(copyOp.getSrc());
  if (succeeded(selectedSrc)) {
    int64_t netId = selectedSrc->records.getPipeNetId();
    std::string name = state.launchDomains.netName(netId);
    std::string msg;
    llvm::raw_string_ostream(msg)
        << "this `ttl.copy(pipe, buffer)` receives data from PipeNet " << name
        << " on a node that is not a destination of any pipe in that "
           "net; wrap the copy in `"
        << name << ".if_dst(...)` or guard with `if " << name
        << ".is_dst(): ...`";
    checkKnownSubset(copyOp, current,
                     getPipeRecordsRoleLaunchNodeDomain(selectedSrc->records,
                                                        PipeRole::Destination),
                     unanalyzable, msg, {{netId, PipeRole::Destination}},
                     state);
  }
}

/// Verify that a `ttl.pipenet_scope` body only executes on nodes participating
/// in at least one selected PipeNet role.
void verifyPipeNetScope(PipeNetScopeOp scopeOp, const LaunchNodeDomain &domain,
                        Operation *unanalyzableOp,
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
    llvm::interleaveComma(uniqueIds, os, [&](int64_t id) {
      os << state.launchDomains.netName(id);
    });
    os << " on launched nodes that are not part of "
       << (uniqueIds.size() == 1 ? "that net" : "those nets")
       << "; wrap the surrounding work in `if "
       << formatGuardExpression(scope.roles, state)
       << ": ...` so non-participating nodes skip it";
  }
  checkKnownSubset(scopeOp, domain, scope.domain, unanalyzableOp, msg,
                   scope.roles, state);
}

/// Dispatch the generic launch-domain callback to the checks that care about
/// a specific operation kind.
void recordGuardOperation(Operation *op, const LaunchNodeDomain &domain,
                          Operation *unanalyzableOp, ModuleState &state) {
  assert(state.dfbIdentities && "guard verification requires DFB identities");
  TypeSwitch<Operation *>(op)
      .Case<CopyOp>(
          [&](CopyOp copy) { verifyCopy(copy, domain, unanalyzableOp, state); })
      .Case<WaitOp>([&](WaitOp wait) {
        verifyPipeWaitGuard(wait, domain, unanalyzableOp, state);
      })
      .Case<CBPushOp>([&](CBPushOp push) {
        FailureOr<int64_t> dfbId =
            state.dfbIdentities->getLogicalId(push.getCb());
        assert(succeeded(dfbId) && "DFB operands were verified");
        state.dfbProducerDomains[*dfbId] =
            state.dfbProducerDomains[*dfbId].unionWith(domain);
      })
      .Case<CBWaitOp>([&](CBWaitOp wait) {
        FailureOr<int64_t> dfbId =
            state.dfbIdentities->getLogicalId(wait.getCb());
        assert(succeeded(dfbId) && "DFB operands were verified");
        state.waitUses.push_back({wait, domain, *dfbId});
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
      })
      .Case<WaitAnyOp>([&](WaitAnyOp wait) {
        state.recordPipeWaitAnyEvent(wait, domain, unanalyzableOp);
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

/// Validate logical DFB identities and every operand read by guard analysis.
LogicalResult
verifyGuardDFBIdentities(ModuleOp module,
                         const DFBLogicalIdentityAnalysis &dfbIdentities) {
  if (!dfbIdentities.succeeded()) {
    Operation *errorOperation = dfbIdentities.getErrorOperation();
    if (!errorOperation) {
      errorOperation = module.getOperation();
    }
    errorOperation->emitOpError() << dfbIdentities.getErrorMessage();
    return failure();
  }

  return verifyDFBOperandIdentities(
      module, "ttl-verify-pipenet-guards",
      [](Operation *operation) { return isa<CBPushOp, CBWaitOp>(operation); },
      [&](Value dfb) { return dfbIdentities.getLogicalId(dfb); },
      "`ttl.cb_push` and `ttl.cb_wait` DFB", DFBIdentityRequirement::Logical);
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

/// Pipe synchronization event specialized to one hardware execution location.
struct PipeScheduleNode {
  Operation *op;
  PipeType pipeType;
  LaunchExecutionLocation location;
  PipeEventKind kind;
  /// Static receive post whose token is observed by this wait.
  Operation *receivePost;
  func::FuncOp kernelFunction;
  SmallVector<PipeCallSite> callSites;
  SmallVector<ActivePipeNetRecord> activeRecords;
  std::optional<std::uint64_t> executionCountDivisor;
  SmallVector<PipeScheduleEdge> successors;
  SmallVector<PipeWaitAnyAlternative> waitAnyAlternatives;
  SmallVector<PipeScheduleNodeId> waitAnyCompletingSends;
};

/// Retain the sends for one logical pipe in deterministic traversal order.
struct PipeOccurrences {
  PipeType pipeType;
  DeviceTransferAttr deviceTransfer;
  SmallVector<PipeScheduleNodeId> sends;
};

using PipeIdentity = std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t,
                                int64_t, int64_t, DeviceTransferAttr>;

using PipeCoordIdentity =
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
               DeviceTransferAttr, int64_t, int64_t>;

/// Return a stable identity for one pipe endpoint relation.
PipeIdentity getPipeIdentity(PipeType pipeType,
                             DeviceTransferAttr deviceTransfer) {
  return {pipeType.getPipeNetId(), pipeType.getSrcX(),
          pipeType.getSrcY(),      pipeType.getDstStartX(),
          pipeType.getDstEndX(),   pipeType.getDstStartY(),
          pipeType.getDstEndY(),   deviceTransfer};
}

/// Return a stable identity for one pipe endpoint relation at one launch node.
PipeCoordIdentity getPipeCoordIdentity(PipeType pipeType,
                                       DeviceTransferAttr deviceTransfer,
                                       LaunchNodeCoord coord) {
  auto [pipeNetId, srcX, srcY, dstStartX, dstEndX, dstStartY, dstEndY,
        transfer] = getPipeIdentity(pipeType, deviceTransfer);
  return {pipeNetId, srcX,    srcY,     dstStartX, dstEndX,
          dstStartY, dstEndY, transfer, coord.x,   coord.y};
}

/// Add one graph node for a synchronization event at one call-site occurrence.
PipeScheduleNodeId
addPipeScheduleNode(SmallVectorImpl<PipeScheduleNode> &nodes,
                    const PipeEvent &event, LaunchExecutionLocation location,
                    func::FuncOp kernelFunction,
                    ArrayRef<PipeCallSite> callSites,
                    ArrayRef<ActivePipeNetRecord> activeRecords,
                    std::optional<std::uint64_t> executionCountDivisor) {
  PipeScheduleNodeId nodeId = nodes.size();
  nodes.push_back({event.op,
                   event.pipeType,
                   location,
                   event.kind,
                   event.receivePost,
                   kernelFunction,
                   SmallVector<PipeCallSite>(callSites),
                   SmallVector<ActivePipeNetRecord>(activeRecords),
                   executionCountDivisor,
                   {},
                   event.waitAnyAlternatives,
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

/// Return true when two nodes were expanded through the same direct calls.
bool haveSamePipeCallSites(ArrayRef<PipeCallSite> lhs,
                           ArrayRef<PipeCallSite> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::all_of(llvm::zip(lhs, rhs), [](auto callSites) {
           return std::get<0>(callSites).call == std::get<1>(callSites).call &&
                  std::get<0>(callSites).callee ==
                      std::get<1>(callSites).callee;
         });
}

/// Return the receiver-post node whose token is observed by `waitNode`.
std::optional<PipeScheduleNodeId>
findReceivePostNodeForWait(ArrayRef<PipeScheduleNode> nodes,
                           PipeScheduleNodeId waitNodeId,
                           ArrayRef<PipeScheduleNodeId> candidatePostNodes) {
  const PipeScheduleNode &waitNode = nodes[waitNodeId];
  assert(waitNode.kind == PipeEventKind::ReceiveWait && waitNode.receivePost &&
         "receive wait must identify its post");
  auto postIt =
      llvm::find_if(candidatePostNodes, [&](PipeScheduleNodeId postId) {
        const PipeScheduleNode &postNode = nodes[postId];
        return postNode.op == waitNode.receivePost &&
               postNode.location == waitNode.location &&
               postNode.kernelFunction == waitNode.kernelFunction &&
               haveSamePipeCallSites(postNode.callSites, waitNode.callSites);
      });
  return postIt == candidatePostNodes.end()
             ? std::nullopt
             : std::optional<PipeScheduleNodeId>(*postIt);
}

/// Return the receiver-post node represented by one wait-any alternative.
std::optional<PipeScheduleNodeId> findReceivePostNodeForWaitAnyAlternative(
    ArrayRef<PipeScheduleNode> nodes, PipeScheduleNodeId waitNodeId,
    const PipeWaitAnyAlternative &alternative,
    ArrayRef<PipeScheduleNodeId> candidatePostNodes) {
  const PipeScheduleNode &waitNode = nodes[waitNodeId];
  auto postIt =
      llvm::find_if(candidatePostNodes, [&](PipeScheduleNodeId postId) {
        const PipeScheduleNode &postNode = nodes[postId];
        return postNode.op == alternative.receivePost &&
               postNode.location == waitNode.location &&
               postNode.kernelFunction == waitNode.kernelFunction &&
               haveSamePipeCallSites(postNode.callSites, waitNode.callSites) &&
               postNode.pipeType == alternative.pipeType;
      });
  return postIt == candidatePostNodes.end()
             ? std::nullopt
             : std::optional<PipeScheduleNodeId>(*postIt);
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

/// Compute the synchronization events that can complete under mandatory graph
/// dependencies and disjunctive wait-any alternatives.
SmallVector<bool>
computeExecutablePipeScheduleNodes(ArrayRef<PipeScheduleNode> nodes) {
  SmallVector<std::size_t> predecessorCounts(nodes.size(), 0);
  for (const PipeScheduleNode &node : nodes) {
    for (const PipeScheduleEdge &edge : node.successors) {
      ++predecessorCounts[edge.successor];
    }
  }

  SmallVector<bool> executable(nodes.size(), false);
  bool changed;
  do {
    changed = false;
    for (PipeScheduleNodeId nodeId = 0; nodeId < nodes.size(); ++nodeId) {
      if (executable[nodeId] || predecessorCounts[nodeId] != 0) {
        continue;
      }
      const PipeScheduleNode &node = nodes[nodeId];
      if (node.kind == PipeEventKind::ReceiveWaitAny &&
          !llvm::any_of(node.waitAnyCompletingSends,
                        [&](PipeScheduleNodeId sendNodeId) {
                          return executable[sendNodeId];
                        })) {
        continue;
      }
      executable[nodeId] = true;
      changed = true;
      for (const PipeScheduleEdge &edge : node.successors) {
        assert(predecessorCounts[edge.successor] > 0 &&
               "executable edge predecessor count underflow");
        --predecessorCounts[edge.successor];
      }
    }
  } while (changed);
  return executable;
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
  case PipeEventKind::ReceiveWaitAny:
    os << "receive wait-any";
    break;
  }
  os << " at core_x=" << node.location.node.x
     << ", core_y=" << node.location.node.y;
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
    if (noteCount >= kMaxPipeScheduleDiagnosticNotes) {
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
    auto diag =
        waitNode.op->emitOpError()
        << "receive wait occurs before the send that completes it on "
           "PipeNet "
        << state.launchDomains.netName(waitNode.pipeType.getPipeNetId());
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
    auto diag =
        sendNode.op->emitOpError()
        << "pipe send occurs before the receiver posts a dataflow "
           "buffer reservation on PipeNet "
        << state.launchDomains.netName(sendNode.pipeType.getPipeNetId());
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
              << state.launchDomains.netName(node.pipeType.getPipeNetId())
              << "; post the receive before the dependent send, or place the "
                 "send and receive in separate data-movement threads";

  emitPipeScheduleCycleNotes(diag, nodes, cycle);
  state.sawError = true;
}

/// Constant and unresolved factors in one schedule occurrence count.
struct PipeExecutionCountExpression {
  std::uint64_t constantFactor = 1;
  struct UnresolvedFactor {
    Operation *op;
    Operation *exclusiveAncestor;
  };
  SmallVector<UnresolvedFactor> unresolvedFactors;
};

std::optional<Value> resolveFunctionArgument(BlockArgument argument,
                                             ArrayRef<PipeCallSite> callSites);

/// Return the exact count with selected callback accessors bound to one active
/// record. The shared location analysis handles cases that do not need that
/// binding and retains its cache.
std::optional<std::uint64_t> getExactPipeExecutionCount(
    Operation *op, const LaunchExecutionLocation &location,
    ArrayRef<PipeCallSite> callSites,
    ArrayRef<ActivePipeNetRecord> activeRecords, ModuleState &state) {
  std::optional<std::uint64_t> count =
      getExactExecutionCountAtLaunchLocation(op, location, state.launchDomains);
  if (count || activeRecords.empty()) {
    return count;
  }
  func::FuncOp function = op->getParentOfType<func::FuncOp>();
  if (!function) {
    return std::nullopt;
  }
  auto resolveActiveFunctionArgument = [&](BlockArgument argument) {
    return resolveFunctionArgument(argument, callSites);
  };
  ExecutionCountAnalysis analysis(
      function.getBody(),
      [&](Value value) -> std::optional<llvm::APInt> {
        if (std::optional<llvm::APInt> recordValue =
                evaluateActivePipeNetRecordValue(
                    value, activeRecords, resolveActiveFunctionArgument)) {
          return recordValue;
        }
        return evaluateIntegerAtLaunchLocation(value, location,
                                               state.launchDomains);
      },
      [&](Region &region) {
        return getRegionInvocationCountAtLaunchLocation(region, location,
                                                        state.launchDomains);
      });
  return analysis.getExecutionCount(op);
}

Operation *
getEnclosingActiveRecordLoop(Operation *op,
                             ArrayRef<ActivePipeNetRecord> activeRecords) {
  Operation *enclosingLoop = nullptr;
  for (const ActivePipeNetRecord &activeRecord : activeRecords) {
    if (activeRecord.loopOp->isProperAncestor(op)) {
      enclosingLoop = activeRecord.loopOp;
    }
  }
  return enclosingLoop;
}

// Selected record loops are separate factors in the complete count.
std::optional<std::uint64_t> getSelectedRecordLocalExecutionCount(
    Operation *op, Operation *recordLoop,
    const LaunchExecutionLocation &location, ArrayRef<PipeCallSite> callSites,
    ArrayRef<ActivePipeNetRecord> activeRecords, ModuleState &state) {
  assert(recordLoop && recordLoop->isProperAncestor(op) &&
         "record-local operation must be nested in its record loop");
  auto resolveActiveFunctionArgument = [&](BlockArgument argument) {
    return resolveFunctionArgument(argument, callSites);
  };
  ExecutionCountAnalysis analysis(
      recordLoop->getRegion(0),
      [&](Value value) -> std::optional<llvm::APInt> {
        if (std::optional<llvm::APInt> recordValue =
                evaluateActivePipeNetRecordValue(
                    value, activeRecords, resolveActiveFunctionArgument)) {
          return recordValue;
        }
        return evaluateIntegerAtLaunchLocation(value, location,
                                               state.launchDomains);
      },
      [&](Region &region) -> std::optional<std::uint64_t> {
        if (llvm::any_of(activeRecords,
                         [&](const ActivePipeNetRecord &activeRecord) {
                           return region.getParentOp() == activeRecord.loopOp;
                         })) {
          return 1;
        }
        return getRegionInvocationCountAtLaunchLocation(region, location,
                                                        state.launchDomains);
      });
  return analysis.getExecutionCount(op);
}

/// Separate proven constant factors from operations whose execution counts are
/// symbolic. Multiplication composes caller invocation counts with the local
/// count of an event inside a helper.
std::optional<PipeExecutionCountExpression>
getPipeExecutionCountExpression(const PipeScheduleNode &node,
                                ModuleState &state) {
  PipeExecutionCountExpression expression;
  auto collectFactor = [&](Operation *op) -> LogicalResult {
    Operation *recordLoop =
        getEnclosingActiveRecordLoop(op, node.activeRecords);
    std::optional<std::uint64_t> maybeCount =
        recordLoop
            ? getSelectedRecordLocalExecutionCount(
                  op, recordLoop, node.location, node.callSites,
                  node.activeRecords, state)
            : getExactPipeExecutionCount(op, node.location, node.callSites,
                                         node.activeRecords, state);
    if (!maybeCount) {
      expression.unresolvedFactors.push_back({op, recordLoop});
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
  for (const ActivePipeNetRecord &activeRecord : node.activeRecords) {
    if (failed(collectFactor(activeRecord.loopOp))) {
      return std::nullopt;
    }
  }
  if (failed(collectFactor(node.op))) {
    return std::nullopt;
  }
  if (!node.activeRecords.empty()) {
    return expression;
  }
  if (!node.executionCountDivisor) {
    return std::nullopt;
  }
  if (*node.executionCountDivisor != 1) {
    if (!expression.unresolvedFactors.empty() ||
        expression.constantFactor % *node.executionCountDivisor != 0) {
      return std::nullopt;
    }
    expression.constantFactor /= *node.executionCountDivisor;
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
      maybeLhs->unresolvedFactors.size() !=
          maybeRhs->unresolvedFactors.size()) {
    return false;
  }
  return llvm::all_of(
      llvm::zip(maybeLhs->unresolvedFactors, maybeRhs->unresolvedFactors),
      [&](auto pair) {
        auto resolveLhsFunctionArgument = [&](BlockArgument argument) {
          return resolveFunctionArgument(argument, lhs.callSites);
        };
        auto resolveRhsFunctionArgument = [&](BlockArgument argument) {
          return resolveFunctionArgument(argument, rhs.callSites);
        };
        const PipeExecutionCountExpression::UnresolvedFactor &lhsFactor =
            std::get<0>(pair);
        const PipeExecutionCountExpression::UnresolvedFactor &rhsFactor =
            std::get<1>(pair);
        auto evaluateLhsContextValue = [&](Value value) {
          return evaluateActivePipeNetRecordValue(value, lhs.activeRecords,
                                                  resolveLhsFunctionArgument);
        };
        auto evaluateRhsContextValue = [&](Value value) {
          return evaluateActivePipeNetRecordValue(value, rhs.activeRecords,
                                                  resolveRhsFunctionArgument);
        };
        return proveEqualUnresolvedExecutionCountWithinScopesAtLaunchLocations(
            lhsFactor.op, lhsFactor.exclusiveAncestor, lhs.location,
            rhsFactor.op, rhsFactor.exclusiveAncestor, rhs.location,
            state.launchDomains, evaluateLhsContextValue,
            evaluateRhsContextValue, resolveLhsFunctionArgument,
            resolveRhsFunctionArgument);
      });
}

/// Return true when the schedule graph orders `successor` after `predecessor`.
bool pipeScheduleNodeReaches(ArrayRef<PipeScheduleNode> nodes,
                             PipeScheduleNodeId predecessor,
                             PipeScheduleNodeId successor) {
  SmallVector<PipeScheduleNodeId> worklist{predecessor};
  llvm::DenseSet<PipeScheduleNodeId> visited{predecessor};
  for (std::size_t index = 0; index < worklist.size(); ++index) {
    for (const PipeScheduleEdge &edge : nodes[worklist[index]].successors) {
      if (edge.successor == successor) {
        return true;
      }
      if (visited.insert(edge.successor).second) {
        worklist.push_back(edge.successor);
      }
    }
  }
  return false;
}

/// Proves that receiver-published rendezvous state has at most one outstanding
/// post for each pipe and receiver.
class PipeRendezvousLifetimeAnalysis {
public:
  PipeRendezvousLifetimeAnalysis(
      ArrayRef<PipeScheduleNode> nodes,
      const llvm::DenseMap<PipeScheduleNodeId, PipeScheduleNodeId>
          &completingSendByPost,
      const llvm::DenseMap<PipeScheduleNodeId, SmallVector<PipeScheduleNodeId>>
          &waitsByPost,
      ModuleState &state)
      : nodes(nodes), completingSendByPost(completingSendByPost),
        waitsByPost(waitsByPost), state(state) {}

  /// Verify one receiver's posts in static execution order.
  LogicalResult verify(ArrayRef<PipeScheduleNodeId> postNodes) const {
    LogicalResult result = success();
    for (PipeScheduleNodeId postNodeId : postNodes) {
      if (failed(verifyRepeatedPost(postNodeId))) {
        result = failure();
      }
    }
    for (auto adjacentPosts : llvm::zip(postNodes, postNodes.drop_front())) {
      if (failed(verifyAdjacentPosts(std::get<0>(adjacentPosts),
                                     std::get<1>(adjacentPosts)))) {
        result = failure();
      }
    }
    return result;
  }

private:
  /// Return true when `completion` executes after `post` in each occurrence of
  /// the same block and under an equal dynamic execution count.
  bool
  provesSameIterationCompletion(PipeScheduleNodeId postNodeId,
                                PipeScheduleNodeId completionNodeId) const {
    const PipeScheduleNode &post = nodes[postNodeId];
    const PipeScheduleNode &completion = nodes[completionNodeId];
    return post.kernelFunction == completion.kernelFunction &&
           haveSamePipeCallSites(post.callSites, completion.callSites) &&
           post.op->getBlock() == completion.op->getBlock() &&
           post.op->isBeforeInBlock(completion.op) &&
           proveEqualPipeScheduleNodeCounts(post, completion, state);
  }

  LogicalResult verifyRepeatedPost(PipeScheduleNodeId postNodeId) const {
    auto sendIt = completingSendByPost.find(postNodeId);
    if (sendIt == completingSendByPost.end()) {
      return success();
    }
    std::optional<PipeExecutionCountExpression> maybeCount =
        getPipeExecutionCountExpression(nodes[postNodeId], state);
    bool mayRepeat = !maybeCount || maybeCount->constantFactor > 1 ||
                     !maybeCount->unresolvedFactors.empty();
    if (!mayRepeat ||
        provesSameIterationCompletion(postNodeId, sendIt->second)) {
      return success();
    }
    auto waitsIt = waitsByPost.find(postNodeId);
    if (waitsIt != waitsByPost.end() &&
        llvm::any_of(waitsIt->second, [&](PipeScheduleNodeId waitNodeId) {
          return provesSameIterationCompletion(postNodeId, waitNodeId);
        })) {
      return success();
    }

    const PipeScheduleNode &post = nodes[postNodeId];
    post.op->emitOpError()
        << "cannot prove that each repeated receiver post is consumed before "
           "the next post on PipeNet "
        << state.launchDomains.netName(post.pipeType.getPipeNetId())
        << " at core_x=" << post.location.node.x
        << ", core_y=" << post.location.node.y
        << "; receiver-published addressing supports one outstanding post per "
           "pipe";
    return failure();
  }

  LogicalResult verifyAdjacentPosts(PipeScheduleNodeId previousPostId,
                                    PipeScheduleNodeId nextPostId) const {
    auto sendIt = completingSendByPost.find(previousPostId);
    if (sendIt == completingSendByPost.end() ||
        mlir::insideMutuallyExclusiveRegions(nodes[previousPostId].op,
                                             nodes[nextPostId].op) ||
        pipeScheduleNodeReaches(nodes, sendIt->second, nextPostId)) {
      return success();
    }

    const PipeScheduleNode &previousPost = nodes[previousPostId];
    const PipeScheduleNode &nextPost = nodes[nextPostId];
    auto diag = nextPost.op->emitOpError()
                << "receiver post may overwrite an outstanding posted address "
                   "on PipeNet "
                << state.launchDomains.netName(nextPost.pipeType.getPipeNetId())
                << " at core_x=" << nextPost.location.node.x
                << ", core_y=" << nextPost.location.node.y
                << "; receiver-published addressing supports one outstanding "
                   "post per pipe";
    diag.attachNote(previousPost.op->getLoc())
        << "the preceding receiver post is not proven consumed before this "
           "post";
    return failure();
  }

  ArrayRef<PipeScheduleNode> nodes;
  const llvm::DenseMap<PipeScheduleNodeId, PipeScheduleNodeId>
      &completingSendByPost;
  const llvm::DenseMap<PipeScheduleNodeId, SmallVector<PipeScheduleNodeId>>
      &waitsByPost;
  ModuleState &state;
};

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
              << state.launchDomains.netName(
                     nodes[unmatchedNode].pipeType.getPipeNetId())
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
    if (noteCount >= kMaxPipeScheduleDiagnosticNotes) {
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
                  << state.launchDomains.netName(
                         nodes[successor].pipeType.getPipeNetId())
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
/// to execute at `location`.
bool hasZeroExecutionCount(ArrayRef<PipeCallSite> callSites, Operation *op,
                           const LaunchExecutionLocation &location,
                           ArrayRef<ActivePipeNetRecord> activeRecords,
                           ModuleState &state) {
  if (llvm::any_of(callSites, [&](const PipeCallSite &callSite) {
        std::optional<std::uint64_t> maybeCount = getExactPipeExecutionCount(
            callSite.call, location, callSites, activeRecords, state);
        return maybeCount && *maybeCount == 0;
      })) {
    return true;
  }
  std::optional<std::uint64_t> maybeCount =
      getExactPipeExecutionCount(op, location, callSites, activeRecords, state);
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

/// Return true when `region` contains a pipe event or a direct call that
/// expands to pipe events during schedule construction.
bool regionContributesPipeEvents(
    Region &region, const ModuleState &state,
    const llvm::DenseSet<Operation *> &functionsWithPipeEvents,
    SymbolTableCollection &symbolTables) {
  WalkResult walkResult = region.walk([&](Operation *op) {
    if (state.pipeEventIndices.contains(op)) {
      return WalkResult::interrupt();
    }
    auto callOp = mlir::dyn_cast<func::CallOp>(op);
    if (!callOp) {
      return WalkResult::advance();
    }
    func::FuncOp callee = symbolTables.lookupNearestSymbolFrom<func::FuncOp>(
        callOp, callOp.getCalleeAttr());
    return callee && functionsWithPipeEvents.contains(callee.getOperation())
               ? WalkResult::interrupt()
               : WalkResult::advance();
  });
  return walkResult.wasInterrupted();
}

/// Reject control flow whose execution order cannot be represented by a
/// linear sequence of synchronization events.
LogicalResult verifyPipeEventRegionsHaveOneBlock(
    ModuleOp module, const ModuleState &state,
    const llvm::DenseSet<Operation *> &functionsWithPipeEvents,
    SymbolTableCollection &symbolTables) {
  LogicalResult result = success();
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    if (!functionsWithPipeEvents.contains(function.getOperation())) {
      continue;
    }
    function->walk([&](Operation *op) {
      for (Region &region : op->getRegions()) {
        if (region.hasOneBlock() ||
            !regionContributesPipeEvents(region, state, functionsWithPipeEvents,
                                         symbolTables)) {
          continue;
        }
        if (auto regionFunction = mlir::dyn_cast<func::FuncOp>(op)) {
          regionFunction.emitOpError()
              << "cannot verify PipeNet synchronization in multi-block "
                 "function @"
              << regionFunction.getSymName()
              << "; schedule verification requires every region containing "
                 "pipe events to have one block";
        } else {
          op->emitOpError()
              << "cannot verify PipeNet synchronization in a multi-block "
                 "region of this operation; schedule verification requires "
                 "every region containing pipe events to have one block";
        }
        result = failure();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
  return result;
}

/// Visit pipe events in the order executed by one kernel thread.
///
/// Direct helper calls are expanded at each call site. A PipeNet foreach body
/// is expanded once per matching record before execution continues after the
/// foreach operation.
WalkResult walkPipeEventsInProgramOrder(
    Operation *op, LaunchNodeCoord coord, ModuleState &state,
    const llvm::DenseSet<Operation *> &functionsWithPipeEvents,
    SymbolTableCollection &symbolTables,
    SmallVectorImpl<func::FuncOp> &activeFunctions,
    SmallVectorImpl<PipeCallSite> &callSites,
    SmallVectorImpl<ActivePipeNetRecord> &activeRecords,
    llvm::DenseSet<Operation *> &diagnosedRecursiveCalls,
    llvm::function_ref<
        WalkResult(const PipeEvent &, const LaunchExecutionLocation &,
                   ArrayRef<PipeCallSite>, ArrayRef<ActivePipeNetRecord>,
                   std::optional<std::uint64_t>)>
        visitEvent) {
  auto resolveGeneratedRecordLoop =
      [](Operation *) -> std::optional<PipeNetRecordLoop> {
    return std::nullopt;
  };
  auto visitOperation = [&](Operation *currentOp, ArrayRef<ActivePipeNetRecord>
                                                      currentActiveRecords) {
    auto eventIt = state.pipeEventIndices.find(currentOp);
    if (eventIt != state.pipeEventIndices.end()) {
      for (std::size_t eventIndex : eventIt->second) {
        const PipeEvent &event = state.pipeEvents[eventIndex];
        if (!knownLaunchNodeDomainContains(event.scheduleDomain, coord)) {
          continue;
        }
        if (event.selectedForeachOp) {
          std::optional<std::uint64_t> activeRecordIndex =
              getActivePipeNetRecordIndex(currentActiveRecords,
                                          event.selectedForeachOp);
          assert(activeRecordIndex &&
                 "selected pipe event must be nested in its foreach "
                 "operation");
          if (*activeRecordIndex !=
              static_cast<std::uint64_t>(event.selectedRecordIndex)) {
            continue;
          }
        }
        auto resolveActiveFunctionArgument = [&](BlockArgument argument) {
          SmallVector<Value> operands;
          if (std::optional<Value> operand =
                  resolveFunctionArgument(argument, callSites)) {
            operands.push_back(*operand);
          }
          return FailureOr<SmallVector<Value>>(std::move(operands));
        };
        PipeEvent resolvedEvent = event;
        if (event.selectedRecordIndex < 0) {
          FailureOr<std::optional<DeviceTransferAttr>> maybeDeviceTransfer =
              findUniquePipeDeviceTransfer(state.valueOrigins, event.pipe,
                                           resolveActiveFunctionArgument);
          if (failed(maybeDeviceTransfer)) {
            event.op->emitOpError()
                << "requires every possible pipe definition at this call "
                   "site to use the same logical-device transfer";
            state.sawError = true;
            return WalkResult::interrupt();
          }
          resolvedEvent.deviceTransfer =
              maybeDeviceTransfer->value_or(DeviceTransferAttr());
        }
        FailureOr<LaunchExecutionLocation> maybeLocation =
            getPipeEventExecutionLocation(resolvedEvent, coord);
        if (failed(maybeLocation)) {
          event.op->emitOpError(
              "device-range fabric transfers require scatter target "
              "lowering");
          state.sawError = true;
          return WalkResult::interrupt();
        }
        ActivePipeNetExecution activeExecution = evaluateActivePipeNetExecution(
            currentActiveRecords, *maybeLocation, resolveGeneratedRecordLoop);
        if (!activeExecution.mayExecute) {
          continue;
        }
        if (!hasZeroExecutionCount(callSites, event.op, *maybeLocation,
                                   currentActiveRecords, state) &&
            visitEvent(resolvedEvent, *maybeLocation, callSites,
                       currentActiveRecords, activeExecution.countDivisor)
                .wasInterrupted()) {
          return WalkResult::interrupt();
        }
      }
    }

    auto callOp = mlir::dyn_cast<func::CallOp>(currentOp);
    if (!callOp) {
      return WalkResult::advance();
    }
    func::FuncOp callee = symbolTables.lookupNearestSymbolFrom<func::FuncOp>(
        callOp, callOp.getCalleeAttr());
    if (!callee || !functionsWithPipeEvents.contains(callee.getOperation()) ||
        hasZeroExecutionCount({}, callOp.getOperation(),
                              LaunchExecutionLocation(coord),
                              currentActiveRecords, state)) {
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
                activeFunctions, callSites, activeRecords,
                diagnosedRecursiveCalls, visitEvent)
                .wasInterrupted()) {
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  };

  return walkPipeNetOpsInProgramOrder(op, coord, resolveGeneratedRecordLoop,
                                      visitOperation, activeRecords);
}

// Verify synchronization dependencies implied by pipe operations. Receive-side
// ttl.copy makes a reserved DFB slot available to the sender, while ttl.wait on
// its handle waits for payload arrival. Modeling availability and completion as
// separate events preserves asynchronous semantics and detects wait-for cycles.
void verifyPipeScheduleCycles(ModuleOp module, ModuleState &state) {
  SmallVector<PipeScheduleNode, 0> nodes;
  llvm::MapVector<PipeIdentity, PipeOccurrences> pipeOccurrences;
  llvm::MapVector<PipeCoordIdentity, SmallVector<PipeScheduleNodeId>>
      receivePostNodes;
  llvm::DenseMap<PipeCoordIdentity, SmallVector<PipeScheduleNodeId>>
      receiveWaitNodes;
  llvm::DenseMap<Operation *, SmallVector<PipeScheduleNodeId>>
      receivePostNodesByOperation;
  SmallVector<PipeScheduleNodeId> allReceiveWaitNodes;
  SmallVector<PipeScheduleNodeId> allReceiveWaitAnyNodes;
  llvm::DenseMap<Operation *, SmallVector<PipeScheduleNodeId>>
      receiveWaitAnyNodesByOperation;
  SymbolTableCollection symbolTables;
  llvm::DenseSet<Operation *> functionsWithPipeEvents =
      getFunctionsWithPipeEvents(module, state, symbolTables);
  llvm::DenseSet<Operation *> reachableFunctions =
      getFunctionsReachableFromKernelThreads(module, symbolTables);
  if (failed(verifyPipeEventFunctionsReachable(state, reachableFunctions)) ||
      failed(verifyPipeEventRegionsHaveOneBlock(
          module, state, functionsWithPipeEvents, symbolTables))) {
    state.sawError = true;
    return;
  }
  llvm::DenseSet<Operation *> diagnosedRecursiveCalls;
  // Bound helper expansion per launch node so a larger receiver domain does
  // not reduce the number of static events accepted at each node.
  std::map<LaunchExecutionLocation, std::size_t> scheduleNodeCounts;
  LaunchNodeDomain eventDomain;
  for (const PipeEvent &event : state.pipeEvents) {
    eventDomain = eventDomain.unionWith(event.scheduleDomain);
  }
  const std::set<LaunchNodeCoord> &scheduleCoords = eventDomain.nodes;

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
      SmallVector<ActivePipeNetRecord> activeRecords;
      std::map<LaunchExecutionLocation, PipeScheduleNodeId> lastNodeByLocation;
      for (Block &block : function.getBody()) {
        for (Operation &op : block) {
          WalkResult walkResult = walkPipeEventsInProgramOrder(
              &op, coord, state, functionsWithPipeEvents, symbolTables,
              activeFunctions, callSites, activeRecords,
              diagnosedRecursiveCalls,
              [&](const PipeEvent &event,
                  const LaunchExecutionLocation &location,
                  ArrayRef<PipeCallSite> activeCallSites,
                  ArrayRef<ActivePipeNetRecord> activePipeNetRecords,
                  std::optional<std::uint64_t> executionCountDivisor) {
                std::size_t &scheduleNodeCount = scheduleNodeCounts[location];
                if (scheduleNodeCount >= kMaxPipeScheduleNodesPerLaunchNode) {
                  event.op->emitOpError()
                      << "cannot verify PipeNet synchronization because the "
                         "schedule contains more than "
                      << kMaxPipeScheduleNodesPerLaunchNode
                      << " pipe events at core_x=" << coord.x
                      << ", core_y=" << coord.y
                      << " after expanding helper calls";
                  state.sawError = true;
                  return WalkResult::interrupt();
                }
                SmallVector<PipeScheduleNodeId> *matchingNodes = nullptr;
                if (event.kind == PipeEventKind::Send) {
                  PipeIdentity pipeIdentity =
                      getPipeIdentity(event.pipeType, event.deviceTransfer);
                  auto pipeIt =
                      pipeOccurrences
                          .try_emplace(pipeIdentity,
                                       PipeOccurrences{event.pipeType,
                                                       event.deviceTransfer,
                                                       {}})
                          .first;
                  matchingNodes = &pipeIt->second.sends;
                } else if (event.kind == PipeEventKind::ReceivePost) {
                  matchingNodes = &receivePostNodes[getPipeCoordIdentity(
                      event.pipeType, event.deviceTransfer, coord)];
                } else if (event.kind == PipeEventKind::ReceiveWait) {
                  matchingNodes = &receiveWaitNodes[getPipeCoordIdentity(
                      event.pipeType, event.deviceTransfer, coord)];
                } else {
                  matchingNodes = &receiveWaitAnyNodesByOperation[event.op];
                }
                if (failed(verifySingleKernelFunction(
                        nodes, *matchingNodes, event, function, state))) {
                  return WalkResult::interrupt();
                }
                PipeScheduleNodeId nodeId = addPipeScheduleNode(
                    nodes, event, location, function, activeCallSites,
                    activePipeNetRecords, executionCountDivisor);
                ++scheduleNodeCount;
                matchingNodes->push_back(nodeId);
                if (event.kind == PipeEventKind::ReceivePost) {
                  receivePostNodesByOperation[event.op].push_back(nodeId);
                } else if (event.kind == PipeEventKind::ReceiveWait) {
                  allReceiveWaitNodes.push_back(nodeId);
                } else if (event.kind == PipeEventKind::ReceiveWaitAny) {
                  allReceiveWaitAnyNodes.push_back(nodeId);
                }
                auto lastNode = lastNodeByLocation.find(location);
                if (lastNode != lastNodeByLocation.end()) {
                  addPipeScheduleEdge(nodes, lastNode->second, nodeId,
                                      PipeScheduleEdgeKind::ProgramOrder);
                }
                lastNodeByLocation[location] = nodeId;
                return WalkResult::advance();
              });
          if (walkResult.wasInterrupted()) {
            return;
          }
        }
      }
    }
  }

  llvm::DenseMap<PipeScheduleNodeId, PipeScheduleNodeId> completingSendByPost;
  llvm::DenseSet<PipeScheduleNodeId> postsWithInvalidCorrespondence;
  llvm::DenseMap<PipeScheduleNodeId, SmallVector<PipeScheduleNodeId>>
      waitsByPost;

  for (const auto &[pipeIdentity, occurrences] : pipeOccurrences) {
    LaunchNodeDomain destinations = getPipeDestinationLaunchNodeDomain(
        occurrences.pipeType, state.launchDomains.baseDomain);
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
        ArrayRef<PipeScheduleNodeId> posts = getReceiverNodes(
            receivePostNodes,
            getPipeCoordIdentity(occurrences.pipeType,
                                 occurrences.deviceTransfer, coord));
        if (haveInvalidPipeOccurrenceCount(posts, occurrences.sends,
                                           /*requireEqualOccurrences=*/true)) {
          postMismatchCoords.push_back(coord);
        }
      }
    }
    if (!postMismatchCoords.empty()) {
      ArrayRef<PipeScheduleNodeId> posts = getReceiverNodes(
          receivePostNodes,
          getPipeCoordIdentity(occurrences.pipeType, occurrences.deviceTransfer,
                               postMismatchCoords.front()));
      emitPipeOccurrenceCountError(nodes, posts, occurrences.sends,
                                   "receiver post", "send", postMismatchCoords,
                                   state);
      for (LaunchNodeCoord coord : destinations.nodes) {
        ArrayRef<PipeScheduleNodeId> unpairedPosts = getReceiverNodes(
            receivePostNodes,
            getPipeCoordIdentity(occurrences.pipeType,
                                 occurrences.deviceTransfer, coord));
        postsWithInvalidCorrespondence.insert(unpairedPosts.begin(),
                                              unpairedPosts.end());
      }
      continue;
    }

    for (LaunchNodeCoord coord : destinations.nodes) {
      PipeCoordIdentity identity = getPipeCoordIdentity(
          occurrences.pipeType, occurrences.deviceTransfer, coord);
      if (!occurrences.sends.empty()) {
        ArrayRef<PipeScheduleNodeId> posts =
            getReceiverNodes(receivePostNodes, identity);
        if (failed(addPipeOccurrenceEdges(
                nodes, posts, occurrences.sends,
                PipeScheduleEdgeKind::ReceivePostEnablesSend, "receiver post",
                "send", coord, state))) {
          postsWithInvalidCorrespondence.insert(posts.begin(), posts.end());
          break;
        }
        for (auto [post, send] : llvm::zip(posts, occurrences.sends)) {
          completingSendByPost[post] = send;
        }
      }
    }
  }

  // A receive wait observes the token produced by one exact post. Repeated
  // waits on that token all depend on the same completing send.
  for (PipeScheduleNodeId waitNodeId : allReceiveWaitNodes) {
    const PipeScheduleNode &waitNode = nodes[waitNodeId];
    auto postNodesIt = receivePostNodesByOperation.find(waitNode.receivePost);
    std::optional<PipeScheduleNodeId> maybePostNode;
    if (postNodesIt != receivePostNodesByOperation.end()) {
      maybePostNode =
          findReceivePostNodeForWait(nodes, waitNodeId, postNodesIt->second);
    }
    if (!maybePostNode) {
      waitNode.op->emitOpError()
          << "cannot associate this receive wait with its defining receiver "
             "post at core_x="
          << waitNode.location.node.x
          << ", core_y=" << waitNode.location.node.y;
      state.sawError = true;
      continue;
    }
    auto sendIt = completingSendByPost.find(*maybePostNode);
    if (sendIt == completingSendByPost.end()) {
      if (postsWithInvalidCorrespondence.contains(*maybePostNode)) {
        continue;
      }
      auto diag =
          waitNode.op->emitOpError()
          << "receive wait has no send corresponding to its defining "
             "receiver post on PipeNet "
          << state.launchDomains.netName(waitNode.pipeType.getPipeNetId())
          << " at core_x=" << waitNode.location.node.x
          << ", core_y=" << waitNode.location.node.y;
      diag.attachNote(nodes[*maybePostNode].op->getLoc())
          << "defining receiver post is here";
      state.sawError = true;
      continue;
    }
    addPipeScheduleEdge(nodes, sendIt->second, waitNodeId,
                        PipeScheduleEdgeKind::SendCompletesReceive);
    waitsByPost[*maybePostNode].push_back(waitNodeId);
  }

  // A wait-any becomes executable when one candidate's corresponding send has
  // completed. Completion alternatives remain disjunctive graph requirements.
  for (PipeScheduleNodeId waitNodeId : allReceiveWaitAnyNodes) {
    PipeScheduleNode &waitNode = nodes[waitNodeId];
    llvm::DenseSet<PipeScheduleNodeId> distinctSends;
    for (const PipeWaitAnyAlternative &alternative :
         waitNode.waitAnyAlternatives) {
      auto postNodesIt =
          receivePostNodesByOperation.find(alternative.receivePost);
      if (postNodesIt == receivePostNodesByOperation.end()) {
        continue;
      }
      std::optional<PipeScheduleNodeId> maybePostNode =
          findReceivePostNodeForWaitAnyAlternative(
              nodes, waitNodeId, alternative, postNodesIt->second);
      if (!maybePostNode) {
        continue;
      }
      auto sendIt = completingSendByPost.find(*maybePostNode);
      if (sendIt == completingSendByPost.end()) {
        continue;
      }
      if (distinctSends.insert(sendIt->second).second) {
        waitNode.waitAnyCompletingSends.push_back(sendIt->second);
      }
    }
    if (waitNode.waitAnyCompletingSends.empty()) {
      waitNode.op->emitOpError()
          << "receive wait-any has no candidate send corresponding to a "
             "defining receiver post at core_x="
          << waitNode.location.node.x
          << ", core_y=" << waitNode.location.node.y;
      state.sawError = true;
    }
  }

  PipeRendezvousLifetimeAnalysis lifetimeAnalysis(nodes, completingSendByPost,
                                                  waitsByPost, state);
  for (const auto &postNodes : llvm::make_second_range(receivePostNodes)) {
    if (failed(lifetimeAnalysis.verify(postNodes))) {
      state.sawError = true;
    }
  }

  SmallVector<bool> executable = computeExecutablePipeScheduleNodes(nodes);
  if (llvm::all_of(executable, [](bool value) { return value; })) {
    return;
  }
  if (std::optional<SmallVector<PipeScheduleNodeId>> maybeCycle =
          findPipeScheduleCycle(nodes)) {
    emitPipeScheduleCycleDiagnostic(nodes, *maybeCycle, state);
    return;
  }
  auto blockedWaitAny =
      llvm::find_if(allReceiveWaitAnyNodes, [&](PipeScheduleNodeId nodeId) {
        return !executable[nodeId];
      });
  if (blockedWaitAny != allReceiveWaitAnyNodes.end()) {
    const PipeScheduleNode &waitNode = nodes[*blockedWaitAny];
    auto diag = waitNode.op->emitOpError()
                << "receive wait-any can block with every candidate send "
                   "ordered after the selection at core_x="
                << waitNode.location.node.x
                << ", core_y=" << waitNode.location.node.y;
    std::size_t noteCount = 0;
    for (PipeScheduleNodeId sendNodeId : waitNode.waitAnyCompletingSends) {
      if (noteCount++ >= kMaxPipeScheduleDiagnosticNotes) {
        break;
      }
      diag.attachNote(nodes[sendNodeId].op->getLoc())
          << "this candidate send cannot complete before the wait-any";
    }
    state.sawError = true;
  }
}

// Reject predicates and scopes that reference no static or record-table
// declaration, since an empty role set would otherwise accept any nested work.
LogicalResult
validatePipeNetReferences(ModuleOp module,
                          const LaunchNodeDomainState &launchDomains) {
  LogicalResult result = success();
  module.walk([&](Operation *op) {
    auto report = [&](int64_t netId) {
      op->emitOpError() << "references unknown PipeNet "
                        << launchDomains.netName(netId) << " (id " << netId
                        << "); no pipe or PipeNet record table declares this "
                           "net";
      result = failure();
    };
    if (auto pred = mlir::dyn_cast<PipeNetPredicateOpInterface>(op)) {
      if (!launchDomains.pipeNetLocs.count(pred.getReferencedPipeNetId())) {
        report(pred.getReferencedPipeNetId());
      }
      return;
    }
    if (auto scopeOp = mlir::dyn_cast<PipeNetScopeOp>(op)) {
      SmallVector<int64_t> ids;
      if (readPipeNetScopeIds(scopeOp, ids)) {
        for (int64_t id : ids) {
          if (!launchDomains.pipeNetLocs.count(id)) {
            report(id);
          }
        }
      }
    }
  });
  return result;
}

/// Verify that one declared pipe relation belongs to the module launch grid.
LogicalResult
validatePipeRelationEndpoints(Operation *declaration, PipeType pipeType,
                              const LaunchNodeDomain &launchDomain) {
  LaunchNodeCoord source{pipeType.getSrcX(), pipeType.getSrcY()};
  if (!knownLaunchNodeDomainContains(launchDomain, source)) {
    return declaration->emitOpError()
           << "declares source core_x=" << source.x << ", core_y=" << source.y
           << " outside the module `ttl.launch_grid`";
  }
  LaunchNodeCoord destinationStart{pipeType.getDstStartX(),
                                   pipeType.getDstStartY()};
  LaunchNodeCoord destinationEnd{pipeType.getDstEndX(), pipeType.getDstEndY()};
  if (!knownLaunchNodeDomainContains(launchDomain, destinationStart) ||
      !knownLaunchNodeDomainContains(launchDomain, destinationEnd)) {
    return declaration->emitOpError()
           << "declares destination range core_x=" << pipeType.getDstStartX()
           << ".." << pipeType.getDstEndX()
           << ", core_y=" << pipeType.getDstStartY() << ".."
           << pipeType.getDstEndY() << " outside the module `ttl.launch_grid`";
  }
  return success();
}

/// Verify that every declared pipe endpoint belongs to the module launch grid.
LogicalResult
validatePipeEndpoints(ModuleOp module,
                      const LaunchNodeDomainState &launchDomains) {
  LogicalResult result = success();
  llvm::DenseSet<PipeNetRecordsAttr> validatedRecordTables;
  module.walk([&](Operation *op) {
    if (auto pipe = mlir::dyn_cast<CreatePipeOp>(op)) {
      PipeType pipeType = mlir::cast<PipeType>(pipe.getResult().getType());
      if (failed(validatePipeRelationEndpoints(op, pipeType,
                                               launchDomains.baseDomain))) {
        result = failure();
      }
      return;
    }

    PipeNetRecordsAttr records =
        llvm::TypeSwitch<Operation *, PipeNetRecordsAttr>(op)
            .Case<PipeNetForeachSrcOp, PipeNetForeachDstOp, SelectPipeSrcOp,
                  SelectPipeDstOp>(
                [](auto recordsOp) { return recordsOp.getRecords(); })
            .Default(PipeNetRecordsAttr());
    if (!records || !validatedRecordTables.insert(records).second) {
      return;
    }
    for (PipeRecordAttr record : records.getPipes()) {
      PipeType pipeType = PipeType::get(
          records.getContext(), record.getSrcX(), record.getSrcY(),
          record.getDstStartX(), record.getDstStartY(), record.getDstEndX(),
          record.getDstEndY(), records.getPipeNetId());
      if (failed(validatePipeRelationEndpoints(op, pipeType,
                                               launchDomains.baseDomain))) {
        result = failure();
      }
    }
  });
  return result;
}

/// Validate the module properties required by both PipeNet verifiers.
LogicalResult validatePipeNetModule(ModuleOp module,
                                    const LaunchNodeDomainState &launchDomains,
                                    StringRef passName) {
  if (failed(validatePipeNetReferences(module, launchDomains))) {
    return failure();
  }
  if (!launchDomains.hasPipes()) {
    return success();
  }
  if (!launchDomains.hasLaunchGrid) {
    module.emitError() << passName
                       << " requires a `ttl.launch_grid` module attribute "
                          "(an i64 array of length 2 with positive entries)";
    return failure();
  }
  return validatePipeEndpoints(module, launchDomains);
}

struct TTLVerifyPipeNetGuardsPass
    : impl::TTLVerifyPipeNetGuardsBase<TTLVerifyPipeNetGuardsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    const PipeNetLaunchNodeDomainAnalysis &launchNodeAnalysis =
        getAnalysis<PipeNetLaunchNodeDomainAnalysis>();
    const LaunchNodeDomainState &launchDomains = launchNodeAnalysis.getState();
    if (failed(validatePipeNetModule(module, launchDomains,
                                     "ttl-verify-pipenet-guards"))) {
      signalPassFailure();
      return;
    }
    if (!launchDomains.hasPipes()) {
      markAllAnalysesPreserved();
      return;
    }
    if (!launchNodeAnalysis.isValid()) {
      signalPassFailure();
      return;
    }

    const DFBLogicalIdentityAnalysis &dfbIdentities =
        getAnalysis<DFBLogicalIdentityAnalysis>();
    if (failed(verifyGuardDFBIdentities(module, dfbIdentities))) {
      signalPassFailure();
      return;
    }

    ValueOriginAnalysis &valueOrigins = getAnalysis<ValueOriginAnalysis>();
    FailureOr<std::unique_ptr<PipeTransferIndex>> maybeTransfers =
        PipeTransferIndex::create(module, valueOrigins);
    if (failed(maybeTransfers)) {
      signalPassFailure();
      return;
    }
    ModuleState state(**maybeTransfers, launchDomains, valueOrigins,
                      dfbIdentities);

    module.walk([&](Operation *op) {
      if (const PipeNetOperationDomainInfo *info =
              launchNodeAnalysis.getOperationInfo(op)) {
        recordGuardOperation(op, info->domain, info->unanalyzableOp, state);
      }
      if (auto scopeOp = mlir::dyn_cast<PipeNetScopeOp>(op)) {
        if (const PipeNetScopeDomainInfo *info =
                launchNodeAnalysis.getScopeInfo(scopeOp)) {
          verifyPipeNetScope(scopeOp, info->domain, info->unanalyzableOp,
                             info->scope, state);
        }
      }
    });

    if (!isDFBProtocolDomainVerificationRelaxed()) {
      verifyCBWaits(state);
    }
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

    const PipeNetLaunchNodeDomainAnalysis &launchNodeAnalysis =
        getAnalysis<PipeNetLaunchNodeDomainAnalysis>();
    const LaunchNodeDomainState &launchDomains = launchNodeAnalysis.getState();
    if (failed(validatePipeNetModule(module, launchDomains,
                                     "ttl-verify-pipenet-schedule"))) {
      signalPassFailure();
      return;
    }
    if (!launchDomains.hasPipes()) {
      markAllAnalysesPreserved();
      return;
    }
    if (!launchNodeAnalysis.isValid()) {
      signalPassFailure();
      return;
    }

    ValueOriginAnalysis &valueOrigins = getAnalysis<ValueOriginAnalysis>();
    FailureOr<std::unique_ptr<PipeTransferIndex>> maybeTransfers =
        PipeTransferIndex::create(module, valueOrigins);
    if (failed(maybeTransfers)) {
      signalPassFailure();
      return;
    }
    ModuleState state(**maybeTransfers, launchDomains, valueOrigins);

    module.walk([&](Operation *op) {
      if (const PipeNetOperationDomainInfo *info =
              launchNodeAnalysis.getOperationInfo(op)) {
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
