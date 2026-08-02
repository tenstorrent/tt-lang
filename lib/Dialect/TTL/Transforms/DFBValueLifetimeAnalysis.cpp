// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBValueLifetimeAnalysis.h"

#include "DFBAcquireReleaseAnalysis.h"

#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::tt::ttl {

namespace {

/// State of one static DFB storage identity at a program point.
enum class DFBStorageState {
  /// Every represented execution has released storage for the identity.
  Unavailable,

  /// Every reachable execution retains storage for the identity.
  Available,

  /// At least one represented execution may have released the storage.
  MayBeUnavailable,
};

static DFBStorageState joinStorageState(DFBStorageState lhs,
                                        DFBStorageState rhs) {
  return lhs == rhs ? lhs : DFBStorageState::MayBeUnavailable;
}

/// Product lattice over the finite set of DFB storage identities in a kernel.
///
/// The uninitialized lattice is unreachable. Every reachable entry state
/// contains exact acquisition identities as `Unavailable` and standalone
/// associations as `Available`. Acquisitions establish `Available`, and
/// releases with proven FIFO owners establish `Unavailable`. A release with
/// unresolved ownership, or one that may invalidate a standalone association,
/// establishes `MayBeUnavailable`. Joining different reachable states also
/// yields `MayBeUnavailable`, so availability is reported only when every
/// reachable predecessor agrees.
class DFBValueAvailabilityLattice : public dataflow::AbstractDenseLattice {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DFBValueAvailabilityLattice)

  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult join(const dataflow::AbstractDenseLattice &rhs) override {
    const auto &other = static_cast<const DFBValueAvailabilityLattice &>(rhs);
    return join(other.initialized, other.storageStates);
  }

  ChangeResult
  join(bool otherInitialized,
       const DenseMap<Operation *, DFBStorageState> &otherStorageStates) {
    ChangeResult changed = ChangeResult::NoChange;
    if (otherInitialized && !initialized) {
      initialized = true;
      changed |= ChangeResult::Change;
    }
    for (auto [identity, otherState] : otherStorageStates) {
      auto [iterator, inserted] =
          storageStates.try_emplace(identity, otherState);
      if (inserted) {
        changed |= ChangeResult::Change;
        continue;
      }
      DFBStorageState joined = joinStorageState(iterator->second, otherState);
      if (joined != iterator->second) {
        iterator->second = joined;
        changed |= ChangeResult::Change;
      }
    }
    return changed;
  }

  ChangeResult initialize(ArrayRef<Operation *> identities,
                          ArrayRef<Operation *> initiallyAvailable) {
    ChangeResult changed = ChangeResult::NoChange;
    if (!initialized) {
      initialized = true;
      changed |= ChangeResult::Change;
    }
    for (Operation *identity : identities) {
      if (storageStates.try_emplace(identity, DFBStorageState::Unavailable)
              .second) {
        changed |= ChangeResult::Change;
      }
    }
    for (Operation *identity : initiallyAvailable) {
      auto state = storageStates.find(identity);
      assert(state != storageStates.end() &&
             "initially available identity must be indexed");
      if (state->second != DFBStorageState::Available) {
        state->second = DFBStorageState::Available;
        changed |= ChangeResult::Change;
      }
    }
    return changed;
  }

  bool isInitialized() const { return initialized; }

  const DenseMap<Operation *, DFBStorageState> &getStorageStates() const {
    return storageStates;
  }

  std::optional<DFBStorageState> getStorageState(Operation *identity) const {
    auto state = storageStates.find(identity);
    if (state == storageStates.end()) {
      return std::nullopt;
    }
    return state->second;
  }

  void print(raw_ostream &output) const override {
    output << (initialized ? "initialized" : "unreachable");
    for (auto [identity, state] : storageStates) {
      output << "\n  " << identity->getName() << ": ";
      switch (state) {
      case DFBStorageState::Unavailable:
        output << "unavailable";
        break;
      case DFBStorageState::Available:
        output << "available";
        break;
      case DFBStorageState::MayBeUnavailable:
        output << "may-be-unavailable";
        break;
      }
    }
  }

private:
  bool initialized = false;
  DenseMap<Operation *, DFBStorageState> storageStates;
};

static std::optional<AttachCBOp> findAssociation(Value value) {
  value = traceUnrealizedCasts(value);
  if (auto slice = value.getDefiningOp<tensor::ExtractSliceOp>()) {
    return findAssociation(slice.getSource());
  }
  if (auto extract = value.getDefiningOp<tensor::ExtractOp>()) {
    return findAssociation(extract.getTensor());
  }
  if (auto association = value.getDefiningOp<AttachCBOp>()) {
    return association;
  }
  return std::nullopt;
}

/// Static acquisition identities and conservative DFB associations used by
/// the dense transfer function.
///
/// An acquire operation identifies one producer or consumer pointer interval.
/// An `ttl.attach_cb` not derived from an acquire identifies only an
/// association, so every release with the same SSA DFB value invalidates it.
class DFBValueIdentityIndex {
public:
  static PlanningResult<std::unique_ptr<DFBValueIdentityIndex>>
  create(func::FuncOp kernel) {
    PlanningResult<std::unique_ptr<DFBAcquireReleaseIndex>> releaseOwners =
        DFBAcquireReleaseIndex::create(kernel);
    if (releaseOwners.isInvalidIR()) {
      const PlanningDiagnostic &diagnostic = releaseOwners.getInvalidIR();
      return PlanningResult<std::unique_ptr<DFBValueIdentityIndex>>::invalidIR(
          diagnostic.operation, diagnostic.message);
    }
    assert(releaseOwners.isPlanned() &&
           "lifecycle indexing has no recoverable rejection");
    return PlanningResult<std::unique_ptr<DFBValueIdentityIndex>>::planned(
        std::unique_ptr<DFBValueIdentityIndex>(new DFBValueIdentityIndex(
            kernel, std::move(releaseOwners).takePlan())));
  }

  const DFBReleaseOwnership &getReleaseOwnership(Operation *release) const {
    return releaseOwners->getReleaseOwnership(release);
  }

  const DFBAcquireReleaseIndex &getAcquireReleaseIndex() const {
    return *releaseOwners;
  }

  ArrayRef<Operation *> getIdentities(Value dfb) const {
    auto identities = identitiesByDFB.find(dfb);
    if (identities == identitiesByDFB.end()) {
      return {};
    }
    return identities->second;
  }

  ArrayRef<Operation *> getAllIdentities() const { return allIdentities; }

  ArrayRef<Operation *> getAssociatedIdentities() const {
    return associatedIdentityOrder;
  }

  bool isAssociated(Operation *identity) const {
    return associatedIdentities.contains(identity);
  }

  std::optional<Operation *> getIdentity(Value value) const {
    if (!getAttachedCB(value)) {
      return std::nullopt;
    }
    if (Operation *acquire = findCBAcquireOp(value)) {
      return acquire;
    }
    std::optional<AttachCBOp> association = findAssociation(value);
    if (!association) {
      return std::nullopt;
    }
    return association->getOperation();
  }

private:
  DFBValueIdentityIndex(func::FuncOp kernel,
                        std::unique_ptr<DFBAcquireReleaseIndex> releaseOwners)
      : releaseOwners(std::move(releaseOwners)) {
    kernel.walk([&](Operation *operation) {
      if (isDFBAcquireOp(operation)) {
        Value dfb = getDFBAcquireDFB(operation);
        identitiesByDFB[dfb].push_back(operation);
        allIdentities.push_back(operation);
        return;
      }
      auto association = dyn_cast<AttachCBOp>(operation);
      if (!association || findCBAcquireOp(association.getResult())) {
        return;
      }
      Value dfb = association.getCb();
      identitiesByDFB[dfb].push_back(operation);
      allIdentities.push_back(operation);
      associatedIdentityOrder.push_back(operation);
      associatedIdentities.insert(operation);
    });
  }

  std::unique_ptr<DFBAcquireReleaseIndex> releaseOwners;
  SmallVector<Operation *> allIdentities;
  SmallVector<Operation *> associatedIdentityOrder;
  DenseMap<Value, SmallVector<Operation *>> identitiesByDFB;
  llvm::DenseSet<Operation *> associatedIdentities;
};

/// Applies acquisitions and releases to the product lattice along executable
/// region and CFG edges discovered by MLIR's baseline dataflow analyses.
class DFBValueAvailabilityDataFlow
    : public dataflow::DenseForwardDataFlowAnalysis<
          DFBValueAvailabilityLattice> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DFBValueAvailabilityDataFlow)

  DFBValueAvailabilityDataFlow(DataFlowSolver &solver, func::FuncOp kernel,
                               const DFBValueIdentityIndex &identities)
      : DenseForwardDataFlowAnalysis(solver), kernel(kernel),
        identities(identities) {}

  void setToEntryState(DFBValueAvailabilityLattice *lattice) override {
    ProgramPoint *point = lattice->getAnchor().dyn_cast<ProgramPoint *>();
    bool isKernelEntry = point &&
                         point->getBlock() == &kernel.getBody().front() &&
                         point->isBlockStart();
    // Unknown region control flow also uses this hook. Only kernel entry is
    // known to begin with externally provided DFB storage; other boundaries
    // start unavailable so an unmodeled region cannot restore released storage.
    ArrayRef<Operation *> initiallyAvailable =
        isKernelEntry ? identities.getAssociatedIdentities()
                      : ArrayRef<Operation *>();
    propagateIfChanged(
        lattice,
        lattice->initialize(identities.getAllIdentities(), initiallyAvailable));
  }

  LogicalResult visitOperation(Operation *operation,
                               const DFBValueAvailabilityLattice &before,
                               DFBValueAvailabilityLattice *after) override {
    bool transferredInitialized = before.isInitialized();
    DenseMap<Operation *, DFBStorageState> transferred(
        before.getStorageStates());

    if (isDFBAcquireOp(operation)) {
      transferred[operation] = DFBStorageState::Available;
    } else if (isDFBReleaseOp(operation)) {
      Value dfb = getDFBReleaseDFB(operation);
      const DFBReleaseOwnership &ownership =
          identities.getReleaseOwnership(operation);
      for (Operation *identity : identities.getIdentities(dfb)) {
        bool isRecordedOwner =
            llvm::is_contained(ownership.candidateOwners, identity);
        if (isRecordedOwner &&
            ownership.ownership != DFBReleaseOwnershipKind::Unresolved) {
          transferred[identity] = DFBStorageState::Unavailable;
        } else if (identities.isAssociated(identity) || isRecordedOwner) {
          transferred[identity] = DFBStorageState::MayBeUnavailable;
        }
      }
    }

    propagateIfChanged(after, after->join(transferredInitialized, transferred));
    return success();
  }

private:
  func::FuncOp kernel;
  const DFBValueIdentityIndex &identities;
};

} // namespace

/// Owns the immutable identity index and solved dense dataflow states.
class DFBValueLifetimeAnalysis::Impl {
public:
  Impl(func::FuncOp kernel, std::unique_ptr<DFBValueIdentityIndex> identities)
      : kernel(kernel), identities(std::move(identities)) {
    dataflow::loadBaselineAnalyses(solver);
    solver.load<DFBValueAvailabilityDataFlow>(kernel, *this->identities);
  }

  LogicalResult run() { return solver.initializeAndRun(kernel); }

  DFBValueAvailability getAvailability(Value value, Operation *consumer) const {
    assert(consumer->getParentOfType<func::FuncOp>() == kernel &&
           "availability consumer must belong to the analyzed kernel");
    if (!getAttachedCB(value)) {
      return DFBValueAvailability::NotDFBBacked;
    }

    std::optional<Operation *> identity = identities->getIdentity(value);
    if (!identity) {
      return DFBValueAvailability::MayBeReleased;
    }

    ProgramPoint *point = solver.getProgramPointBefore(consumer);
    const auto *lattice =
        solver.lookupState<DFBValueAvailabilityLattice>(point);
    ProgramPoint *blockStart =
        solver.getProgramPointBefore(consumer->getBlock());
    const auto *executable =
        solver.lookupState<dataflow::Executable>(blockStart);
    // Dead code analysis records reachability on blocks. Dense analysis does
    // not create operation lattices inside a dead block, so absence of the
    // value lattice alone cannot distinguish unreachable code from an
    // unavailable analysis fact.
    if (executable && !executable->isLive()) {
      return DFBValueAvailability::DefinitelyAvailable;
    }
    if (!lattice || !lattice->isInitialized()) {
      return DFBValueAvailability::MayBeReleased;
    }

    std::optional<DFBStorageState> state = lattice->getStorageState(*identity);
    if (state && *state == DFBStorageState::Available) {
      return DFBValueAvailability::DefinitelyAvailable;
    }
    return DFBValueAvailability::MayBeReleased;
  }

  const DFBAcquireReleaseIndex &getAcquireReleaseIndex() const {
    return identities->getAcquireReleaseIndex();
  }

private:
  func::FuncOp kernel;
  std::unique_ptr<DFBValueIdentityIndex> identities;
  mutable DataFlowSolver solver;
};

PlanningResult<std::unique_ptr<DFBValueLifetimeAnalysis>>
DFBValueLifetimeAnalysis::create(func::FuncOp kernel) {
  PlanningResult<std::unique_ptr<DFBValueIdentityIndex>> identities =
      DFBValueIdentityIndex::create(kernel);
  if (identities.isInvalidIR()) {
    const PlanningDiagnostic &diagnostic = identities.getInvalidIR();
    return PlanningResult<std::unique_ptr<DFBValueLifetimeAnalysis>>::invalidIR(
        diagnostic.operation, diagnostic.message);
  }
  assert(identities.isPlanned() &&
         "DFB identity indexing has no recoverable rejection");
  auto impl = std::make_unique<Impl>(kernel, std::move(identities).takePlan());
  if (failed(impl->run())) {
    return PlanningResult<std::unique_ptr<DFBValueLifetimeAnalysis>>::invalidIR(
        kernel, "failed to solve dataflow buffer value lifetimes");
  }
  return PlanningResult<std::unique_ptr<DFBValueLifetimeAnalysis>>::planned(
      std::unique_ptr<DFBValueLifetimeAnalysis>(
          new DFBValueLifetimeAnalysis(std::move(impl))));
}

DFBValueLifetimeAnalysis::DFBValueLifetimeAnalysis(std::unique_ptr<Impl> impl)
    : impl(std::move(impl)) {}

DFBValueLifetimeAnalysis::~DFBValueLifetimeAnalysis() = default;

DFBValueAvailability
DFBValueLifetimeAnalysis::getAvailability(Value value,
                                          Operation *consumer) const {
  return impl->getAvailability(value, consumer);
}

bool DFBValueLifetimeAnalysis::anyValueMayBeReleased(
    ValueRange values, Operation *consumer) const {
  return llvm::any_of(values, [&](Value value) {
    return getAvailability(value, consumer) ==
           DFBValueAvailability::MayBeReleased;
  });
}

const DFBAcquireReleaseIndex &
DFBValueLifetimeAnalysis::getAcquireReleaseIndex() const {
  return impl->getAcquireReleaseIndex();
}

} // namespace mlir::tt::ttl
