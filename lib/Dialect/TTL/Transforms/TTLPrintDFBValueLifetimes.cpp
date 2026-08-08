// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBValueLifetimeAnalysis.h"

#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLPRINTDFBVALUELIFETIMES
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

static StringRef stringifyKind(DFBAcquireReleaseKind kind) {
  switch (kind) {
  case DFBAcquireReleaseKind::Producer:
    return "producer";
  case DFBAcquireReleaseKind::Consumer:
    return "consumer";
  }
  llvm_unreachable("unknown DFB acquisition kind");
}

static StringRef stringifyAvailability(DFBValueAvailability availability) {
  switch (availability) {
  case DFBValueAvailability::NotDFBBacked:
    return "not-dfb-backed";
  case DFBValueAvailability::DefinitelyAvailable:
    return "available";
  case DFBValueAvailability::MayBeReleased:
    return "may-be-released";
  }
  llvm_unreachable("unknown DFB value availability");
}

static void
printReleaseOwners(raw_ostream &output, const DFBReleaseOwnership &ownership,
                   const DenseMap<Operation *, unsigned> &acquisitionIds) {
  switch (ownership.ownership) {
  case DFBReleaseOwnershipKind::Exact:
    output << "exact A" << acquisitionIds.at(ownership.getExactOwner());
    return;
  case DFBReleaseOwnershipKind::Multiple:
    output << "multiple [";
    break;
  case DFBReleaseOwnershipKind::Unresolved:
    output << "unresolved [";
    break;
  }

  llvm::interleaveComma(ownership.candidateOwners, output,
                        [&](Operation *candidate) {
                          output << "A" << acquisitionIds.at(candidate);
                        });
  output << "]";
}

struct TTLPrintDFBValueLifetimesPass
    : public impl::TTLPrintDFBValueLifetimesBase<
          TTLPrintDFBValueLifetimesPass> {
  using TTLPrintDFBValueLifetimesBase::TTLPrintDFBValueLifetimesBase;

  void runOnOperation() override {
    func::FuncOp kernel = getOperation();
    if (kernel.isExternal()) {
      return;
    }

    PlanningResult<std::unique_ptr<DFBValueLifetimeAnalysis>> plannedLifetimes =
        DFBValueLifetimeAnalysis::create(kernel);
    if (plannedLifetimes.isInvalidIR()) {
      const PlanningDiagnostic &diagnostic = plannedLifetimes.getInvalidIR();
      diagnostic.operation->emitOpError(diagnostic.message);
      signalPassFailure();
      return;
    }
    assert(plannedLifetimes.isPlanned() &&
           "lifetime analysis has no recoverable rejection");
    std::unique_ptr<DFBValueLifetimeAnalysis> lifetimes =
        std::move(plannedLifetimes).takePlan();

    const DFBAcquireReleaseIndex &lifecycles =
        lifetimes->getAcquireReleaseIndex();
    DenseMap<Operation *, unsigned> acquisitionIds;
    for (auto [acquisitionId, acquire] :
         llvm::enumerate(lifecycles.getAcquisitions())) {
      acquisitionIds.try_emplace(acquire, acquisitionId);
    }

    raw_ostream &output = llvm::errs();
    output << "DFB value lifetimes @" << kernel.getSymName() << "\n";
    for (Operation *acquire : lifecycles.getAcquisitions()) {
      const DFBTransactionRecord &transaction =
          lifecycles.getTransaction(acquire);
      output << "  A" << acquisitionIds.at(acquire) << " "
             << stringifyKind(transaction.kind)
             << " tiles=" << transaction.numTiles << "\n";
    }
    for (auto [releaseId, release] :
         llvm::enumerate(lifecycles.getReleases())) {
      const DFBReleaseOwnership &ownership =
          lifecycles.getReleaseOwnership(release);
      output << "  R" << releaseId << " " << stringifyKind(ownership.kind)
             << " tiles=" << ownership.numTiles << " owner=";
      printReleaseOwners(output, ownership, acquisitionIds);
      output << "\n";
    }

    SmallVector<std::pair<Value, std::string>> probes;
    for (Operation *acquire : lifecycles.getAcquisitions()) {
      assert(acquire->getNumResults() == 1 &&
             "DFB acquisitions must have one result");
      probes.emplace_back(acquire->getResult(0),
                          "A" + Twine(acquisitionIds.at(acquire)).str());
    }
    unsigned associationId = 0;
    kernel.walk([&](AttachCBOp association) {
      if (!findCBAcquireOp(association.getResult())) {
        probes.emplace_back(association.getResult(),
                            "S" + Twine(associationId++).str());
      }
    });

    DominanceInfo dominance(kernel);
    unsigned operationId = 0;
    kernel.walk([&](Operation *operation) {
      if (operation == kernel.getOperation()) {
        return;
      }
      unsigned currentOperationId = operationId++;
      for (const auto &[value, identity] : probes) {
        Operation *definition = value.getDefiningOp();
        if (!definition || definition == operation ||
            !dominance.properlyDominates(definition, operation)) {
          continue;
        }
        output << "  O" << currentOperationId << " " << operation->getName()
               << " " << identity << "="
               << stringifyAvailability(
                      lifetimes->getAvailability(value, operation))
               << "\n";
      }
    });
  }
};

} // namespace

} // namespace mlir::tt::ttl
