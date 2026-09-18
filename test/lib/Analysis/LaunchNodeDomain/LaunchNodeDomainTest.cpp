// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This test driver validates domain algebra and prints launch-node lattices and
// conditional-execution equivalence results requested by test attributes.

#include "ttlang/Dialect/TTCore/IR/TTCore.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"
#include "ttlang/Dialect/TTL/Transforms/PipeNetExecutionUtils.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/raw_ostream.h"

namespace {

constexpr llvm::StringLiteral kLabelAttrName = "test.label";
constexpr llvm::StringLiteral kConditionalPairAttrName =
    "test.conditional_pair";

bool verifyLaunchNodeDomainAlgebra() {
  using mlir::tt::ttl::getFullLaunchNodeDomain;
  using mlir::tt::ttl::getSingleLaunchNodeDomain;
  using mlir::tt::ttl::LaunchNodeDomain;
  using mlir::tt::ttl::launchNodeDomainsOverlap;

  LaunchNodeDomain leftColumn = getSingleLaunchNodeDomain({0, 0}).unionWith(
      getSingleLaunchNodeDomain({0, 1}));
  LaunchNodeDomain rightColumn = getSingleLaunchNodeDomain({1, 0}).unionWith(
      getSingleLaunchNodeDomain({1, 1}));
  LaunchNodeDomain fullDomain = getFullLaunchNodeDomain(2, 2);
  LaunchNodeDomain boundedLeft = LaunchNodeDomain::unknownWithin(leftColumn);
  LaunchNodeDomain boundedRight = LaunchNodeDomain::unknownWithin(rightColumn);
  LaunchNodeDomain unbounded = LaunchNodeDomain::unknown();

  bool valid = !boundedLeft.known &&
               boundedLeft.isUpperBoundSubsetOf(fullDomain) &&
               !boundedLeft.isUpperBoundSubsetOf(rightColumn) &&
               boundedLeft.unionWith(boundedRight) ==
                   LaunchNodeDomain::unknownWithin(fullDomain) &&
               leftColumn.unionWith(boundedLeft) == leftColumn &&
               boundedLeft.unionWith(leftColumn) == leftColumn &&
               fullDomain.unionWith(boundedLeft) == fullDomain &&
               boundedLeft.unionWith(fullDomain) == fullDomain &&
               boundedLeft.intersectWith(rightColumn) == LaunchNodeDomain{} &&
               boundedLeft.unionWith(boundedRight).subtract(rightColumn) ==
                   LaunchNodeDomain::unknownWithin(leftColumn) &&
               unbounded.intersectWith(leftColumn) == boundedLeft &&
               unbounded.unionWith(leftColumn) == unbounded &&
               unbounded.subtract(leftColumn) == unbounded &&
               leftColumn.subtract(unbounded) == boundedLeft &&
               launchNodeDomainsOverlap(boundedLeft, leftColumn) &&
               !launchNodeDomainsOverlap(boundedLeft, rightColumn) &&
               launchNodeDomainsOverlap(unbounded, rightColumn);
  if (!valid) {
    llvm::errs() << "launch-node domain algebra validation failed\n";
  }
  return valid;
}

bool verifyPipeNetRecordInductionValues() {
  using mlir::tt::ttl::getPipeNetRecordLoopInductionValue;
  using mlir::tt::ttl::LaunchExecutionLocation;
  using mlir::tt::ttl::PipeNetRecordLoop;
  using mlir::tt::ttl::PipeNetRecordSelection;

  PipeNetRecordLoop directLoop{
      mlir::tt::ttl::PipeNetRecordsAttr(), PipeNetRecordSelection::Source, {}};
  LaunchExecutionLocation leftNode({0, 0});
  LaunchExecutionLocation rightNode({1, 0});
  std::optional<std::uint64_t> directInduction =
      getPipeNetRecordLoopInductionValue(directLoop, leftNode, 4);

  PipeNetRecordLoop indirectLoop{mlir::tt::ttl::PipeNetRecordsAttr(),
                                 PipeNetRecordSelection::Destination,
                                 {}};
  indirectLoop.indirectInductionValues.try_emplace({leftNode, 4}, 1);
  indirectLoop.indirectInductionValues.try_emplace({rightNode, 4}, 3);
  std::optional<std::uint64_t> leftInduction =
      getPipeNetRecordLoopInductionValue(indirectLoop, leftNode, 4);
  std::optional<std::uint64_t> rightInduction =
      getPipeNetRecordLoopInductionValue(indirectLoop, rightNode, 4);
  std::optional<std::uint64_t> absentInduction =
      getPipeNetRecordLoopInductionValue(indirectLoop, leftNode, 2);

  bool valid = directInduction == 4 && leftInduction == 1 &&
               rightInduction == 3 && !absentInduction;
  if (!valid) {
    llvm::errs() << "PipeNet record induction validation failed\n";
  }
  return valid;
}

} // namespace

int main(int argumentCount, char **argumentValues) {
  if (argumentCount != 2) {
    llvm::errs() << "usage: ttlang-launch-node-domain-test <input.mlir>\n";
    return 1;
  }
  if (!verifyLaunchNodeDomainAlgebra()) {
    return 1;
  }
  if (!verifyPipeNetRecordInductionValues()) {
    return 1;
  }

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry
      .insert<mlir::tt::ttcore::TTCoreDialect,
              mlir::tt::ttkernel::TTKernelDialect, mlir::tt::ttl::TTLDialect>();
  mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects();
  mlir::ParserConfig parserConfig(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(argumentValues[1], parserConfig);
  if (!module) {
    return 1;
  }

  mlir::tt::ttl::LaunchNodeDomainState state;
  state.initialize(*module);
  if (!state.hasLaunchGrid) {
    module->emitError("test requires a valid ttl.launch_grid attribute");
    return 1;
  }

  mlir::DataFlowSolver solver;
  mlir::dataflow::loadBaselineAnalyses(solver);
  solver.load<mlir::tt::ttl::LaunchNodeDomainAnalysis>(state);
  if (mlir::failed(solver.initializeAndRun(*module))) {
    return 1;
  }

  bool succeeded = true;
  llvm::MapVector<llvm::StringRef, llvm::SmallVector<mlir::Operation *, 2>>
      conditionalPairs;
  module->walk([&](mlir::Operation *operation) {
    if (auto pair = operation->getAttrOfType<mlir::StringAttr>(
            kConditionalPairAttrName)) {
      conditionalPairs[pair.getValue()].push_back(operation);
    }
    auto label = operation->getAttrOfType<mlir::StringAttr>(kLabelAttrName);
    if (!label) {
      return;
    }
    const auto *lattice =
        solver.lookupState<mlir::tt::ttl::LaunchNodeDomainLattice>(
            solver.getProgramPointBefore(operation));
    if (!lattice) {
      operation->emitError("launch-node lattice is unavailable");
      succeeded = false;
      return;
    }
    llvm::outs() << label.getValue() << " = ";
    lattice->print(llvm::outs());
    llvm::outs() << "\n";
  });
  for (const auto &[pairName, operations] : conditionalPairs) {
    if (operations.size() != 2) {
      module->emitError() << "conditional pair '" << pairName
                          << "' requires exactly two operations";
      succeeded = false;
      continue;
    }
    bool equivalent =
        mlir::tt::ttl::proveEquivalentConditionalExecutionAtLaunchNodes(
            operations[0], {0, 0}, operations[1], {0, 0}, state);
    llvm::outs() << pairName << " = "
                 << (equivalent ? "equivalent" : "not-equivalent") << "\n";
  }
  return succeeded ? 0 : 1;
}
