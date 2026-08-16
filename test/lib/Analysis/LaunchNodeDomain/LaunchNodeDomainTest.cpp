// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This test driver validates domain algebra and prints the launch-node lattice
// immediately before every operation with a test.label attribute.

#include "ttlang/Dialect/TTCore/IR/TTCore.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/Transforms/LaunchNodeDomainAnalysis.h"

#include "mlir/Analysis/DataFlow/Utils.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"

namespace {

constexpr llvm::StringLiteral kLabelAttrName = "test.label";

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

} // namespace

int main(int argumentCount, char **argumentValues) {
  if (argumentCount != 2) {
    llvm::errs() << "usage: ttlang-launch-node-domain-test <input.mlir>\n";
    return 1;
  }
  if (!verifyLaunchNodeDomainAlgebra()) {
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
  module->walk([&](mlir::Operation *operation) {
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
  return succeeded ? 0 : 1;
}
