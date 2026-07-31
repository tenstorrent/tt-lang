// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This test driver prints the launch-node lattice immediately before every
// operation with a test.label attribute.

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

} // namespace

int main(int argumentCount, char **argumentValues) {
  if (argumentCount != 2) {
    llvm::errs() << "usage: ttlang-launch-node-domain-test <input.mlir>\n";
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
