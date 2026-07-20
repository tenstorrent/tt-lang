// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Analysis/ExecutionCountAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>

namespace {

constexpr llvm::StringLiteral kAnalysisRootAttrName = "test.analysis_root";
constexpr llvm::StringLiteral kExpectedCountAttrName = "test.expected_count";
constexpr llvm::StringLiteral kLabelAttrName = "test.label";
constexpr llvm::StringLiteral kMaxIterationsAttrName =
    "test.max_enumerated_iterations";
constexpr llvm::StringLiteral kRegionInvocationCountAttrName =
    "test.region_invocation_count";
constexpr llvm::StringLiteral kValueAttrName = "test.value";

std::optional<llvm::APInt> evaluateFunctionArgument(mlir::func::FuncOp function,
                                                    mlir::Value value) {
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!argument || argument.getOwner() != &function.getBody().front()) {
    return std::nullopt;
  }
  auto maybeValueAttr = function.getArgAttrOfType<mlir::IntegerAttr>(
      argument.getArgNumber(), kValueAttrName);
  return maybeValueAttr ? std::optional(maybeValueAttr.getValue())
                        : std::nullopt;
}

bool verifyExpectedCount(mlir::Operation *operation,
                         std::optional<std::uint64_t> maybeActualCount) {
  mlir::Attribute expectedAttr = operation->getAttr(kExpectedCountAttrName);
  auto labelAttr = operation->getAttrOfType<mlir::StringAttr>(kLabelAttrName);
  if (!expectedAttr || !labelAttr) {
    operation->emitError() << "test targets require both "
                           << kExpectedCountAttrName << " and "
                           << kLabelAttrName;
    return false;
  }

  auto expectedInteger = mlir::dyn_cast<mlir::IntegerAttr>(expectedAttr);
  auto expectedString = mlir::dyn_cast<mlir::StringAttr>(expectedAttr);
  if ((!expectedInteger && !expectedString) ||
      (expectedString && expectedString.getValue() != "unknown")) {
    operation->emitError() << kExpectedCountAttrName
                           << " must be an integer or the string \"unknown\"";
    return false;
  }

  llvm::outs() << labelAttr.getValue() << " = ";
  if (maybeActualCount) {
    llvm::outs() << *maybeActualCount << "\n";
  } else {
    llvm::outs() << "unknown\n";
  }

  if (expectedInteger) {
    if (maybeActualCount && expectedInteger.getValue().getActiveBits() <= 64 &&
        *maybeActualCount == expectedInteger.getValue().getZExtValue()) {
      return true;
    }
  } else if (!maybeActualCount) {
    return true;
  }

  operation->emitError() << "execution count does not match "
                         << kExpectedCountAttrName;
  return false;
}

std::optional<std::uint64_t>
evaluateRegionInvocationCount(mlir::Region &region) {
  mlir::Operation *parent = region.getParentOp();
  // The parent attribute cannot assign different counts to multiple regions.
  if (parent->getNumRegions() != 1) {
    return std::nullopt;
  }
  auto maybeCountAttr =
      parent->getAttrOfType<mlir::IntegerAttr>(kRegionInvocationCountAttrName);
  if (!maybeCountAttr || maybeCountAttr.getValue().getActiveBits() > 64) {
    return std::nullopt;
  }
  return maybeCountAttr.getValue().getZExtValue();
}

bool verifyTargets(mlir::func::FuncOp function, mlir::Operation *targetScope) {
  mlir::tt::ExecutionCountAnalysis::Options options;
  if (auto maybeLimit =
          function->getAttrOfType<mlir::IntegerAttr>(kMaxIterationsAttrName)) {
    if (maybeLimit.getValue().getActiveBits() > 64) {
      function.emitError() << kMaxIterationsAttrName << " does not fit uint64";
      return false;
    }
    options.maxEnumeratedIterations = maybeLimit.getValue().getZExtValue();
  }

  mlir::tt::ExecutionCountAnalysis analysis(
      function.getBody(),
      [&](mlir::Value value) {
        return evaluateFunctionArgument(function, value);
      },
      evaluateRegionInvocationCount, options);

  bool succeeded = true;
  targetScope->walk([&](mlir::Operation *operation) {
    if (!operation->hasAttr(kExpectedCountAttrName)) {
      return;
    }
    succeeded &=
        verifyExpectedCount(operation, analysis.getExecutionCount(operation));
  });
  return succeeded;
}

} // namespace

int main(int argumentCount, char **argumentValues) {
  if (argumentCount != 2) {
    llvm::errs() << "usage: ttlang-execution-count-test <input.mlir>\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                  mlir::func::FuncDialect, mlir::scf::SCFDialect>();
  mlir::MLIRContext context(registry);
  mlir::ParserConfig parserConfig(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(argumentValues[1], parserConfig);
  if (!module) {
    return 1;
  }

  std::string originalIR;
  {
    llvm::raw_string_ostream output(originalIR);
    module->print(output);
  }

  bool succeeded = true;
  llvm::SmallVector<mlir::func::FuncOp> analysisRoots;
  module->walk([&](mlir::func::FuncOp function) {
    if (function->hasAttr(kAnalysisRootAttrName)) {
      analysisRoots.push_back(function);
    }
  });
  if (analysisRoots.size() > 1) {
    module->emitError() << "expected at most one " << kAnalysisRootAttrName;
    return 1;
  } else if (analysisRoots.size() == 1) {
    succeeded &= verifyTargets(analysisRoots.front(), module.get());
  } else {
    module->walk([&](mlir::func::FuncOp function) {
      succeeded &= verifyTargets(function, function);
    });
  }

  // The public analysis contract excludes mutations, including those performed
  // by operation fold hooks.
  std::string analyzedIR;
  {
    llvm::raw_string_ostream output(analyzedIR);
    module->print(output);
  }
  if (originalIR != analyzedIR) {
    module->emitError("execution-count analysis modified the input IR");
    return 1;
  }
  return succeeded ? 0 : 1;
}
