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
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>

namespace {

constexpr llvm::StringLiteral kExpectedCountAttrName = "test.expected_count";
constexpr llvm::StringLiteral kLabelAttrName = "test.label";
constexpr llvm::StringLiteral kMaxIterationsAttrName =
    "test.max_enumerated_iterations";
constexpr llvm::StringLiteral kValueAttrName = "test.value";

std::optional<llvm::APInt> evaluateFunctionArgument(mlir::func::FuncOp function,
                                                    mlir::Value value) {
  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!argument || argument.getOwner() != &function.getBody().front()) {
    return std::nullopt;
  }
  auto valueAttr = function.getArgAttrOfType<mlir::IntegerAttr>(
      argument.getArgNumber(), kValueAttrName);
  return valueAttr ? std::optional(valueAttr.getValue()) : std::nullopt;
}

bool verifyExpectedCount(mlir::Operation *operation,
                         std::optional<std::uint64_t> actualCount) {
  mlir::Attribute expectedAttr = operation->getAttr(kExpectedCountAttrName);
  auto labelAttr = operation->getAttrOfType<mlir::StringAttr>(kLabelAttrName);
  if (!expectedAttr || !labelAttr) {
    operation->emitError() << "test targets require both "
                           << kExpectedCountAttrName << " and "
                           << kLabelAttrName;
    return false;
  }

  llvm::outs() << labelAttr.getValue() << " = ";
  if (actualCount) {
    llvm::outs() << *actualCount << "\n";
  } else {
    llvm::outs() << "unknown\n";
  }

  if (auto expectedInteger = mlir::dyn_cast<mlir::IntegerAttr>(expectedAttr)) {
    if (actualCount && expectedInteger.getValue().getActiveBits() <= 64 &&
        *actualCount == expectedInteger.getValue().getZExtValue()) {
      return true;
    }
  } else if (auto expectedString =
                 mlir::dyn_cast<mlir::StringAttr>(expectedAttr)) {
    if (expectedString.getValue() == "unknown" && !actualCount) {
      return true;
    }
  }

  operation->emitError() << "execution count does not match "
                         << kExpectedCountAttrName;
  return false;
}

bool verifyFunction(mlir::func::FuncOp function) {
  mlir::tt::ExecutionCountAnalysis::Options options;
  if (auto limit =
          function->getAttrOfType<mlir::IntegerAttr>(kMaxIterationsAttrName)) {
    if (limit.getValue().getActiveBits() > 64) {
      function.emitError() << kMaxIterationsAttrName << " does not fit uint64";
      return false;
    }
    options.maxEnumeratedIterations = limit.getValue().getZExtValue();
  }

  mlir::tt::ExecutionCountAnalysis analysis(
      function.getBody(),
      [&](mlir::Value value) {
        return evaluateFunctionArgument(function, value);
      },
      /*regionInvocationCountEvaluator=*/{}, options);

  bool succeeded = true;
  function.walk([&](mlir::Operation *operation) {
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

  bool succeeded = true;
  module->walk([&](mlir::func::FuncOp function) {
    succeeded &= verifyFunction(function);
  });
  return succeeded ? 0 : 1;
}
