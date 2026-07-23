// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// This test driver parses generic test operations, resolves the possible
// origins of each test.query operand, and compares their labels with the
// test.expected_origins attribute used by MLIR lit tests.

#include "ttlang/Analysis/ValueOriginAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>

namespace {

constexpr llvm::StringLiteral kExpectedOriginsAttrName =
    "test.expected_origins";
constexpr llvm::StringLiteral kLabelAttrName = "test.label";
constexpr llvm::StringLiteral kMaxIndexTuplesAttrName =
    "test.max_enumerated_index_tuples";
constexpr llvm::StringLiteral kMaxLoopIterationsAttrName =
    "test.max_enumerated_loop_iterations";

/// Read a test-only limit used to exercise conservative results.
bool readLimit(mlir::Operation *query, llvm::StringRef attributeName,
               std::uint64_t &limit) {
  auto attribute = query->getAttrOfType<mlir::IntegerAttr>(attributeName);
  if (!attribute) {
    return true;
  }
  if (!attribute.getValue().isIntN(64)) {
    query->emitError() << attributeName << " must fit uint64";
    return false;
  }
  limit = attribute.getValue().getZExtValue();
  return true;
}

/// Return the test label attached to an origin definition or function input.
std::optional<llvm::StringRef> getOriginLabel(mlir::Value value) {
  if (mlir::Operation *definition = value.getDefiningOp()) {
    if (auto label =
            definition->getAttrOfType<mlir::StringAttr>(kLabelAttrName)) {
      return label.getValue();
    }
    return std::nullopt;
  }

  auto argument = mlir::cast<mlir::BlockArgument>(value);
  auto function = mlir::dyn_cast_or_null<mlir::func::FuncOp>(
      argument.getOwner()->getParentOp());
  if (!function || argument.getOwner() != &function.getBody().front()) {
    return std::nullopt;
  }
  if (auto label = function.getArgAttrOfType<mlir::StringAttr>(
          argument.getArgNumber(), kLabelAttrName)) {
    return label.getValue();
  }
  return std::nullopt;
}

/// Compare one query's computed origins with its expected labels.
bool verifyQuery(mlir::Operation *query,
                 mlir::tt::ValueOriginAnalysis &defaultAnalysis) {
  auto expected =
      query->getAttrOfType<mlir::ArrayAttr>(kExpectedOriginsAttrName);
  if (!expected || query->getNumOperands() != 1) {
    query->emitError() << "origin queries require one operand and an array "
                          "test.expected_origins attribute";
    return false;
  }

  mlir::tt::ValueOriginAnalysis::Options options;
  if (!readLimit(query, kMaxLoopIterationsAttrName,
                 options.maxEnumeratedLoopIterations) ||
      !readLimit(query, kMaxIndexTuplesAttrName,
                 options.maxEnumeratedIndexTuples)) {
    return false;
  }
  std::unique_ptr<mlir::tt::ValueOriginAnalysis> customAnalysis;
  mlir::tt::ValueOriginAnalysis *analysis = &defaultAnalysis;
  if (query->hasAttr(kMaxLoopIterationsAttrName) ||
      query->hasAttr(kMaxIndexTuplesAttrName)) {
    customAnalysis = std::make_unique<mlir::tt::ValueOriginAnalysis>(
        query->getParentOfType<mlir::func::FuncOp>(), options);
    analysis = customAnalysis.get();
  }

  llvm::SmallVector<llvm::StringRef> actualLabels;
  for (mlir::Value origin : analysis->getOrigins(query->getOperand(0))) {
    std::optional<llvm::StringRef> maybeLabel = getOriginLabel(origin);
    if (!maybeLabel) {
      query->emitError() << "origin lacks a test.label attribute";
      return false;
    }
    actualLabels.push_back(*maybeLabel);
  }
  llvm::sort(actualLabels);

  llvm::SmallVector<llvm::StringRef> expectedLabels;
  for (mlir::Attribute attribute : expected) {
    auto label = mlir::dyn_cast<mlir::StringAttr>(attribute);
    if (!label) {
      query->emitError() << "expected origin labels must be strings";
      return false;
    }
    expectedLabels.push_back(label.getValue());
  }
  llvm::sort(expectedLabels);

  auto queryLabel = query->getAttrOfType<mlir::StringAttr>(kLabelAttrName);
  llvm::outs() << (queryLabel ? queryLabel.getValue() : "query") << " = [";
  llvm::interleaveComma(actualLabels, llvm::outs());
  llvm::outs() << "]\n";
  if (actualLabels == expectedLabels) {
    return true;
  }
  query->emitError() << "value origins do not match "
                     << kExpectedOriginsAttrName;
  return false;
}

} // namespace

int main(int argumentCount, char **argumentValues) {
  if (argumentCount != 2) {
    llvm::errs() << "usage: ttlang-value-origin-test <input.mlir>\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::cf::ControlFlowDialect,
                  mlir::func::FuncDialect, mlir::scf::SCFDialect,
                  mlir::tensor::TensorDialect>();
  mlir::MLIRContext context(registry);
  context.allowUnregisteredDialects();
  mlir::ParserConfig parserConfig(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(argumentValues[1], parserConfig);
  if (!module) {
    return 1;
  }

  bool succeeded = true;
  module->walk([&](mlir::func::FuncOp function) {
    mlir::tt::ValueOriginAnalysis analysis(function);
    function.walk([&](mlir::Operation *operation) {
      if (operation->hasAttr(kExpectedOriginsAttrName)) {
        succeeded &= verifyQuery(operation, analysis);
      }
    });
  });
  return succeeded ? 0 : 1;
}
