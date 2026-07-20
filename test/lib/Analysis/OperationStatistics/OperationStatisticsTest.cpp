// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Analysis/ExecutionCountAnalysis.h"
#include "ttlang/Dialect/TTCore/IR/TTCore.h"
#include "ttlang/Dialect/TTKernel/IR/TTKernel.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/CheckedArithmetic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>

namespace {

/// Counts static operations and their total runtime instances for one name.
struct OperationStatistics {
  std::uint64_t staticOccurrences = 0;
  std::optional<std::uint64_t> dynamicInstances = 0;
};

/// Add one operation, preserving unknown when any dynamic count is unknown.
void addOperationCount(OperationStatistics &statistics,
                       std::optional<std::uint64_t> maybeExecutionCount) {
  ++statistics.staticOccurrences;
  if (!statistics.dynamicInstances || !maybeExecutionCount) {
    statistics.dynamicInstances = std::nullopt;
    return;
  }
  statistics.dynamicInstances = llvm::checkedAddUnsigned(
      *statistics.dynamicInstances, *maybeExecutionCount);
}

/// Print deterministic operation statistics for one function body invocation.
void printFunctionStatistics(mlir::func::FuncOp function) {
  mlir::tt::ExecutionCountAnalysis analysis(function.getBody());
  llvm::StringMap<OperationStatistics> statisticsByName;
  function.getBody().walk([&](mlir::Operation *operation) {
    addOperationCount(statisticsByName[operation->getName().getStringRef()],
                      analysis.getExecutionCount(operation));
  });

  llvm::SmallVector<llvm::StringRef> operationNames;
  operationNames.reserve(statisticsByName.size());
  for (const auto &entry : statisticsByName) {
    operationNames.push_back(entry.getKey());
  }
  llvm::sort(operationNames);

  llvm::outs() << "func @" << function.getSymName() << "\n";
  for (llvm::StringRef operationName : operationNames) {
    const OperationStatistics &statistics = statisticsByName[operationName];
    llvm::outs() << "  " << operationName
                 << " static_occurrences=" << statistics.staticOccurrences
                 << " dynamic_instances=";
    if (statistics.dynamicInstances) {
      llvm::outs() << *statistics.dynamicInstances;
    } else {
      llvm::outs() << "unknown";
    }
    llvm::outs() << "\n";
  }
}

} // namespace

int main(int argumentCount, char **argumentValues) {
  llvm::InitLLVM initLLVM(argumentCount, argumentValues);
  llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                           llvm::cl::desc("<input MLIR file>"),
                                           llvm::cl::init("-"));
  llvm::cl::ParseCommandLineOptions(argumentCount, argumentValues,
                                    "tt-lang operation statistics\n");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry
      .insert<mlir::tt::ttcore::TTCoreDialect,
              mlir::tt::ttkernel::TTKernelDialect, mlir::tt::ttl::TTLDialect>();

  mlir::MLIRContext context(registry);
  mlir::ParserConfig parserConfig(&context);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(inputFilename, parserConfig);
  if (!module) {
    return 1;
  }

  module->walk(printFunctionStatistics);
  return 0;
}
