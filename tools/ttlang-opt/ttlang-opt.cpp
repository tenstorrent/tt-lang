// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Config.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Pipelines/TTLPipelines.h"
#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTCore/IR/TTCoreOps.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "llvm/Support/CommandLine.h"

int main(int argc, char **argv) {
  llvm::cl::AddExtraVersionPrinter([](llvm::raw_ostream &os) {
    os << "ttlang-opt version " << TTLANG_VERSION << "\n";
  });

  // Register upstream MLIR passes
  mlir::registerAllPasses();

  // Register minimal tt-mlir passes (TTKernel only — TTCore passes not needed)
  mlir::tt::ttkernel::registerPasses();

  // Register TTKernel-to-EmitC conversion pass
  mlir::registerPass([]() { return mlir::tt::createConvertTTKernelToEmitC(); });

  // Register tt-lang passes and pipelines
  mlir::tt::ttl::registerTTLPasses();
  mlir::tt::ttl::registerTTLPipelines();

  mlir::DialectRegistry registry;

  // Register upstream MLIR dialects
  mlir::registerAllDialects(registry);

  // Register minimal tt-mlir dialects
  registry.insert<mlir::tt::ttcore::TTCoreDialect>();
  registry.insert<mlir::tt::ttkernel::TTKernelDialect>();
  registry.insert<mlir::tt::ttmetal::TTMetalDialect>();

  // Register tt-lang dialects
  registry.insert<mlir::tt::ttl::TTLDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ttlang optimizer driver\n", registry));
}
