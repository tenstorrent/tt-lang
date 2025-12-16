// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang/Dialect/TTL/IR/TTL.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "ttlang/Dialect/TTKernel/Passes.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Pipelines/TTLPipelines.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/RegisterAll.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::tt::registerAllPasses();
  mlir::tt::ttkernel::registerTTKernelPasses();
  mlir::tt::ttl::registerTTLPasses();
  mlir::tt::ttl::registerTTLPipelines();

  mlir::DialectRegistry registry;
  mlir::tt::registerAllDialects(registry);
  mlir::tt::registerAllExtensions(registry);
  registry.insert<mlir::tt::ttl::TTLDialect>();
  registry.insert<mlir::tt::ttkernel::TTKernelDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "ttlang optimizer driver\n", registry));
}
