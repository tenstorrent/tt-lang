// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "Dialects.h"

#include "mlir/CAPI/Registration.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "ttmlir/Conversion/TTKernelToEmitC/TTKernelToEmitC.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTKernel/Transforms/Passes.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"
#include "ttmlir/Target/TTKernel/TTKernelToCpp.h"

#include <cstdlib>
#include <cstring>

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TT, tt, mlir::tt::ttcore::TTCoreDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TTKernel, ttkernel,
                                      mlir::tt::ttkernel::TTKernelDialect)
MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TTMetal, ttmetal,
                                      mlir::tt::ttmetal::TTMetalDialect)

void ttmlirMinimalRegisterAllDialects(MlirDialectRegistry registry) {
  mlir::DialectRegistry *reg = unwrap(registry);
  reg->insert<mlir::tt::ttcore::TTCoreDialect>();
  reg->insert<mlir::tt::ttkernel::TTKernelDialect>();
  reg->insert<mlir::tt::ttmetal::TTMetalDialect>();
}

void ttmlirMinimalRegisterPasses() {
  mlir::tt::ttkernel::registerPasses();
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::tt::createConvertTTKernelToEmitC();
  });
}

bool ttmlirMinimalRunTTKernelToEmitC(MlirModule module) {
  mlir::Operation *op = unwrap(mlirModuleGetOperation(module));
  mlir::PassManager pm(op->getName());
  pm.addPass(mlir::tt::createConvertTTKernelToEmitC());
  return mlir::succeeded(pm.run(op));
}

char *ttmlirMinimalTranslateKernelToCpp(MlirModule module,
                                        const char *kernelName) {
  mlir::ModuleOp mod =
      mlir::cast<mlir::ModuleOp>(unwrap(mlirModuleGetOperation(module)));
  std::string output;
  llvm::raw_string_ostream os(output);
  if (mlir::failed(mlir::tt::ttkernel::translateTopLevelKernelToCpp(
          mod, os, kernelName))) {
    return nullptr;
  }
  char *result = static_cast<char *>(malloc(output.size() + 1));
  memcpy(result, output.c_str(), output.size() + 1);
  return result;
}
