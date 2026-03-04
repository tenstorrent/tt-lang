// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "Dialects.h"

#include "mlir/CAPI/Registration.h"
#include "ttmlir/Dialect/TTCore/IR/TTCore.h"
#include "ttmlir/Dialect/TTKernel/IR/TTKernel.h"
#include "ttmlir/Dialect/TTMetal/IR/TTMetal.h"

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
