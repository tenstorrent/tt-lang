// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang-c/Dialects.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/Transforms/BufferizableOpInterfaceImpl.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"

using namespace mlir;
using namespace mlir::tt::ttl;

//===----------------------------------------------------------------------===//
// TTL Dialect Registration
//===----------------------------------------------------------------------===//

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(TTL, ttl, TTLDialect)

void ttlangRegisterAllDialects(MlirContext context) {
  MLIRContext *ctx = unwrap(context);
  DialectRegistry registry;
  registry.insert<TTLDialect>();
  registerBufferizableOpInterfaceExternalModels(registry);
  ctx->appendDialectRegistry(registry);
}

void ttlangRegisterTTLDialect(MlirDialectRegistry registry) {
  auto *unwrapped = unwrap(registry);
  unwrapped->insert<TTLDialect>();
  registerBufferizableOpInterfaceExternalModels(*unwrapped);
}
