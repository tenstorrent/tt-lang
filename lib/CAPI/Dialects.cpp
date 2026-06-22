// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttlang-c/Dialects.h"
#include "ttlang/Dialect/TTL/IR/TTL.h"
#include "ttlang/Dialect/TTL/Passes.h"
#include "ttlang/Dialect/TTL/Pipelines/TTLPipelines.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

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
  ctx->appendDialectRegistry(registry);
}

void ttlangRegisterTTLDialect(MlirDialectRegistry registry) {
  unwrap(registry)->insert<TTLDialect>();
}

void ttlangRegisterUpstreamDialects(MlirDialectRegistry registry) {
  unwrap(registry)
      ->insert<func::FuncDialect, arith::ArithDialect, memref::MemRefDialect,
               scf::SCFDialect, cf::ControlFlowDialect, affine::AffineDialect,
               emitc::EmitCDialect, index::IndexDialect, ub::UBDialect,
               math::MathDialect>();
}

void ttlangRegisterPasses() {
  mlir::tt::ttl::registerTTLPasses();
  mlir::tt::ttl::registerTTLPipelines();

  // Upstream passes the tt-lang pipeline runs. Registered explicitly (instead
  // of via MLIR's RegisterEverything) so the Python CAPI library links only the
  // passes the pipeline uses. Registered by factory so each pass keeps its
  // textual name (canonicalize, cse, symbol-dce, lower-affine).
  registerPass([] { return createCanonicalizerPass(); });
  registerPass([] { return createCSEPass(); });
  registerPass([] { return createSymbolDCEPass(); });
  registerPass([] { return createLowerAffinePass(); });
}
