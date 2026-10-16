// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "DFBNetworkPlanning.h"

#include "ttlang/Dialect/TTL/IR/TTLOpsUtils.h"
#include "ttlang/Dialect/TTL/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::tt::ttl {

#define GEN_PASS_DEF_TTLPRINTDFBNETWORKPLAN
#include "ttlang/Dialect/TTL/Passes.h.inc"

namespace {

/// Prints the function that contains `op`, e.g. "@reader".
static void printEnclosingFunction(Operation *op, raw_ostream &output) {
  auto function = op->getParentOfType<func::FuncOp>();
  output << "@" << (function ? function.getSymName() : "<none>");
}

struct TTLPrintDFBNetworkPlanPass
    : public impl::TTLPrintDFBNetworkPlanBase<TTLPrintDFBNetworkPlanPass> {
  void runOnOperation() override {
    DFBNetworkPlanner planner(getOperation());
    DFBNetworkPlanningResult result = planner.build();

    if (result.isInvalidIR()) {
      emitPlanningDiagnostic(result.getInvalidIR());
      return signalPassFailure();
    }

    raw_ostream &output = llvm::errs();
    if (result.isRejected()) {
      output << "DFB network plan: rejected\n";
      for (const DFBNetworkRejection &rejection : result.getRejection()) {
        output << "  " << rejection.reason << "\n";
      }
      return;
    }
    output << "DFB network plan: planned\n";
    for (const DFBNetworkExpansion &expansion :
         result.getPlan().getNetworks()) {
      DFBNetworkOp network = expansion.network;
      output << "  network @" << network.getSymName() << "\n";
      for (const DFBNetworkRecordPlan &record : expansion.records) {
        output << "    " << record.record->getName() << " @"
               << cast<SymbolOpInterface>(record.record).getName() << "\n";
        for (const DFBNetworkEndpoint &endpoint : record.endpoints) {
          output << "      DFB " << endpoint.dfbId << " "
                 << stringifyDFBNetworkRole(endpoint.role) << ": bindings=[";
          llvm::interleaveComma(endpoint.bindings, output,
                                [&](BindCBOp binding) {
                                  printEnclosingFunction(binding, output);
                                });
          output << "] kernel-ops=[";
          llvm::interleaveComma(endpoint.kernelOps, output, [&](Operation *op) {
            printEnclosingFunction(op, output);
            output << " " << op->getName();
          });
          output << "]\n";
        }
      }
    }
  }
};

} // namespace

} // namespace mlir::tt::ttl
