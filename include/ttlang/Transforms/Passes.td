#ifndef TTLANG_TRANSFORMS_PASSES_TD
#define TTLANG_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def D2MSimpleAllocate : Pass<"d2m-simple-allocate", "::mlir::func::FuncOp"> {
  let summary = "Simplified L1/DRAM address allocation for tt-lang DSL";
  let description = [{
    Assigns L1/DRAM addresses to memref.alloc operations using liveness-based
    first-fit allocation. Designed for tt-lang DSL where streams are explicit.
  }];

  let dependentDialects = [
    "::mlir::memref::MemRefDialect",
    "::mlir::tt::ttcore::TTCoreDialect"
  ];

  let options = [
    Option<"testAssumeL1Capacity", "test-assume-l1-capacity", "int64_t",
           /*default=*/"0",
           "Override L1 capacity for testing (0 = use hardware capacity)">
  ];
}

#endif // TTLANG_TRANSFORMS_PASSES_TD
