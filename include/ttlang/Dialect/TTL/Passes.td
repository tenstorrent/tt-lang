#ifndef TTLANG_DIALECT_TTL_PASSES_TD
#define TTLANG_DIALECT_TTL_PASSES_TD

include "mlir/Pass/PassBase.td"

def TTLConvertTTLToTTKernel
    : Pass<"convert-ttl-to-ttkernel", "::mlir::ModuleOp"> {
  let summary = "Lower TTL DMA ops to TTKernel using global barriers (temporary)";
  let description = [{
    Converts TTL DMA ops to TTKernel noc ops. Uses global barriers until TRID
    barriers are available. Covers bind_cb, copy, wait MVP path.

    TODO(ttl): Refine lowering to emit real CB handles and proper NOC addresses.
    Issue: #77 (umbrella issue with subtasks #78-#89).
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tt::ttkernel::TTKernelDialect"
  ];
}

def TTLAssignCBIndices
    : Pass<"ttl-assign-cb-indices", "::mlir::func::FuncOp"> {
  let summary = "Assign circular buffer indices and check uniqueness";
  let description = [{
    Assigns missing circular buffer indices on ttl.bind_cb ops, preserving
    user-specified indices and diagnosing index conflicts. Enforces a fixed
    per-core slot budget (0..31).
  }];

  let dependentDialects = [
    "::mlir::func::FuncDialect"
  ];
}

def TTLConvertTTLToCompute
    : Pass<"convert-ttl-to-compute", "::mlir::func::FuncOp"> {
  let summary = "Lower TTL elementwise tensor ops to ttl.compute with tile ops";
  let description = [{
    Lowers TTL elementwise tensor operations (e.g., add, mul, exp, relu) to
    ttl.compute with SSA-style tile operations in the body. This enables:
    - Fusion of matmul + elementwise operations in the same compute body
    - DST register tiling via TilingInterface
    - Standard SSA optimizations (CSE, DCE, canonicalization)
  }];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

def TTLAssignDSTRegisters
    : Pass<"ttl-assign-dst-registers", "::mlir::func::FuncOp"> {
  let summary = "Assign DST registers with a simple linear scan (MVP, no spill)";
  let description = [{
    Performs a lightweight linear scan over ttl.compute producers to annotate
    expected DST usage. Errors if requested capacity exceeds a fixed budget.
    This is an MVP placeholder for a fuller allocator.
  }];

  let dependentDialects = [
    "::mlir::func::FuncDialect"
  ];
}

#endif // TTLANG_DIALECT_TTL_PASSES_TD
