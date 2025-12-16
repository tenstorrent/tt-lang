#ifndef TTLANG_DIALECT_TTL_PASSES_TD
#define TTLANG_DIALECT_TTL_PASSES_TD

include "mlir/Pass/PassBase.td"

def TTLConvertTTLToTTKernel
    : Pass<"convert-ttl-to-ttkernel", "::mlir::ModuleOp"> {
  let summary = "Lower TTL DMA ops to TTKernel using global barriers (temporary)";
  let description = [{
    Converts TTL DMA ops to TTKernel noc ops. Uses global barriers until TRID
    barriers are available. Covers create_cb, copy, wait MVP path.

    TODO(ttl): Refine lowering to emit real CB handles and proper NOC addresses.
    Issue: #77 (umbrella issue with subtasks #78-#89).
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tt::ttkernel::TTKernelDialect"
  ];
}

def TTLConvertTTLToLinalg
    : Pass<"convert-ttl-to-linalg", "::mlir::func::FuncOp"> {
  let summary = "Lower TTL elementwise compute ops to linalg.generic";
  let description = [{
    Lowers TTL elementwise operations (e.g., add, mul, exp, relu) to linalg.generic on
    tensors as a first stage before TTKernel conversion.
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::linalg::LinalgDialect",
    "::mlir::math::MathDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::tensor::TensorDialect"
  ];
}

def TTLAssignDSTRegisters
    : Pass<"ttl-assign-dst-registers", "::mlir::func::FuncOp"> {
  let summary = "Assign DST registers with a simple linear scan (MVP, no spill)";
  let description = [{
    Performs a lightweight linear scan over tiled linalg.generic producers to
    annotate expected DST usage. Errors if requested capacity exceeds a fixed
    budget. This is an MVP placeholder for a fuller allocator.
  }];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::linalg::LinalgDialect"
  ];
}

#endif // TTLANG_DIALECT_TTL_PASSES_TD
