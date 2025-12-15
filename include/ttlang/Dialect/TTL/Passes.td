#ifndef TTLANG_DIALECT_TTL_PASSES_TD
#define TTLANG_DIALECT_TTL_PASSES_TD

include "mlir/Pass/PassBase.td"

def TTLKernelRegionsToFuncs
    : Pass<"ttl-kernel-regions-to-funcs", "::mlir::ModuleOp"> {
  let summary = "Extract ttl.kernel regions to func.func with thread attributes";
  let description = [{
    Extracts each thread region from `ttl.kernel` operations into a separate
    `func.func` with the appropriate `#ttkernel.thread` attribute.

    For each region:
    1. Creates a private func.func with generated name (kernel_<type>_<index>)
    2. Adds #ttkernel.thread<noc> for datamovement or #ttkernel.thread<compute>
       for compute threads
    3. Moves the region body into the new function
    4. Updates the kernel op's threads attribute with function symbol references

    After this pass, ttl.kernel regions are empty and contain only the thread
    type and symbol reference to the extracted function.
  }];

  let dependentDialects = [
    "::mlir::func::FuncDialect",
    "::mlir::tt::ttkernel::TTKernelDialect"
  ];
}

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

#endif // TTLANG_DIALECT_TTL_PASSES_TD
