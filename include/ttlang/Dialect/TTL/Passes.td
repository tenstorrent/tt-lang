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
  let summary = "Lower TTL ops to TTKernel";
  let description = [{
    Converts TTL ops to TTKernel ops:
    - CB args converted to get_compile_time_arg_val
    - CB sync ops (reserve/push/wait/pop) to TTKernel equivalents
    - Copy ops to NOC async read/write using runtime args for L1 addresses
    - Wait ops to NOC barriers based on transfer direction
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tt::ttkernel::TTKernelDialect"
  ];
}

def TTLEraseDeadOps
    : Pass<"ttl-erase-dead-ops", "::mlir::ModuleOp"> {
  let summary = "Erase dead ops after TTL conversion";
  let description = [{
    Cleans up dead operations after TTL to TTKernel conversion:
    - Dead ttcore.get_global ops (no longer needed after conversion)
    - Dead unrealized_conversion_cast ops from type conversions

    Iterates until fixpoint to handle cascading dead ops.
  }];
}

#endif // TTLANG_DIALECT_TTL_PASSES_TD
