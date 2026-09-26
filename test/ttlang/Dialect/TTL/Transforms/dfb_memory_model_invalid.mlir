// Verify that an unknown DFB memory model fails before allocation analysis.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=unknown})'

// expected-error @below {{unknown memory model: unknown}}
module {
  func.func @unknown_memory_model() {
    return
  }
}
