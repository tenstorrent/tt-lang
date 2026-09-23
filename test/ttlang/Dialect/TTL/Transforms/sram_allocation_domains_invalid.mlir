// Independent domains require a complete launch grid before allocation.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 sram-allocation-mode=per-node})'
// expected-error @below {{per-node SRAM allocation requires an exact launch grid}}
module {
  func.func @missing_grid() { return }
}
