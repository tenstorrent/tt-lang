// Metal descriptors do not support independent compiler SRAM layouts.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{sram-allocation-mode=per-core})'
// expected-error @below {{per-core SRAM allocation requires memory-model=compiler-l1}}
module attributes {ttl.launch_grid = [1, 1]} {
  func.func @metal_mode() { return }
}
