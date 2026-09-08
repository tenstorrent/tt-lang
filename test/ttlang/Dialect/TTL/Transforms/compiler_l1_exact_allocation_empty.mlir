// Verifies exact placement accepts an allocation problem without payload regions.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 l1-allocation-strategy=exact})' | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: ttl.dfb_allocations = []
// CHECK-SAME: ttl.l1_arena_bytes = 0 : i64
// CHECK-SAME: ttl.memory_model = "compiler-l1"
module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
}
