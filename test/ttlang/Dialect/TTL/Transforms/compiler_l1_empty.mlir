// Verifies that an empty allocation plan does not reserve an arena argument.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1})' | FileCheck %s

// CHECK: module attributes {
// CHECK-SAME: ttl.dfb_allocations = []
// CHECK-SAME: ttl.l1_arena_bytes = 0 : i64
// CHECK-SAME: ttl.memory_model = "compiler-l1"
// CHECK-LABEL: func.func @no_storage
// CHECK-SAME: ttl.base_cta_index = 0 : i32
module attributes {ttl.launch_grid = [1, 1]} {
  func.func @no_storage()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 0 : i32} {
    return
  }
}
