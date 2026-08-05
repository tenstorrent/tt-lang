// Tests physical DFB-index dataflow through pure integer operations.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s

// A comparison derives a predicate, not another physical index. The predicate
// may control a region without hiding a DFB-index use from the analysis.

// CHECK: module attributes {ttl.dfb_allocations = [{{.*}}dfb_index = 0 : i32{{.*}}]}
// CHECK-LABEL: func.func @compare_physical_index
// CHECK: %[[DFB:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK: %[[INDEX:.*]] = ttl.get_dfb_id %[[DFB]]
// CHECK: %[[IS_ZERO:.*]] = arith.cmpi eq, %[[INDEX]], %{{.*}} : i32
// CHECK: scf.if %[[IS_ZERO]]

module attributes {ttl.launch_grid = [1 : i64, 1 : i64]} {
  func.func @compare_physical_index()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb_index = ttl.get_dfb_id %dfb
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %zero = arith.constant 0 : i32
    %is_zero = arith.cmpi eq, %dfb_index, %zero : i32
    scf.if %is_zero {
    }
    return
  }
}
