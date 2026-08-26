// Tests ordered allocation-group reuse after a synchronous external inspection
// leaves DFB state unchanged.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPORT

// CHECK: module attributes {ttl.dfb_allocations = [{allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 2 : i32, dfb_index = 0 : i32
// CHECK-LABEL: func.func @ordered_inspect_access
// CHECK: %[[DESCRIPTOR:.*]] = ttl.bind_cb{cb_index = 0, block_count = 1} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
// CHECK-NEXT: %[[QUEUE:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
// CHECK: ttl.opaque_call "inspect" dfb_dependencies(%[[DESCRIPTOR]] : !ttl.cb<{{.*}}>) dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>]

// REPORT: DFB allocation group #ttl.dfb_allocation_group<0> members=[0, 1]
// REPORT: DFB logical_id=0 bounded=1
// REPORT: access 0 effect=none non_transactional=inspect
// REPORT: node (0,0) lifecycle_completion=complete domain_assumption=exact conditional_execution=0 inspection_only=1
// REPORT-NOT: DFB conflict lhs=0 rhs=1

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @ordered_inspect_access()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %descriptor = ttl.bind_cb {cb_index = 0, block_count = 1}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %queue = ttl.bind_cb {cb_index = 1, block_count = 2}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "inspect"
        dfb_dependencies(%descriptor : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>]
        () {header = "inspect.hpp"} : () -> ()
    %produced = ttl.cb_reserve %queue
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %queue : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %consumed = ttl.cb_wait %queue
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %queue : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
