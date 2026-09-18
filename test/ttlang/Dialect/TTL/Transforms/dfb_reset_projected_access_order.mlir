// Tests reset ordering when a nested access lacks a dedicated event span.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPORT

// The dynamic scratch access projects to the outer loop, which also contains
// every reset. Its incomplete local ordering must not create reverse relations
// between reset boundaries or prevent the interleaved group from sharing.

// CHECK: module attributes {ttl.dfb_allocations = [{allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 3 : i32, dfb_index = 0 : i32
// CHECK-SAME: {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 3 : i32, dfb_index = 1 : i32
// CHECK: %{{.*}} = ttl.bind_cb{cb_index = 0, block_count = 3} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
// CHECK: %{{.*}} = ttl.bind_cb{cb_index = 0, block_count = 3} {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
// CHECK: %{{.*}} = ttl.bind_cb{cb_index = 1, block_count = 3} {dfb_id = 2 : index}
// REPORT: DFB allocation group #ttl.dfb_allocation_group<0> launch_node=(0,0) epoch_order=[0:0, 1:0, 0:1, 1:1]
// REPORT: DFB allocation group #ttl.dfb_allocation_group<0> members=[0, 1] envelope_bytes=6144 handoff=proven
// REPORT: Total DFB count: 2

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @nested_reset_target_access()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "nested_reset_target_access">,
                  ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %second = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %dynamic_scratch = ttl.bind_cb {cb_index = 2, block_count = 3}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %dynamic_count = ttl.opaque_call "dynamic_count" () {header = "count.hpp"} : () -> i32
    %dynamic_upper_bound = arith.index_cast %dynamic_count : i32 to index
    scf.for %outer = %zero to %one step %one {
      ttl.opaque_call "first_epoch_0" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
      ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "nested_reset_target_access">, <kind = data_movement, identity = "reader", operation = "nested_reset_target_access">, <kind = data_movement, identity = "writer", operation = "nested_reset_target_access">]>
      ttl.opaque_call "second_epoch_0" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
      ttl.reset_all_dfbs <1, participants[<kind = compute, identity = "compute", operation = "nested_reset_target_access">, <kind = data_movement, identity = "reader", operation = "nested_reset_target_access">, <kind = data_movement, identity = "writer", operation = "nested_reset_target_access">]>
      scf.for %scratch_iteration = %zero to %dynamic_upper_bound step %one {
        %scratch_output = ttl.cb_reserve %dynamic_scratch : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_push %dynamic_scratch : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
        %scratch_input = ttl.cb_wait %dynamic_scratch : <[1, 1], !ttcore.tile<32x32, bf16>, 3> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        ttl.cb_pop %dynamic_scratch : <[1, 1], !ttcore.tile<32x32, bf16>, 3>
      }
      ttl.opaque_call "first_epoch_1" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
      ttl.reset_all_dfbs <2, participants[<kind = compute, identity = "compute", operation = "nested_reset_target_access">, <kind = data_movement, identity = "reader", operation = "nested_reset_target_access">, <kind = data_movement, identity = "writer", operation = "nested_reset_target_access">]>
      ttl.opaque_call "second_epoch_1" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    }
    return
  }

  func.func @reader()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "nested_reset_target_access">,
                  ttl.noc_index = 0 : i32, ttl.base_cta_index = 3 : i32,
                  ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %second = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %dynamic_scratch = ttl.bind_cb {cb_index = 2, block_count = 3}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %outer = %zero to %one step %one {
      ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "nested_reset_target_access">, <kind = data_movement, identity = "reader", operation = "nested_reset_target_access">, <kind = data_movement, identity = "writer", operation = "nested_reset_target_access">]>
      ttl.reset_all_dfbs <1, participants[<kind = compute, identity = "compute", operation = "nested_reset_target_access">, <kind = data_movement, identity = "reader", operation = "nested_reset_target_access">, <kind = data_movement, identity = "writer", operation = "nested_reset_target_access">]>
      ttl.reset_all_dfbs <2, participants[<kind = compute, identity = "compute", operation = "nested_reset_target_access">, <kind = data_movement, identity = "reader", operation = "nested_reset_target_access">, <kind = data_movement, identity = "writer", operation = "nested_reset_target_access">]>
    }
    return
  }

  func.func @writer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "nested_reset_target_access">,
                  ttl.noc_index = 1 : i32, ttl.base_cta_index = 3 : i32,
                  ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %second = ttl.bind_cb {cb_index = 1, block_count = 3}
        {allocation_group = #ttl.dfb_allocation_group<0>, dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %dynamic_scratch = ttl.bind_cb {cb_index = 2, block_count = 3}
        {dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    scf.for %outer = %zero to %one step %one {
      ttl.reset_all_dfbs <0, participants[<kind = compute, identity = "compute", operation = "nested_reset_target_access">, <kind = data_movement, identity = "reader", operation = "nested_reset_target_access">, <kind = data_movement, identity = "writer", operation = "nested_reset_target_access">]>
      ttl.reset_all_dfbs <1, participants[<kind = compute, identity = "compute", operation = "nested_reset_target_access">, <kind = data_movement, identity = "reader", operation = "nested_reset_target_access">, <kind = data_movement, identity = "writer", operation = "nested_reset_target_access">]>
      ttl.reset_all_dfbs <2, participants[<kind = compute, identity = "compute", operation = "nested_reset_target_access">, <kind = data_movement, identity = "reader", operation = "nested_reset_target_access">, <kind = data_movement, identity = "writer", operation = "nested_reset_target_access">]>
    }
    return
  }
}
