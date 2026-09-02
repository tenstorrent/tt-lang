// Verifies that a lifecycle cannot enter a conditional reconfiguration epoch
// unless its accesses share the boundary predicate.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

// CHECK: DFB logical_id=0 bounded=1
// CHECK: entry_reconfiguration=initial,active_configurations=[initial],terminal_reset=none,terminal_reconfiguration=0
// CHECK: DFB logical_id=1 bounded=0
// CHECK: lifecycle_completion=unsupported-control-flow
// CHECK: Total DFB count: 2
// CHECK: DFB assignment: logical DFB 0 -> physical index 0 storage index 1 (bounded)
// CHECK: DFB assignment: logical DFB 1 -> physical index 1 storage index 0 (unbounded)

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 3} {dfb_id = 1 : index}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 3>
    %zero = arith.constant 0 : i64
    %condition = ttl.opaque_call "compute_condition" () {
        condition_result = #ttl.dispatch_condition<0, i64>,
        header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %condition, %zero : i64
    scf.if %active {
      ttl.opaque_call "first" dfb_dependencies(
          %first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                       #ttl.dfb_protocol_effect<push, 0, 1>,
                       #ttl.dfb_protocol_effect<wait, 0, 1>,
                       #ttl.dfb_protocol_effect<pop, 0, 1>]
          () {header = "effects.hpp"} : () -> ()
      ttl.dfb_reconfiguration #boundary
    }
    ttl.opaque_call "second" dfb_dependencies(
        %second : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 3>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 2>,
                     #ttl.dfb_protocol_effect<push, 0, 2>,
                     #ttl.dfb_protocol_effect<wait, 0, 2>,
                     #ttl.dfb_protocol_effect<pop, 0, 2>]
        () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %zero = arith.constant 0 : i64
    %condition = ttl.opaque_call "reader_condition" () {
        condition_result = #ttl.dispatch_condition<0, i64>,
        header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %condition, %zero : i64
    scf.if %active {
      ttl.dfb_reconfiguration #boundary
    }
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    %zero = arith.constant 0 : i64
    %condition = ttl.opaque_call "writer_condition" () {
        condition_result = #ttl.dispatch_condition<0, i64>,
        header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %condition, %zero : i64
    scf.if %active {
      ttl.dfb_reconfiguration #boundary
    }
    return
  }
}
