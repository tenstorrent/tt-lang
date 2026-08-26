// Verifies that live DFB state crosses a conditionally executed
// reconfiguration when the DFB accesses do not share its predicate.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

// CHECK: DFB logical_id=0 bounded=1
// CHECK: epochs=[{accesses=[0, 1, 2, 3]
// CHECK-SAME: entry_reconfiguration=initial
// CHECK-SAME: active_configurations=[initial, 0]
// CHECK-SAME: terminal_reconfiguration=none

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "produce" dfb_dependencies(
        %dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
    %zero = arith.constant 0 : i64
    %condition = ttl.opaque_call "compute_condition" () {
        condition_result = #ttl.dispatch_condition<0, i64>,
        header = "condition.hpp"} : () -> i64
    %active = arith.cmpi ne, %condition, %zero : i64
    scf.if %active {
      ttl.dfb_reconfiguration #boundary
    }
    ttl.opaque_call "consume" dfb_dependencies(
        %dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
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
