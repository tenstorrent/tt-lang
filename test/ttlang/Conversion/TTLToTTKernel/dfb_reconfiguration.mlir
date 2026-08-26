// Summary: Verifies DFB reconfiguration lowering to a runtime interface call.
// RUN: ttlang-opt --convert-ttl-to-ttkernel %s | FileCheck %s

// CHECK-LABEL: func.func @boundary
// CHECK: %[[OFFSET:.*]] = arith.constant 1 : i32
// CHECK-NEXT: %[[CALLER_COUNT:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> i32
// CHECK-NEXT: %[[INDEX:.*]] = arith.addi %[[CALLER_COUNT]], %[[OFFSET]] : i32
// CHECK-NEXT: %[[ADDRESS:.*]] = ttkernel.get_arg_val(%[[INDEX]]) : (i32) -> ui32
// CHECK: ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"(%[[ADDRESS]]) {header = "<cstdint>", unsigned_arg_indices = array<i32: 0>} : (ui32) -> ()
module attributes {
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0, 1>,
    dfbs = []
  }
} {
  func.func @boundary() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>
  } {
    ttl.dfb_reconfiguration #ttl.dfb_reconfiguration<1, participants[#ttl.logical_kernel<kind = compute>, #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">, #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">]>
    return
  }
}
