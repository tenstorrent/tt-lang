// Summary: Verifies DFB reconfiguration lowering to a runtime interface call.
// RUN: ttlang-opt --convert-ttl-to-ttkernel %s | FileCheck %s

// CHECK-LABEL: func.func @boundary
// CHECK: %[[INDEX:.*]] = arith.constant 0 : index
// CHECK: %[[ADDRESS:.*]] = ttkernel.get_arg_val(%[[INDEX]]) : (index) -> ui32
// CHECK: ttkernel.opaque_call "experimental::reconfigure_dfb_interfaces"(%[[ADDRESS]]) {header = "<cstdint>", unsigned_arg_indices = array<i32: 0>} : (ui32) -> ()
module attributes {
  ttl.dfb_reconfiguration_plan = {
    boundary_ordinals = array<i64: 0>,
    dfbs = []
  }
} {
  func.func @boundary() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>
  } {
    ttl.dfb_reconfiguration #ttl.dfb_reconfiguration<0, participants[#ttl.logical_kernel<kind = compute>, #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">, #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">]>
    return
  }
}
