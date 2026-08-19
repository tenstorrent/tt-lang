// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttkernel-specialize-and-annotate-dfb-use)' | FileCheck %s

// This characterizes the kernel half of per-core specialization. Core (0, 0)
// retains its DFB use while core (0, 1) has the complete DFB-dependent branch
// folded away. Python runtime tests verify that these attributes scope the
// corresponding host DFB descriptors to the surviving core set.

// CHECK-NOT: func.func @conditional_dfb_user()
// CHECK-LABEL: func.func @conditional_dfb_user_c0_0
// CHECK-SAME: ttl.core_coord = {{\[\[}}[[$X:[0-9]+]], [[Y0:[0-9]+]]]]
// CHECK-SAME: ttl.used_dfb_indices = array<i32: [[DFB:[0-9]+]]>
// CHECK: ttkernel.get_compile_time_arg_val([[DFB]])
// CHECK: ttkernel.cb_wait_front
// CHECK-NOT: scf.if
// CHECK-LABEL: func.func @conditional_dfb_user_c0_1
// CHECK-SAME: ttl.core_coord = {{\[\[}}[[$X]], [[Y1:[0-9]+]]]]
// CHECK-SAME: ttl.used_dfb_indices = array<i32>
// CHECK-NOT: ttkernel.get_compile_time_arg_val
// CHECK-NOT: ttkernel.cb_wait_front
// CHECK-NOT: scf.if
// CHECK: return

module attributes {ttl.launch_grid = [1 : i64, 2 : i64]} {
  func.func @conditional_dfb_user() attributes {
      ttl.base_cta_index = 1 : i32,
      ttkernel.thread = #ttkernel.thread<noc>} {
    %c0 = arith.constant 0 : index
    %pages = arith.constant 3 : i32
    %y = "ttkernel.my_logical_y_"() : () -> index
    %is_active = arith.cmpi eq, %y, %c0 : index
    scf.if %is_active {
      %dfb = ttkernel.get_compile_time_arg_val(0)
          : () -> !ttkernel.cb<3, !ttcore.tile<32x32, bf16>>
      ttkernel.cb_wait_front(%dfb, %pages)
          : (!ttkernel.cb<3, !ttcore.tile<32x32, bf16>>, i32) -> ()
    }
    return
  }
}
