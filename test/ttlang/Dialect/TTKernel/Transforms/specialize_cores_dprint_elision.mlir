// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttkernel-specialize-and-annotate-dfb-use)' 2>%t.err | FileCheck %s
// RUN: FileCheck --check-prefix=WARN %s < %t.err

// A DFB print sits outside the coordinate branch, so it is cloned onto every
// core. After folding, core (0, 0) still waits on the DFB and keeps the print.
// Core (0, 1) has only the print left; that print is dropped so the host does
// not allocate the DFB only for debugging.

// WARN: warning: eliminating debug print of unused DFB {{[0-9]+}} on specialized core

// CHECK-NOT: func.func @print_outside_branch()
// CHECK-LABEL: func.func @print_outside_branch_c0_0
// CHECK-SAME: ttl.used_dfb_indices = array<i32: [[DFB:[0-9]+]]>
// CHECK: ttkernel.get_compile_time_arg_val([[DFB]])
// CHECK: emitc.verbatim "ttmlir::dprint(ttmlir::CBPrinter({}));"
// CHECK: ttkernel.cb_wait_front
// CHECK-LABEL: func.func @print_outside_branch_c0_1
// CHECK-SAME: ttl.used_dfb_indices = array<i32>
// CHECK-NOT: ttkernel.get_compile_time_arg_val
// CHECK-NOT: CBPrinter
// CHECK-NOT: ttkernel.cb_wait_front
// CHECK: return

module attributes {ttl.launch_grid = [1 : i64, 2 : i64]} {
  func.func @print_outside_branch() attributes {
      ttl.base_cta_index = 1 : i32,
      ttkernel.thread = #ttkernel.thread<noc>} {
    %dfb = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<3, !ttcore.tile<32x32, bf16>>
    %id = ttkernel.get_compile_time_arg_val(0) : () -> i32
    emitc.verbatim "ttmlir::dprint(ttmlir::CBPrinter({}));" args %id : i32
    emitc.verbatim "DPRINT(\"\\n\");"
    %c0 = arith.constant 0 : index
    %pages = arith.constant 3 : i32
    %y = "ttkernel.my_logical_y_"() : () -> index
    %is_active = arith.cmpi eq, %y, %c0 : index
    scf.if %is_active {
      ttkernel.cb_wait_front(%dfb, %pages)
          : (!ttkernel.cb<3, !ttcore.tile<32x32, bf16>>, i32) -> ()
    }
    return
  }
}
