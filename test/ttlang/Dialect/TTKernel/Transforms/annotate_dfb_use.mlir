// RUN: ttlang-opt %s -ttkernel-annotate-dfb-use | FileCheck %s

// Records physical DFB compile-time argument indices on each function after
// walking get_compile_time_arg_val uses and transitively following calls.

// CHECK-LABEL: func.func private @unknown()
// CHECK-SAME: ttl.used_dfb_indices = array<i32>

// CHECK-LABEL: func.func @helper()
// CHECK-SAME: ttl.used_dfb_indices = array<i32: 1, 2>

// CHECK-LABEL: func.func @calls_helper()
// CHECK-SAME: ttl.used_dfb_indices = array<i32: 1, 2>

// CHECK-LABEL: func.func @calls_unknown()
// CHECK-SAME: ttl.used_dfb_indices = array<i32: 0, 1>

module {
  func.func private @unknown()

  func.func @helper() attributes {ttl.base_cta_index = 3 : i32} {
    %scalar_dfb_id = ttkernel.get_compile_time_arg_val(1) : () -> i32
    %dfb = ttkernel.get_compile_time_arg_val(2)
        : () -> !ttkernel.cb<3, !ttcore.tile<32x32, bf16>>
    %pages = arith.constant 1 : i32
    %effective_pages = arith.addi %scalar_dfb_id, %pages : i32
    ttkernel.cb_wait_front(%dfb, %effective_pages)
        : (!ttkernel.cb<3, !ttcore.tile<32x32, bf16>>, i32) -> ()
    return
  }

  func.func @calls_helper() attributes {ttl.base_cta_index = 3 : i32} {
    func.call @helper() : () -> ()
    return
  }

  func.func @calls_unknown() attributes {ttl.base_cta_index = 2 : i32} {
    func.call @unknown() : () -> ()
    return
  }
}
