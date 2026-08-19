// RUN: ttlang-opt %s -ttkernel-annotate-dfb-use | FileCheck %s

// Records physical DFB compile-time argument indices on ttkernel.thread
// functions after walking get_compile_time_arg_val uses and propagating
// through the call graph, including recursive SCCs. Helpers remain in the
// analysis; declarations are not annotated.

// CHECK: func.func private @unknown()
// CHECK-NEXT: func.func @helper() attributes {ttl.base_cta_index = 3 : i32} {

// CHECK-LABEL: func.func @calls_helper()
// CHECK-SAME: ttl.used_dfb_indices = array<i32: 1, 2>

// CHECK-LABEL: func.func @calls_unknown()
// CHECK-SAME: ttl.used_dfb_indices = array<i32: 0, 1>

// CHECK-LABEL: func.func @recursive()
// CHECK-SAME: ttl.used_dfb_indices = array<i32: 0>

// CHECK-LABEL: func.func @cycle_a()
// CHECK-SAME: ttl.used_dfb_indices = array<i32: 0, 1>

// CHECK-LABEL: func.func @cycle_b()
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

  func.func @calls_helper() attributes {
      ttl.base_cta_index = 3 : i32,
      ttkernel.thread = #ttkernel.thread<compute>} {
    func.call @helper() : () -> ()
    return
  }

  func.func @calls_unknown() attributes {
      ttl.base_cta_index = 2 : i32,
      ttkernel.thread = #ttkernel.thread<noc>} {
    func.call @unknown() : () -> ()
    return
  }

  // Self-recursive call stays in one SCC and keeps the local DFB use.
  func.func @recursive() attributes {
      ttl.base_cta_index = 2 : i32,
      ttkernel.thread = #ttkernel.thread<compute>} {
    %0 = ttkernel.get_compile_time_arg_val(0) : () -> i32
    func.call @recursive() : () -> ()
    return
  }

  // Mutually recursive pair inherits each other's local DFB uses.
  func.func @cycle_a() attributes {
      ttl.base_cta_index = 2 : i32,
      ttkernel.thread = #ttkernel.thread<compute>} {
    %0 = ttkernel.get_compile_time_arg_val(0) : () -> i32
    func.call @cycle_b() : () -> ()
    return
  }

  func.func @cycle_b() attributes {
      ttl.base_cta_index = 2 : i32,
      ttkernel.thread = #ttkernel.thread<compute>} {
    %0 = ttkernel.get_compile_time_arg_val(1) : () -> i32
    func.call @cycle_a() : () -> ()
    return
  }
}
