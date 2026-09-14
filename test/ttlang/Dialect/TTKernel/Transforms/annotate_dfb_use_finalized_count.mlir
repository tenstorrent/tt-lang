// RUN: ttlang-opt %s -ttkernel-annotate-dfb-use | FileCheck %s

// A reconfiguration table address follows the physical DFB arguments. The
// finalized allocation count prevents that additional argument from being
// recorded as a DFB use.

// CHECK-LABEL: func.func @kernel()
// CHECK-SAME: ttl.used_dfb_indices = array<i32: 0>

module attributes {ttl.dfb_allocations = [{dfb_index = 0 : i32}]} {
  func.func @kernel() attributes {
      ttl.base_cta_index = 2 : i32,
      ttkernel.thread = #ttkernel.thread<compute>} {
    %dfb = ttkernel.get_compile_time_arg_val(0)
        : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
    %one = arith.constant 1 : i32
    ttkernel.cb_wait_front(%dfb, %one)
        : (!ttkernel.cb<1, !ttcore.tile<32x32, bf16>>, i32) -> ()
    %configuration_table = ttkernel.get_compile_time_arg_val(1) : () -> i32
    %ignored = arith.addi %configuration_table, %one : i32
    return
  }
}
