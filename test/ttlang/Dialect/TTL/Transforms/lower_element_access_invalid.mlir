// Negative tests for unsafe_element_read/unsafe_element_write lowering
// through convert-ttl-to-ttkernel. Verifies that the pass rejects invalid
// inputs when the block has no attached CB.

// RUN: ttlang-opt %s --split-input-file --verify-diagnostics \
// RUN:   -pass-pipeline='builtin.module(convert-ttl-to-ttkernel)'

// B2: unsafe_element_read on a block that does not trace to cb_wait or
// cb_reserve. The block is a bare function argument with no attach_cb
// provenance, so getAttachedCB returns null and the pattern cannot match.

// expected-error @+1 {{failed to legalize}}
module {
func.func @read_bare_arg(%block: tensor<1x1x!ttcore.tile<32x32, f32>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %val = ttl.unsafe_element_read %block[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>> -> i32
  func.return
}
}

// -----

// B2: unsafe_element_write on a block with no attached CB.

// expected-error @+1 {{failed to legalize}}
module {
func.func @write_bare_arg(%block: tensor<1x1x!ttcore.tile<32x32, f32>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %val = arith.constant 42 : i32
  ttl.unsafe_element_write %block[%c0, %c0], %val : tensor<1x1x!ttcore.tile<32x32, f32>>, i32
  func.return
}
}
