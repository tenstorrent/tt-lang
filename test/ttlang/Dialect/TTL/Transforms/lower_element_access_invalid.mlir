// Negative tests for ttl-lower-element-access-to-emitc pass. Verifies that
// the pass rejects invalid inputs with appropriate error diagnostics.

// RUN: ttlang-opt %s --split-input-file --verify-diagnostics \
// RUN:   -pass-pipeline='builtin.module(ttl-lower-element-access-to-emitc)'

// B2: element_read on a block that does not trace to cb_wait or cb_reserve.
// The block is a bare function argument with no attach_cb provenance, so
// getAttachedCB returns null and the conversion pattern cannot match.

func.func @read_bare_arg(%block: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  // expected-error @below {{cannot find attached CB for element_read block}}
  // expected-error @below {{failed to legalize operation 'ttl.element_read'}}
  %val = ttl.element_read %block[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> i32
  func.return
}

// -----

// B2: element_write on a block with no attached CB.

func.func @write_bare_arg(%block: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %c0 = arith.constant 0 : index
  %val = arith.constant 42 : i32
  // expected-error @below {{cannot find attached CB for element_write block}}
  // expected-error @below {{failed to legalize operation 'ttl.element_write'}}
  ttl.element_write %block[%c0, %c0], %val : tensor<1x1x!ttcore.tile<32x32, bf16>>, i32
  func.return
}
