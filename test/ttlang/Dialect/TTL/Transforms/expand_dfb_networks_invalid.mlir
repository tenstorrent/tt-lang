// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --ttl-expand-dfb-networks -o /dev/null

// Summary: Verifies that the expansion pass reports a DFB network that breaks a
// module-level rule as invalid, with one case per rule.

// Test that a record naming a DFB that no kernel binds is invalid.
module {
  ttl.dfb.network @unbound_output {
    // expected-error @below {{DFB 2 has no ttl.bind_cb in any kernel}}
    ttl.dfb.fork @fork 0 : index -> [1, 2] storage = <replicated>
  }

  func.func @kernel() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %output = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Test that a DFB used by records in two networks is invalid.
module {
  ttl.dfb.network @first {
    // expected-note @below {{first used in network @first here}}
    ttl.dfb.fork @fork 0 : index -> [1, 2] storage = <replicated>
  }
  ttl.dfb.network @second {
    // expected-error @below {{DFB 0 is used by more than one network}}
    ttl.dfb.fork @fork 0 : index -> [3, 4] storage = <replicated>
  }

  func.func @kernel() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb0 = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb1 = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb2 = ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 2 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb3 = ttl.bind_cb{cb_index = 3, block_count = 2} {dfb_id = 3 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb4 = ttl.bind_cb{cb_index = 4, block_count = 2} {dfb_id = 4 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Test that a kernel consuming a fork source is invalid: the fork holds the
// source's consumer capability.
module {
  ttl.dfb.network @consumed_source {
    // expected-error @below {{DFB 0 is the source of @fork, but a kernel consumes it}}
    ttl.dfb.fork @fork 0 : index -> [1, 2] storage = <replicated>
  }

  func.func @kernel() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb0 = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb1 = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb2 = ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 2 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-note @below {{consumed here}}
    %block = ttl.cb_wait %dfb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %dfb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Test that a kernel consuming a merge input is invalid: the merge holds each
// input's consumer capability.
module {
  ttl.dfb.network @consumed_input {
    // expected-error @below {{DFB 1 is an input of @merge, but a kernel consumes it}}
    ttl.dfb.merge @merge [0 : index, 1 : index] policy = <round_robin>
  }

  func.func @kernel() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb0 = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb1 = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-note @below {{consumed here}}
    %block = ttl.cb_wait %dfb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %dfb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Test that a kernel producing a split output is invalid: the split is the
// output's only producer.
module {
  ttl.dfb.network @produced_output {
    // expected-error @below {{DFB 2 is an output of @split, but a kernel produces it}}
    ttl.dfb.split @split 0 : index -> [1, 2] policy = <round_robin>
  }

  func.func @kernel() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb0 = ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb1 = ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %dfb2 = ttl.bind_cb{cb_index = 2, block_count = 2} {dfb_id = 2 : index} : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    // expected-note @below {{produced here}}
    %block = ttl.cb_reserve %dfb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %dfb2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
