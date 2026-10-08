// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Summary: Test that invalid DFB networks, records, and network enum attribute
// spellings are rejected by the parser or the verifiers.

// Test that a non-module parent of ttl.dfb.network is invalid.
func.func @non_module_parent() {
  // expected-error @below {{'ttl.dfb.network' op expects parent op 'builtin.module'}}
  ttl.dfb.network @net {}
  return
}

// -----

// Test that 'contiguous' is rejected as a fork storage option.
// expected-error @below {{got: contiguous}}
// expected-error @below {{failed to parse TTL_DFBNetworkForkStorageAttr parameter 'value'}}
ttl.dfb.network @fork_invalid_contiguous attributes {a = #ttl.dfb_network_fork_storage<contiguous>} {}

// -----

// Test that 'replicated' is rejected as a split policy option.
// expected-error @below {{got: replicated}}
// expected-error @below {{failed to parse TTL_DFBNetworkSplitPolicyAttr parameter 'value'}}
ttl.dfb.network @split_invalid_replicated attributes {a = #ttl.dfb_network_split_policy<replicated>} {}

// -----

// Test that 'auto' is rejected as a split policy option.
// expected-error @below {{got: auto}}
// expected-error @below {{failed to parse TTL_DFBNetworkSplitPolicyAttr parameter 'value'}}
ttl.dfb.network @split_invalid_auto attributes {a = #ttl.dfb_network_split_policy<auto>} {}

// -----

// Test that 'first_ready' is rejected as a split policy option.
// expected-error @below {{got: first_ready}}
// expected-error @below {{failed to parse TTL_DFBNetworkSplitPolicyAttr parameter 'value'}}
ttl.dfb.network @split_invalid_first_ready attributes {a = #ttl.dfb_network_split_policy<first_ready>} {}

// -----

// Test that 'replicated' is rejected as a merge policy option.
// expected-error @below {{got: replicated}}
// expected-error @below {{failed to parse TTL_DFBNetworkMergePolicyAttr parameter 'value'}}
ttl.dfb.network @merge_invalid_replicated attributes {a = #ttl.dfb_network_merge_policy<replicated>} {}

// -----

// Test that 'auto' is rejected as a merge policy option.
// expected-error @below {{got: auto}}
// expected-error @below {{failed to parse TTL_DFBNetworkMergePolicyAttr parameter 'value'}}
ttl.dfb.network @merge_invalid_auto attributes {a = #ttl.dfb_network_merge_policy<auto>} {}

// -----

// Test that 'contiguous' is rejected as a merge policy option.
// expected-error @below {{got: contiguous}}
// expected-error @below {{failed to parse TTL_DFBNetworkMergePolicyAttr parameter 'value'}}
ttl.dfb.network @merge_invalid_contiguous attributes {a = #ttl.dfb_network_merge_policy<contiguous>} {}

// -----

// Test that a fork with no outputs is rejected.
ttl.dfb.network @fork_invalid_no_outputs_net {
  // expected-error @below {{requires at least one output}}
  ttl.dfb.fork @fork 0 : index -> [] storage = <auto>
}

// -----

// Test that a fork with duplicated outputs is rejected.
ttl.dfb.network @fork_invalid_duplicated_outputs_net {
  // expected-error @below {{output 0 is duplicated}}
  ttl.dfb.fork @fork 0 : index -> [0, 0] storage = <auto>
}

// -----

// Test that a fork with a source DFB id that is also an output is rejected.
ttl.dfb.network @fork_invalid_source_dfb_id_is_output_net {
  // expected-error @below {{source DFB id 0 is also an output}}
  ttl.dfb.fork @fork 0 : index -> [0] storage = <auto>
}

// -----

// Test that a split with no outputs is rejected.
ttl.dfb.network @split_invalid_no_outputs_net {
  // expected-error @below {{requires at least one output}}
  ttl.dfb.split @split 0 : index -> [] policy = <round_robin>
}

// -----

// Test that a split with duplicated outputs is rejected.
ttl.dfb.network @split_invalid_duplicated_outputs_net {
  // expected-error @below {{output 0 is duplicated}}
  ttl.dfb.split @split 0 : index -> [0, 0] policy = <round_robin>
}

// -----

// Test that a split with a source DFB id that is also an output is rejected.
ttl.dfb.network @split_invalid_source_dfb_id_is_output_net {
  // expected-error @below {{source DFB id 0 is also an output}}
  ttl.dfb.split @split 0 : index -> [0] policy = <round_robin>
}
// -----

// Test that a fork with a duplicated output is rejected.
ttl.dfb.network @fork_invalid_duplicate_output_net {
  // expected-error @below {{output 1 is duplicated}}
  ttl.dfb.fork @fork 0 : index -> [1, 1] storage = <auto>
}

// -----

// Test that a fork whose source is also an output is rejected.
ttl.dfb.network @fork_invalid_source_is_output_net {
  // expected-error @below {{source DFB id 0 is also an output}}
  ttl.dfb.fork @fork 0 : index -> [0, 1] storage = <auto>
}

// -----

// Test that a fork with a negative output is rejected.
ttl.dfb.network @fork_invalid_negative_output_net {
  // expected-error @below {{'outputs' failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  ttl.dfb.fork @fork 0 : index -> [-1] storage = <auto>
}

// -----

// Test that a fork with a negative source is rejected.
ttl.dfb.network @fork_invalid_negative_source_net {
  // expected-error @below {{'source' failed to satisfy constraint: non-negative DFB id or merged read handle}}
  ttl.dfb.fork @fork -1 : index -> [1] storage = <auto>
}

// -----

// Test that a fork outside a network is rejected.
// expected-error @below {{'ttl.dfb.fork' op expects parent op 'ttl.dfb.network'}}
ttl.dfb.fork @fork 0 : index -> [1] storage = <auto>

// -----

// Test that a split with no outputs is rejected.
ttl.dfb.network @split_invalid_no_outputs_net {
  // expected-error @below {{requires at least one output}}
  ttl.dfb.split @split 0 : index -> [] policy = <round_robin>
}

// -----

// Test that a split with a duplicated output is rejected.
ttl.dfb.network @split_invalid_duplicate_output_net {
  // expected-error @below {{output 1 is duplicated}}
  ttl.dfb.split @split 0 : index -> [1, 1] policy = <round_robin>
}

// -----

// Test that a split whose source is also an output is rejected.
ttl.dfb.network @split_invalid_source_is_output_net {
  // expected-error @below {{source DFB id 0 is also an output}}
  ttl.dfb.split @split 0 : index -> [0, 1] policy = <round_robin>
}

// -----

// Test that a split with a negative output is rejected.
ttl.dfb.network @split_invalid_negative_output_net {
  // expected-error @below {{'outputs' failed to satisfy constraint: i64 dense array attribute whose value is non-negative}}
  ttl.dfb.split @split 0 : index -> [-1] policy = <round_robin>
}

// -----

// Test that a split with a negative source is rejected.
ttl.dfb.network @split_invalid_negative_source_net {
  // expected-error @below {{'source' failed to satisfy constraint: non-negative DFB id or merged read handle}}
  ttl.dfb.split @split -1 : index -> [1] policy = <round_robin>
}

// -----

// Test that a merge with no inputs is rejected.
ttl.dfb.network @merge_invalid_no_inputs_net {
  // expected-error @below {{requires at least one input}}
  ttl.dfb.merge @merge [] policy = <round_robin>
}

// -----

// Test that a merge with a duplicated DFB id input is rejected.
ttl.dfb.network @merge_invalid_duplicate_dfb_id_net {
  // expected-error @below {{input 1 : index is duplicated}}
  ttl.dfb.merge @merge [1 : index, 1 : index] policy = <round_robin>
}

// -----

// Test that a merge with a duplicated merged-handle input is rejected.
ttl.dfb.network @merge_invalid_duplicate_symbol_net {
  ttl.dfb.merge @m1 [1 : index, 2 : index] policy = <round_robin>
  // expected-error @below {{input @m1 is duplicated}}
  ttl.dfb.merge @merge [@m1, @m1] policy = <round_robin>
}

// -----

// Test that a merge with a negative DFB id input is rejected.
ttl.dfb.network @merge_invalid_negative_input_net {
  // expected-error @below {{'inputs' failed to satisfy constraint: non-negative DFB ids or merged read handles}}
  ttl.dfb.merge @merge [-1 : index, 1 : index] policy = <round_robin>
}
