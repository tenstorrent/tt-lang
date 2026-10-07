// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Summary: Test invalid ttl.dfb.network and DFB network enum attributes syntax.

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
