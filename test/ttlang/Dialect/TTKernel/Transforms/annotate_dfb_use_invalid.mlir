// Verify that DFB resource metadata cannot reference descriptors outside the
// enclosing function's compile-time DFB argument range.
// RUN: ttlang-opt %s -ttkernel-annotate-dfb-use --verify-diagnostics --split-input-file

// Compiler-owned L1 requires finalized allocation metadata.
// expected-error @below {{compiler-l1 requires finalized allocation metadata}}
module attributes {ttl.memory_model = "compiler-l1"} {
}

// -----

// Allocation metadata must use the finalized array representation.
// expected-error @below {{compiler-l1 requires finalized allocation metadata}}
module attributes {ttl.memory_model = "compiler-l1", ttl.dfb_allocations = 0 : i64} {
}

// -----

module {
  func.func @invalid_resource_index() attributes {
      ttl.base_cta_index = 2 : i32,
      ttkernel.thread = #ttkernel.thread<noc>} {
    // expected-error @below {{'ttkernel.opaque_call' op DFB resource index 2 is outside the enclosing function's DFB range [0, 2)}}
    ttkernel.opaque_call "inspect"() {dfb_resource_indices = array<i32: 2>, header = "inspect.hpp"} : () -> ()
    return
  }
}
