// RUN: ttlang-opt %s -ttkernel-annotate-dfb-use --verify-diagnostics --split-input-file

// Verify that DFB resource metadata cannot reference descriptors outside the
// finalized module allocation range.
module attributes {ttl.dfb_allocations = [{}, {}]} {
  func.func @invalid_resource_index() attributes {
      ttl.base_cta_index = 2 : i32,
      ttkernel.thread = #ttkernel.thread<noc>} {
    // expected-error @below {{'ttkernel.opaque_call' op DFB resource index 2 is outside the enclosing function's DFB range [0, 2)}}
    ttkernel.opaque_call "inspect"() {dfb_resource_indices = array<i32: 2>, header = "inspect.hpp"} : () -> ()
    return
  }
}

// -----

// expected-error @below {{`ttkernel-annotate-dfb-use` requires finalized DFB allocation metadata; run `ttl-finalize-dfb-indices` first}}
module {
}

// -----

// expected-error @below {{`ttkernel-annotate-dfb-use` requires finalized DFB allocation metadata; run `ttl-finalize-dfb-indices` first}}
module attributes {ttl.dfb_allocations = 0 : i64} {
}
