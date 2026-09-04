// Verify that DFB resource metadata cannot reference descriptors outside the
// enclosing function's compile-time DFB argument range.
// RUN: ttlang-opt %s -ttkernel-annotate-dfb-use --verify-diagnostics --split-input-file

module {
  func.func @invalid_resource_index() attributes {
      ttl.base_cta_index = 2 : i32,
      ttkernel.thread = #ttkernel.thread<noc>} {
    // expected-error @below {{'ttkernel.opaque_call' op DFB resource index 2 is outside the enclosing function's DFB range [0, 2)}}
    ttkernel.opaque_call "inspect"() {dfb_resource_indices = array<i32: 2>, header = "inspect.hpp"} : () -> ()
    return
  }
}
