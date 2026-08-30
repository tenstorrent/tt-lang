// RUN: ttlang-opt %s -ttkernel-annotate-dfb-use --verify-diagnostics

// An opaque call cannot reference a DFB outside the enclosing kernel's
// compile-time DFB argument range.
func.func @out_of_range_opaque_dfb() attributes {
    ttl.base_cta_index = 2 : i32,
    ttkernel.thread = #ttkernel.thread<compute>} {
  // expected-error @below {{'ttkernel.opaque_call' op DFB descriptor index 2 is outside [0, 1] for the enclosing function}}
  ttkernel.opaque_call "describe" () {dfb_descriptor_indices = array<i32: 2>, header = "describe.hpp"} : () -> ()
  return
}
