// Verify ttkernel.opaque_call rejects malformed static arguments.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// An external call requires a callee name.
func.func @empty_callee() {
  // expected-error @below {{'ttkernel.opaque_call' op callee name must not be empty}}
  ttkernel.opaque_call "" () {header = "h.hpp"} : () -> ()
  return
}

// -----

// An external call requires the declaring header.
func.func @empty_header() {
  // expected-error @below {{'ttkernel.opaque_call' op header path must not be empty}}
  ttkernel.opaque_call "foo" () {header = ""} : () -> ()
  return
}

// -----

// A signless integer has no defined signed or unsigned C++ literal spelling.
func.func @signless_integer_template_arg() {
  // expected-error @below {{'ttkernel.opaque_call' op integer template arg must have type si32 or ui32}}
  ttkernel.opaque_call "foo" template_args [5 : i32] () {header = "h.hpp"} : () -> ()
  return
}

// -----

// Wider integer values are outside the initial external template contract.
func.func @wide_integer_template_arg() {
  // expected-error @below {{'ttkernel.opaque_call' op integer template arg must have type si32 or ui32}}
  ttkernel.opaque_call "foo" template_args [5 : si64] () {header = "h.hpp"} : () -> ()
  return
}

// -----

// Index-typed IntegerAttr values are outside the C++ integer contract.
func.func @index_integer_template_arg() {
  // expected-error @below {{'ttkernel.opaque_call' op integer template arg must have type si32 or ui32}}
  ttkernel.opaque_call "foo" template_args [0 : index] () {header = "h.hpp"} : () -> ()
  return
}

// -----

// Only typed static argument attributes may appear in the ordered list.
func.func @unsupported_template_arg() {
  // expected-error @below {{'ttkernel.opaque_call' op template arg must be a signed i32, boolean, unsigned i32, or DFB descriptor attribute}}
  ttkernel.opaque_call "foo" template_args ["unsupported"] () {header = "h.hpp"} : () -> ()
  return
}

// -----

// Descriptor fields must fit the generated uint32_t template parameters.
func.func @descriptor_field_out_of_range() {
  // expected-error @below {{page_size_bytes must be positive and representable as uint32_t, got 4294967296}}
  ttkernel.opaque_call "foo" template_args [#ttkernel.dfb_descriptor<0, 1, 1, 4294967296>] () {header = "h.hpp"} : () -> ()
  return
}

// -----

// Unsigned argument indices must refer to function operands.
func.func @unsigned_arg_out_of_range(%arg0: i32) {
  // expected-error @below {{'ttkernel.opaque_call' op unsigned function argument index 1 is out of range for 1 arguments}}
  ttkernel.opaque_call "foo" (%arg0) {header = "h.hpp", unsigned_arg_indices = array<i32: 1>} : (i32) -> ()
  return
}

// -----

// Unsigned argument indices have one canonical order without duplicates.
func.func @unsigned_arg_indices_not_increasing(%arg0: i32, %arg1: i32) {
  // expected-error @below {{'ttkernel.opaque_call' op unsigned function argument indices must be strictly increasing}}
  ttkernel.opaque_call "foo" (%arg0, %arg1) {header = "h.hpp", unsigned_arg_indices = array<i32: 1, 0>} : (i32, i32) -> ()
  return
}

// -----

// Unsigned coercion is defined only for 32-bit integer operands.
func.func @unsigned_arg_not_integer(%arg0: f32) {
  // expected-error @below {{'ttkernel.opaque_call' op unsigned function argument index 0 must reference a 32-bit integer operand, got 'f32'}}
  ttkernel.opaque_call "foo" (%arg0) {header = "h.hpp", unsigned_arg_indices = array<i32: 0>} : (f32) -> ()
  return
}
