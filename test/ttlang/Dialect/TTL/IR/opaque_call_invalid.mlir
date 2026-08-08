// Verify that opaque_call rejects invalid inputs.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// -----
// Test: empty callee name
func.func @empty_callee() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  // expected-error @below {{'ttl.opaque_call' op callee name must not be empty}}
  ttl.opaque_call "" () {header = "h.hpp"} : () -> ()
  return
}

// -----
// Test: empty header path
func.func @empty_header() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  // expected-error @below {{'ttl.opaque_call' op header path must not be empty}}
  ttl.opaque_call "foo" () {header = ""} : () -> ()
  return
}

// -----
// Test: unsupported attribute in the ordered template argument list.
func.func @unsupported_template_arg() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  // expected-error @below {{'ttl.opaque_call' op template argument list must contain only #ttl.external_template_arg attributes}}
  ttl.opaque_call "foo" template_args [42 : i32] () {header = "h.hpp"} : () -> ()
  return
}

// -----
// Test: a DFB reference must identify an available template DFB operand.
func.func @out_of_range_template_dfb() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  // expected-error @below {{'ttl.opaque_call' op template DFB operand index 0 is out of range for 0 operands}}
  ttl.opaque_call "foo" template_args [#ttl.external_template_arg<dfb_index, 0>] () {header = "h.hpp"} : () -> ()
  return
}

// -----
// Test: every template DFB operand must be used by the ordered list.
func.func @unreferenced_template_dfb() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  // expected-error @below {{'ttl.opaque_call' op every template DFB operand must be referenced by an ordered template argument}}
  ttl.opaque_call "foo" template_args [#ttl.external_template_arg<signed_integer, 1>] template_dfbs(%cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) () {header = "h.hpp"} : () -> ()
  return
}

// -----
// Test: unsigned argument indices must refer to function operands.
func.func @unsigned_arg_out_of_range(%arg0: i32) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  // expected-error @below {{'ttl.opaque_call' op unsigned function argument index 1 is out of range for 1 arguments}}
  ttl.opaque_call "foo" (%arg0) {header = "h.hpp", unsigned_arg_indices = array<i32: 1>} : (i32) -> ()
  return
}

// -----
// Test: unsigned argument indices have one canonical order without duplicates.
func.func @unsigned_arg_indices_not_increasing(%arg0: i32, %arg1: i32) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  // expected-error @below {{'ttl.opaque_call' op unsigned function argument indices must be strictly increasing}}
  ttl.opaque_call "foo" (%arg0, %arg1) {header = "h.hpp", unsigned_arg_indices = array<i32: 1, 0>} : (i32, i32) -> ()
  return
}

// -----
// Test: unsigned coercion is defined only for 32-bit integer operands.
func.func @unsigned_arg_not_integer(%arg0: f32) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  // expected-error @below {{'ttl.opaque_call' op unsigned function argument index 0 must reference a 32-bit integer operand, got 'f32'}}
  ttl.opaque_call "foo" (%arg0) {header = "h.hpp", unsigned_arg_indices = array<i32: 0>} : (f32) -> ()
  return
}

// -----
// Test: compute kernels do not receive TensorAccessor compile-time arguments.
#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>
func.func @tensor_arg_in_compute(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  // expected-error @below {{'ttl.opaque_call' op tensor function arguments require a data movement (noc) thread}}
  ttl.opaque_call "foo" (%arg0) {header = "h.hpp"} : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> ()
  return
}

// -----
// Test: a derived tensor has no common-runtime-argument mapping.
#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>
func.func @derived_tensor_arg(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %derived = tensor.cast %arg0 : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> to tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
  // expected-error @below {{'ttl.opaque_call' op tensor operands must be arguments of the enclosing kernel function with TTL layout encoding; slices/views are not supported}}
  ttl.opaque_call "foo" (%derived) {header = "h.hpp"} : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> ()
  return
}

// -----
// Test: a tensor without TTL layout cannot be mapped to TensorAccessor metadata.
func.func @tensor_arg_without_layout(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  // expected-error @below {{'ttl.opaque_call' op tensor operands must be arguments of the enclosing kernel function with TTL layout encoding; slices/views are not supported}}
  ttl.opaque_call "foo" (%arg0) {header = "h.hpp"} : (tensor<1x1x!ttcore.tile<32x32, f32>>) -> ()
  return
}

// -----
// Test: TensorAccessor support is limited to the documented dtype contract.
#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f16>,
                      buffer = dram, grid = [1, 1], memory = interleaved>
func.func @unsupported_tensor_accessor_dtype(%arg0: tensor<1x1x!ttcore.tile<32x32, f16>, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  // expected-error @below {{'ttl.opaque_call' op TensorAccessor operands support only bf16 and f32 tile types}}
  ttl.opaque_call "foo" (%arg0) {header = "h.hpp"} : (tensor<1x1x!ttcore.tile<32x32, f16>, #layout>) -> ()
  return
}

// -----
// Test: a protocol effect must select an existing dependency occurrence.
func.func @effect_dependency_out_of_range() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  // expected-error @below {{'ttl.opaque_call' op DFB protocol effect 0 dependency index 1 is out of range for 1 dependencies}}
  ttl.opaque_call "foo" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 1, 1>] () {header = "h.hpp"} : () -> ()
  return
}

// -----
// Test: a protocol effect dependency index must be nonnegative.
func.func @effect_negative_dependency_index() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  // expected-error @below {{DFB dependency index must be nonnegative, got -1}}
  ttl.opaque_call "foo" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) dfb_effects [#ttl.dfb_protocol_effect<reserve, -1, 1>] () {header = "h.hpp"} : () -> ()
  return
}

// -----
// Test: a protocol transaction count must be positive.
func.func @effect_nonpositive_tile_count() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  // expected-error @below {{DFB protocol tile count must be positive, got 0}}
  ttl.opaque_call "foo" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 0>] () {header = "h.hpp"} : () -> ()
  return
}

// -----
// Test: one protocol action cannot exceed the dependency's physical capacity.
func.func @effect_tile_count_exceeds_capacity() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  // expected-error @below {{'ttl.opaque_call' op DFB protocol effect 0 tile count 2 exceeds dependency 0 capacity 1}}
  ttl.opaque_call "foo" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 2>] () {header = "h.hpp"} : () -> ()
  return
}
