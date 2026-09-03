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
// Test: compute tensor access cannot address DRAM as core-local storage.
#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>
func.func @tensor_arg_in_compute(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  // expected-error @below {{'ttl.opaque_call' op tensor operand 0 in a compute kernel uses DRAM storage; compute tensor accessors require sharded SRAM (L1 or L1Small buffer type)}}
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

// Row-major tensor access accepts only scalar dtypes exposed by TTNN.
// expected-error @below {{layout element type must be a ttcore tile or one of f32, bf16, si32, ui32, ui16, or ui8, got 'f16'}}
#layout = #ttl.layout<shape = [1, 32], element_type = f16,
                      buffer = dram, grid = [1, 1], memory = interleaved>
func.func @unsupported_row_major_dtype(%arg0: tensor<1x32xf16, #layout>)
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "foo" (%arg0) {header = "h.hpp"} : (tensor<1x32xf16, #layout>) -> ()
  return
}

// -----

// Test: compute tensor access requires sharded local memory.
#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = l1, grid = [1, 1], memory = interleaved>
func.func @interleaved_compute_tensor(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  // expected-error @below {{'ttl.opaque_call' op tensor operand 0 in a compute kernel uses an unsupported memory layout; compute tensor accessors require height-, width-, or block-sharded SRAM}}
  ttl.opaque_call "foo" (%arg0) {header = "h.hpp"} : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> ()
  return
}

// -----

// Test: ND sharding uses distributed TensorAccessor metadata, not a local bank base.
#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = l1, grid = [1, 1], memory = nd_sharded>
func.func @nd_sharded_compute_tensor(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  // expected-error @below {{'ttl.opaque_call' op tensor operand 0 in a compute kernel uses an unsupported memory layout; compute tensor accessors require height-, width-, or block-sharded SRAM}}
  ttl.opaque_call "foo" (%arg0) {header = "h.hpp"} : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> ()
  return
}

// -----

// Test: data-movement tensor access requires device storage.
#layout = #ttl.layout<shape = [1, 32], element_type = bf16,
                      buffer = system_memory, grid = [1, 1], memory = interleaved>
func.func @host_data_movement_tensor(%arg0: tensor<1x32xbf16, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  // expected-error @below {{'ttl.opaque_call' op tensor operand 0 in a data movement kernel uses SystemMemory storage; data movement tensor accessors require device DRAM or SRAM}}
  ttl.opaque_call "foo" (%arg0) {header = "h.hpp"} : (tensor<1x32xbf16, #layout>) -> ()
  return
}

// -----

// Test: the L1Small SRAM buffer type requires sharded storage.
#layout = #ttl.layout<shape = [1, 32], element_type = ui8,
                      buffer = l1_small, grid = [1, 1], memory = interleaved>
func.func @interleaved_l1_small_data_movement_tensor(%arg0: tensor<1x32xui8, #layout>) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  // expected-error @below {{'ttl.opaque_call' op tensor operand 0 in a data movement kernel uses non-sharded SRAM with L1Small buffer type; L1Small requires sharded storage}}
  ttl.opaque_call "foo" (%arg0) {header = "h.hpp"} : (tensor<1x32xui8, #layout>) -> ()
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

// -----
// Test: a non-transactional access must select an existing dependency.
func.func @access_dependency_out_of_range() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  // expected-error @below {{'ttl.opaque_call' op DFB non-transactional access 0 dependency index 1 is out of range for 1 dependencies}}
  ttl.opaque_call "foo" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 1>] () {header = "h.hpp"} : () -> ()
  return
}

// -----
// Test: one dependency occurrence has one non-transactional summary.
func.func @duplicate_non_transactional_access() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  // expected-error @below {{'ttl.opaque_call' op DFB dependency 0 has more than one non-transactional access summary}}
  ttl.opaque_call "foo" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>, #ttl.dfb_non_transactional_access<inspect, 0>] () {header = "h.hpp"} : () -> ()
  return
}

// -----
// Test: one dependency occurrence cannot mix queue and non-transactional contracts.
func.func @protocol_and_non_transactional_access() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  // expected-error @below {{'ttl.opaque_call' op DFB dependency 0 cannot declare both protocol effects and a non-transactional access}}
  ttl.opaque_call "foo" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 1>] dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>] () {header = "h.hpp"} : () -> ()
  return
}

// -----
// Test: an external call produces at most one scalar result.
func.func @multiple_results() {
  // expected-error @below {{'ttl.opaque_call' op result group starting at #0 requires 0 or 1 element, but found 2}}
  %results:2 = ttl.opaque_call "foo" () {header = "h.hpp"} : () -> (i32, i64)
  return
}

// -----
// Test: an external call result uses a declared signless integer carrier.
func.func @unsupported_result_type() {
  // expected-error @below {{'ttl.opaque_call' op result #0 must be 32-bit signless integer or 64-bit signless integer, but got 'f32'}}
  %result = ttl.opaque_call "foo" () {header = "h.hpp"} : () -> f32
  return
}

// -----
// Test: a dispatch condition ordinal identifies one module-local declaration.
func.func @negative_dispatch_condition_ordinal() {
  // expected-error @below {{dispatch condition ordinal must be nonnegative}}
  %result = ttl.opaque_call "foo" () {condition_result = #ttl.dispatch_condition<-1, i64>, header = "h.hpp"} : () -> i64
  return
}

// -----
// Test: a dispatch condition uses one supported external scalar carrier.
func.func @unsupported_dispatch_condition_type() {
  // expected-error @below {{dispatch condition scalar type must be signless i32 or i64}}
  %result = ttl.opaque_call "foo" () {condition_result = #ttl.dispatch_condition<0, i16>, header = "h.hpp"} : () -> i16
  return
}

// -----
// Test: a dispatch condition declaration requires a scalar result.
func.func @dispatch_condition_without_result() {
  // expected-error @below {{'ttl.opaque_call' op condition result requires one scalar result}}
  ttl.opaque_call "foo" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "h.hpp"} : () -> ()
  return
}

// -----
// Test: the result type is part of the dispatch condition declaration.
func.func @dispatch_condition_result_type_mismatch() {
  // expected-error @below {{'ttl.opaque_call' op condition result type 'i32' does not match declared scalar type 'i64'}}
  %result = ttl.opaque_call "foo" () {condition_result = #ttl.dispatch_condition<0, i64>, header = "h.hpp"} : () -> i32
  return
}

// -----
// Test: stable condition evaluation cannot access DFB state.
func.func @stateful_dispatch_condition() {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  // expected-error @below {{'ttl.opaque_call' op condition result call cannot access DFB state}}
  %result = ttl.opaque_call "foo" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) () {condition_result = #ttl.dispatch_condition<0, i64>, header = "h.hpp"} : () -> i64
  return
}

// -----
// Test: an index template argument is still a DFB operand.
func.func @dispatch_condition_with_dfb_index() {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  // expected-error @below {{'ttl.opaque_call' op condition result call cannot access DFB state}}
  %result = ttl.opaque_call "foo" template_args [#ttl.external_template_arg<dfb_index, 0>] template_dfbs(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) () {condition_result = #ttl.dispatch_condition<0, i64>, header = "h.hpp"} : () -> i64
  return
}
