// Verify ttl.opaque_call preserves ordered static arguments and DFB identity.
// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// A void call omits both optional template segments.
// CHECK-LABEL: func.func @void_no_template_args
// CHECK: ttl.opaque_call "my_func" () {header = "my_header.hpp"} : () -> ()
func.func @void_no_template_args() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "my_func" () {header = "my_header.hpp"} : () -> ()
  return
}

// -----

// Scalar template arguments are attributes rather than artificial SSA values.
// CHECK-LABEL: func.func @scalar_template_args
// CHECK: ttl.opaque_call "add_vals" template_args [#ttl.external_template_arg<signed_integer, 1>, #ttl.external_template_arg<boolean, 1>, #ttl.external_template_arg<unsigned_integer, 3212836864>] () {header = "math.hpp"} : () -> ()
func.func @scalar_template_args() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "add_vals" template_args [#ttl.external_template_arg<signed_integer, 1>, #ttl.external_template_arg<boolean, 1>, #ttl.external_template_arg<unsigned_integer, 3212836864>] () {header = "math.hpp"} : () -> ()
  return
}

// -----

// Function operands remain independent from the static argument list.
// CHECK-LABEL: func.func @func_and_template_args
// CHECK: ttl.opaque_call "process" template_args [#ttl.external_template_arg<signed_integer, 42>] (%arg0) {header = "proc.hpp"} : (i32) -> ()
func.func @func_and_template_args(%arg0: i32) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "process" template_args [#ttl.external_template_arg<signed_integer, 42>] (%arg0) {header = "proc.hpp"} : (i32) -> ()
  return
}

// -----

// Unsigned address semantics remain explicit at the external call boundary.
// CHECK-LABEL: func.func @unsigned_func_arg
// CHECK: ttl.opaque_call "use_address" (%arg0) {header = "address.hpp", unsigned_arg_indices = array<i32: 0>} : (i32) -> ()
func.func @unsigned_func_arg(%arg0: i32) attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "use_address" (%arg0) {header = "address.hpp", unsigned_arg_indices = array<i32: 0>} : (i32) -> ()
  return
}

// -----

// A DFB index entry references a typed DFB operand without declaring access.
// CHECK-LABEL: func.func @dfb_index_template_arg
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = 0, block_count = 2}
// CHECK-NEXT: ttl.opaque_call "inspect" template_args [#ttl.external_template_arg<dfb_index, 0>] template_dfbs(%[[CB]] : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) (%[[CB]]) {header = "inspect.hpp"}
func.func @dfb_index_template_arg() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.opaque_call "inspect" template_args [#ttl.external_template_arg<dfb_index, 0>] template_dfbs(%cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) (%cb) {header = "inspect.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
  return
}

// -----

// A descriptor entry makes its referenced DFB a direct storage dependency.
// CHECK-LABEL: func.func @dfb_descriptor_template_arg
// CHECK: %[[CB:.*]] = ttl.bind_cb{cb_index = 3, block_count = 2}
// CHECK-NEXT: ttl.opaque_call "describe" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%[[CB]] : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>) () {header = "describe.hpp"}
func.func @dfb_descriptor_template_arg() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  ttl.opaque_call "describe" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%cb : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>) () {header = "describe.hpp"} : () -> ()
  return
}

// -----

// Dependency-only DFBs, typed effects, and unknown access round-trip without
// changing the function-argument segment.
// CHECK-LABEL: func.func @dfb_protocol_metadata
// CHECK: %[[ARG:.*]] = ttl.bind_cb
// CHECK-NEXT: %[[DESCRIPTOR:.*]] = ttl.bind_cb
// CHECK-NEXT: %[[DEPENDENCY:.*]] = ttl.bind_cb
// CHECK-NEXT: ttl.opaque_call "protocol" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%[[DESCRIPTOR]] : !ttl.cb<{{.*}}>) dfb_dependencies(%[[DEPENDENCY]] : !ttl.cb<{{.*}}>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 2>, #ttl.dfb_protocol_effect<reserve, 2, 1>, #ttl.dfb_protocol_effect<push, 2, 1>] (%[[ARG]]) {header = "protocol.hpp", unknown_dfb_access}
func.func @dfb_protocol_metadata() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %arg = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %descriptor = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %dependency = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.opaque_call "protocol" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%descriptor : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_dependencies(%dependency : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 2>, #ttl.dfb_protocol_effect<reserve, 2, 1>, #ttl.dfb_protocol_effect<push, 2, 1>] (%arg) {header = "protocol.hpp", unknown_dfb_access} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
  return
}

// -----

// A typed non-transactional access retains its dependency occurrence.
// CHECK-LABEL: func.func @dfb_inspect_access
// CHECK: %[[DESCRIPTOR:.*]] = ttl.bind_cb
// CHECK-NEXT: ttl.opaque_call "inspect" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%[[DESCRIPTOR]] : !ttl.cb<{{.*}}>) dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>] () {header = "inspect.hpp"}
func.func @dfb_inspect_access() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %descriptor = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  ttl.opaque_call "inspect" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%descriptor : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) dfb_accesses [#ttl.dfb_non_transactional_access<inspect, 0>] () {header = "inspect.hpp"} : () -> ()
  return
}

// -----

// Dependency occurrence indices remain valid when operand adaptation maps a
// dependency-only operand to the same DFB as a function argument.
// CHECK-LABEL: func.func @adapted_dependency_occurrences
// CHECK: %[[DFB:.*]] = ttl.bind_cb
// CHECK-NEXT: ttl.opaque_call "adapted" dfb_dependencies(%[[DFB]] : !ttl.cb<{{.*}}>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 1, 1>] (%[[DFB]])
func.func @adapted_dependency_occurrences() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  ttl.opaque_call "adapted" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 1, 1>] (%dfb) {header = "adapted.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) -> ()
  return
}

// -----

// Scalar results are restricted to the frontend's declared carrier types.
// CHECK-LABEL: func.func @scalar_results
// CHECK: %[[I32:.*]] = ttl.opaque_call "result_i32" () {header = "result.hpp"} : () -> i32
// CHECK-NEXT: %[[I64:.*]] = ttl.opaque_call "result_i64" () {header = "result.hpp"} : () -> i64
func.func @scalar_results() {
  %i32 = ttl.opaque_call "result_i32" () {header = "result.hpp"} : () -> i32
  %i64 = ttl.opaque_call "result_i64" () {header = "result.hpp"} : () -> i64
  return
}
