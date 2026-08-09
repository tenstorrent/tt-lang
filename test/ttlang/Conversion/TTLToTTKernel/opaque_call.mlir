// Verify opaque_call lowering from TTL to TTKernel.
// Checks template arg forwarding, DFB-to-CB conversion for func_args,
// get_dfb_id lowering, tensor func-arg TensorAccessor materialization,
// and ttl.raw_addr lowering.
// RUN: ttlang-opt --convert-ttl-to-ttkernel --split-input-file %s | FileCheck %s

// Void call with no args lowers directly.
// CHECK-LABEL: func.func @void_call_no_args
// CHECK: ttkernel.opaque_call "noop"() {header = "noop.hpp"} : () -> ()
func.func @void_call_no_args() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "noop" () {header = "noop.hpp"} : () -> ()
  return
}

// -----

// Constant template kinds become ordered static TTKernel attributes.
// CHECK-LABEL: func.func @call_with_template_args
// CHECK: ttkernel.opaque_call "compute" template_args [-3 : si32, true, 4294967295 : ui32]() {header = "compute.hpp"} : () -> ()
func.func @call_with_template_args() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "compute" template_args [#ttl.external_template_arg<signed_integer, -3>, #ttl.external_template_arg<boolean, 1>, #ttl.external_template_arg<unsigned_integer, 4294967295>] () {header = "compute.hpp"} : () -> ()
  return
}

// -----

// A finalized DFB index becomes an unsigned static argument.
// CHECK-LABEL: func.func @call_with_dfb_template_arg
// CHECK-NOT: ttkernel.get_dfb_id
// CHECK: ttkernel.opaque_call "drain" template_args [2 : ui32]() {header = "drain.hpp"} : () -> ()
func.func @call_with_dfb_template_arg() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  ttl.opaque_call "drain" template_args [#ttl.external_template_arg<dfb_index, 0>] template_dfbs(%cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) dfb_dependencies(%cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) () {header = "drain.hpp"} : () -> ()
  return
}

// -----

// A DFB template operand preserves its finalized index and allocation geometry.
// CHECK-LABEL: func.func @call_with_dfb_descriptor
// CHECK: ttkernel.opaque_call "describe" template_args [#ttkernel.dfb_descriptor<2, 6, 4, 4096>]() {header = "describe.hpp"} : () -> ()
func.func @call_with_dfb_descriptor() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 2, block_count = 4} : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 4>
  ttl.opaque_call "describe" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%cb : !ttl.cb<[2, 3], !ttcore.tile<32x32, f32>, 4>) () {header = "describe.hpp"} : () -> ()
  return
}

// -----

// Subtile dimensions change bytes per page, not the number of pages in a block.
// CHECK-LABEL: func.func @call_with_subtile_dfb_descriptor
// CHECK: ttkernel.opaque_call "describe" template_args [#ttkernel.dfb_descriptor<2, 6, 4, 32>]() {header = "describe.hpp"} : () -> ()
func.func @call_with_subtile_dfb_descriptor() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 2, block_count = 4} : !ttl.cb<[2, 3], !ttcore.tile<1x16, bf16>, 4>
  ttl.opaque_call "describe" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%cb : !ttl.cb<[2, 3], !ttcore.tile<1x16, bf16>, 4>) () {header = "describe.hpp"} : () -> ()
  return
}

// -----

// Scalar DFBs use one scalar element per page rather than tile storage size.
// CHECK-LABEL: func.func @call_with_scalar_dfb_descriptor
// CHECK: ttkernel.opaque_call "describe" template_args [#ttkernel.dfb_descriptor<2, 6, 4, 4>]() {header = "describe.hpp"} : () -> ()
func.func @call_with_scalar_dfb_descriptor() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 2, block_count = 4} : !ttl.cb<[2, 3], f32, 4>
  ttl.opaque_call "describe" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%cb : !ttl.cb<[2, 3], f32, 4>) () {header = "describe.hpp"} : () -> ()
  return
}

// -----

// DFB func_arg is lowered to an unsigned physical index.
// CHECK-LABEL: func.func @call_with_dfb_func_arg
// CHECK: %[[CB_IDX:.*]] = ttkernel.get_compile_time_arg_val(1) : () -> ui32
// CHECK-NEXT: ttkernel.opaque_call "use_cb"(%[[CB_IDX]]) {header = "use_cb.hpp"} : (ui32) -> ()
func.func @call_with_dfb_func_arg() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.opaque_call "use_cb" (%cb) {header = "use_cb.hpp"} : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
  return
}

// -----

#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

// Bare tensor func_arg is lowered to a TensorAccessor.
// CHECK-LABEL: func.func @call_with_tensor_func_arg
// CHECK-DAG: %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[BANK_BASE:.*]] = ttkernel.get_common_arg_val(%[[C0_IDX]]) : (index) -> i32
// CHECK-DAG: %[[ACC_ARGS:.*]] = ttkernel.TensorAccessorArgs(
// CHECK-DAG: %[[ACC:.*]] = ttkernel.TensorAccessor(%[[ACC_ARGS]], %[[BANK_BASE]], {{.*}}) : (!ttkernel.TensorAccessorArgs, i32, i32) -> !ttkernel.TensorAccessor
// CHECK: ttkernel.opaque_call "use_tensor"(%[[ACC]]) {header = "use_tensor.hpp"} : (!ttkernel.TensorAccessor) -> ()
func.func @call_with_tensor_func_arg(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "use_tensor" (%arg0) {header = "use_tensor.hpp"} : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> ()
  return
}

// -----

#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

// ttl.raw_addr lowers to get_common_arg_val and forwards an i32 arg.
// CHECK-LABEL: func.func @call_with_raw_addr_func_arg
// CHECK-DAG: %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ADDR:.*]] = ttkernel.get_common_arg_val(%[[C0_IDX]]) : (index) -> i32
// CHECK: ttkernel.opaque_call "use_addr"(%[[ADDR]]) {header = "use_addr.hpp", unsigned_arg_indices = array<i32: 0>} : (i32) -> ()
func.func @call_with_raw_addr_func_arg(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
  %addr = ttl.raw_addr %arg0 : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> i32
  ttl.opaque_call "use_addr" (%addr) {header = "use_addr.hpp", unsigned_arg_indices = array<i32: 0>} : (i32) -> ()
  return
}

// -----

#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
                      buffer = dram, grid = [1, 1], memory = interleaved>

// Raw addresses need no TensorAccessor, so compute kernels can use their tensor runtime argument.
// CHECK-LABEL: func.func @call_with_compute_raw_addr_func_arg
// CHECK-DAG: %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[ADDR:.*]] = ttkernel.get_common_arg_val(%[[C0_IDX]]) : (index) -> i32
// CHECK: ttkernel.opaque_call "use_addr"(%[[ADDR]]) {header = "use_addr.hpp", unsigned_arg_indices = array<i32: 0>} : (i32) -> ()
func.func @call_with_compute_raw_addr_func_arg(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<compute>} {
  %addr = ttl.raw_addr %arg0 : tensor<1x1x!ttcore.tile<32x32, f32>, #layout> -> i32
  ttl.opaque_call "use_addr" (%addr) {header = "use_addr.hpp", unsigned_arg_indices = array<i32: 0>} : (i32) -> ()
  return
}

// -----

// Dependency and protocol metadata describe external behavior without adding
// TTKernel call arguments or protocol operations.
// CHECK-LABEL: func.func @dfb_protocol_metadata
// CHECK-NOT: ttkernel.cb_
// CHECK: ttkernel.opaque_call "hidden_protocol"() {header = "hidden_protocol.hpp"} : () -> ()
// CHECK-NOT: ttkernel.cb_
func.func @dfb_protocol_metadata() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %dfb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.opaque_call "hidden_protocol" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>] () {header = "hidden_protocol.hpp", unknown_dfb_access} : () -> ()
  return
}
