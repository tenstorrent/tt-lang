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

#row_f32 = #ttl.layout<shape = [1, 32], element_type = f32, buffer = dram, grid = [1, 1], memory = interleaved>
#row_bf16 = #ttl.layout<shape = [1, 32], element_type = bf16, buffer = dram, grid = [1, 1], memory = interleaved>
#row_si32 = #ttl.layout<shape = [1, 32], element_type = si32, buffer = dram, grid = [1, 1], memory = interleaved>
#row_ui32 = #ttl.layout<shape = [1, 32], element_type = ui32, buffer = dram, grid = [1, 1], memory = interleaved>
#row_ui16 = #ttl.layout<shape = [1, 32], element_type = ui16, buffer = dram, grid = [1, 1], memory = interleaved>
#row_ui8 = #ttl.layout<shape = [1, 32], element_type = ui8, buffer = dram, grid = [1, 1], memory = interleaved>

// Every supported row-major scalar type forms a descriptor-based accessor.
// CHECK-LABEL: func.func @call_with_every_row_major_dtype
// CHECK-COUNT-6: = ttkernel.TensorAccessor({{.*}}) : (!ttkernel.TensorAccessorArgs, i32) -> !ttkernel.TensorAccessor
// CHECK: ttkernel.opaque_call "use_all_row_major_dtypes"
func.func @call_with_every_row_major_dtype(
    %arg0: tensor<1x32xf32, #row_f32>,
    %arg1: tensor<1x32xbf16, #row_bf16>,
    %arg2: tensor<1x32xsi32, #row_si32>,
    %arg3: tensor<1x32xui32, #row_ui32>,
    %arg4: tensor<1x32xui16, #row_ui16>,
    %arg5: tensor<1x32xui8, #row_ui8>)
    attributes {ttl.base_cta_index = 0 : i32,
                ttl.crta_indices = [0, 1, 2, 3, 4, 5],
                ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "use_all_row_major_dtypes" (%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) {header = "use_all_row_major_dtypes.hpp"} :
      (tensor<1x32xf32, #row_f32>,
       tensor<1x32xbf16, #row_bf16>,
       tensor<1x32xsi32, #row_si32>,
       tensor<1x32xui32, #row_ui32>,
       tensor<1x32xui16, #row_ui16>,
       tensor<1x32xui8, #row_ui8>) -> ()
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
// CHECK-DAG: %[[ACC:.*]] = ttkernel.TensorAccessor(%[[ACC_ARGS]], %[[BANK_BASE]]) : (!ttkernel.TensorAccessorArgs, i32) -> !ttkernel.TensorAccessor
// CHECK: ttkernel.opaque_call "use_tensor"(%[[ACC]]) {header = "use_tensor.hpp"} : (!ttkernel.TensorAccessor) -> ()
func.func @call_with_tensor_func_arg(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "use_tensor" (%arg0) {header = "use_tensor.hpp"} : (tensor<1x1x!ttcore.tile<32x32, f32>, #layout>) -> ()
  return
}

// -----

#layout = #ttl.layout<shape = [1, 1], element_type = !ttcore.tile<32x32, bfp_bf8>,
                      buffer = dram, grid = [1, 1], memory = nd_sharded>

// ND-sharded device tensors use the distributed TensorAccessor interface.
// CHECK-LABEL: func.func @call_with_nd_sharded_tensor_func_arg
// CHECK-DAG: %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[BANK_BASE:.*]] = ttkernel.get_common_arg_val(%[[C0_IDX]]) : (index) -> i32
// CHECK-DAG: %[[ACC_ARGS:.*]] = ttkernel.TensorAccessorArgs(
// CHECK-DAG: %[[ACC:.*]] = ttkernel.TensorAccessor(%[[ACC_ARGS]], %[[BANK_BASE]]) : (!ttkernel.TensorAccessorArgs, i32) -> !ttkernel.TensorAccessor
// CHECK: ttkernel.opaque_call "use_tensor"(%[[ACC]]) {header = "use_tensor.hpp"} : (!ttkernel.TensorAccessor) -> ()
func.func @call_with_nd_sharded_tensor_func_arg(%arg0: tensor<1x1x!ttcore.tile<32x32, bfp_bf8>, #layout>) attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0], ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "use_tensor" (%arg0) {header = "use_tensor.hpp"} : (tensor<1x1x!ttcore.tile<32x32, bfp_bf8>, #layout>) -> ()
  return
}

// -----

#layout_f32 = #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, f32>, buffer = dram, grid = [1, 1], memory = interleaved>
#layout_f16 = #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, f16>, buffer = dram, grid = [1, 1], memory = interleaved>
#layout_bf16 = #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bf16>, buffer = dram, grid = [1, 1], memory = interleaved>
#layout_bfp_f8 = #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bfp_f8>, buffer = dram, grid = [1, 1], memory = interleaved>
#layout_bfp_bf8 = #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bfp_bf8>, buffer = dram, grid = [1, 1], memory = interleaved>
#layout_bfp_f4 = #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bfp_f4>, buffer = dram, grid = [1, 1], memory = interleaved>
#layout_bfp_bf4 = #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bfp_bf4>, buffer = dram, grid = [1, 1], memory = interleaved>
#layout_bfp_f2 = #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bfp_f2>, buffer = dram, grid = [1, 1], memory = interleaved>
#layout_bfp_bf2 = #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, bfp_bf2>, buffer = dram, grid = [1, 1], memory = interleaved>
#layout_u32 = #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, u32>, buffer = dram, grid = [1, 1], memory = interleaved>
#layout_u16 = #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, u16>, buffer = dram, grid = [1, 1], memory = interleaved>
#layout_u8 = #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, u8>, buffer = dram, grid = [1, 1], memory = interleaved>
#layout_si32 = #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, si32>, buffer = dram, grid = [1, 1], memory = interleaved>
#layout_i1 = #ttl.layout<shape = [32, 32], element_type = !ttcore.tile<32x32, i1>, buffer = dram, grid = [1, 1], memory = interleaved>

// Every ttcore tile dtype forms the same descriptor-based accessor.
// CHECK-LABEL: func.func @call_with_every_tile_dtype
// CHECK-COUNT-14: = ttkernel.TensorAccessor({{.*}}) : (!ttkernel.TensorAccessorArgs, i32) -> !ttkernel.TensorAccessor
// CHECK: ttkernel.opaque_call "use_all_tile_dtypes"
func.func @call_with_every_tile_dtype(
    %arg0: tensor<1x1x!ttcore.tile<32x32, f32>, #layout_f32>,
    %arg1: tensor<1x1x!ttcore.tile<32x32, f16>, #layout_f16>,
    %arg2: tensor<1x1x!ttcore.tile<32x32, bf16>, #layout_bf16>,
    %arg3: tensor<1x1x!ttcore.tile<32x32, bfp_f8>, #layout_bfp_f8>,
    %arg4: tensor<1x1x!ttcore.tile<32x32, bfp_bf8>, #layout_bfp_bf8>,
    %arg5: tensor<1x1x!ttcore.tile<32x32, bfp_f4>, #layout_bfp_f4>,
    %arg6: tensor<1x1x!ttcore.tile<32x32, bfp_bf4>, #layout_bfp_bf4>,
    %arg7: tensor<1x1x!ttcore.tile<32x32, bfp_f2>, #layout_bfp_f2>,
    %arg8: tensor<1x1x!ttcore.tile<32x32, bfp_bf2>, #layout_bfp_bf2>,
    %arg9: tensor<1x1x!ttcore.tile<32x32, u32>, #layout_u32>,
    %arg10: tensor<1x1x!ttcore.tile<32x32, u16>, #layout_u16>,
    %arg11: tensor<1x1x!ttcore.tile<32x32, u8>, #layout_u8>,
    %arg12: tensor<1x1x!ttcore.tile<32x32, si32>, #layout_si32>,
    %arg13: tensor<1x1x!ttcore.tile<32x32, i1>, #layout_i1>)
    attributes {ttl.base_cta_index = 0 : i32,
                ttl.crta_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "use_all_tile_dtypes" (%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9, %arg10, %arg11, %arg12, %arg13) {header = "use_all_tile_dtypes.hpp"} :
      (tensor<1x1x!ttcore.tile<32x32, f32>, #layout_f32>,
       tensor<1x1x!ttcore.tile<32x32, f16>, #layout_f16>,
       tensor<1x1x!ttcore.tile<32x32, bf16>, #layout_bf16>,
       tensor<1x1x!ttcore.tile<32x32, bfp_f8>, #layout_bfp_f8>,
       tensor<1x1x!ttcore.tile<32x32, bfp_bf8>, #layout_bfp_bf8>,
       tensor<1x1x!ttcore.tile<32x32, bfp_f4>, #layout_bfp_f4>,
       tensor<1x1x!ttcore.tile<32x32, bfp_bf4>, #layout_bfp_bf4>,
       tensor<1x1x!ttcore.tile<32x32, bfp_f2>, #layout_bfp_f2>,
       tensor<1x1x!ttcore.tile<32x32, bfp_bf2>, #layout_bfp_bf2>,
       tensor<1x1x!ttcore.tile<32x32, u32>, #layout_u32>,
       tensor<1x1x!ttcore.tile<32x32, u16>, #layout_u16>,
       tensor<1x1x!ttcore.tile<32x32, u8>, #layout_u8>,
       tensor<1x1x!ttcore.tile<32x32, si32>, #layout_si32>,
       tensor<1x1x!ttcore.tile<32x32, i1>, #layout_i1>) -> ()
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

#local_layout = #ttl.layout<shape = [1, 32], element_type = ui8,
                            buffer = l1_small, grid = [1, 1], memory = block_sharded>

// A compute call receives one shared local accessor for repeated tensor operands.
// CHECK-LABEL: func.func @call_with_compute_local_tensor
// CHECK-DAG: %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[BANK_BASE:.*]] = ttkernel.get_common_arg_val(%[[C0_IDX]]) : (index) -> i32
// CHECK: %[[LOCAL:.*]] = ttkernel.LocalTensorAccessor(%[[BANK_BASE]]) : (i32) -> !ttkernel.LocalTensorAccessor
// CHECK-NOT: ttkernel.LocalTensorAccessor
// CHECK: ttkernel.opaque_call "use_local_pair"(%[[LOCAL]], %[[LOCAL]]) {header = "use_local.hpp"} : (!ttkernel.LocalTensorAccessor, !ttkernel.LocalTensorAccessor) -> ()
func.func @call_with_compute_local_tensor(
    %arg0: tensor<1x32xui8, #local_layout>)
    attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                ttl.kernel_thread = #ttkernel.thread<compute>} {
  ttl.opaque_call "use_local_pair" (%arg0, %arg0) {header = "use_local.hpp"} : (tensor<1x32xui8, #local_layout>, tensor<1x32xui8, #local_layout>) -> ()
  return
}

// -----

#row_major_layout = #ttl.layout<shape = [1, 32], element_type = ui16,
                                buffer = dram, grid = [1, 1], memory = interleaved>

// Row-major data-movement tensors use the descriptor's aligned page size.
// CHECK-LABEL: func.func @call_with_row_major_tensor
// CHECK: %[[ARGS:.*]] = ttkernel.TensorAccessorArgs(
// CHECK: %[[ACCESSOR:.*]] = ttkernel.TensorAccessor(%[[ARGS]], {{.*}}) : (!ttkernel.TensorAccessorArgs, i32) -> !ttkernel.TensorAccessor
// CHECK: ttkernel.opaque_call "use_row_major"(%[[ACCESSOR]]) {header = "use_row_major.hpp"} : (!ttkernel.TensorAccessor) -> ()
func.func @call_with_row_major_tensor(
    %arg0: tensor<1x32xui16, #row_major_layout>)
    attributes {ttl.base_cta_index = 1 : i32, ttl.crta_indices = [0],
                ttl.kernel_thread = #ttkernel.thread<noc>} {
  ttl.opaque_call "use_row_major" (%arg0) {header = "use_row_major.hpp"} : (tensor<1x32xui16, #row_major_layout>) -> ()
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
