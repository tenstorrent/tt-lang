// Summary: Verifies tensor-backed DFB storage across configuration epochs.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

// CHECK: ttl.dfb_allocations = [
// CHECK-SAME: block_count = 2 : i32
// CHECK-SAME: dfb_index = 0 : i32
// CHECK-SAME: byte_offset = 0
// CHECK-SAME: byte_size = 4096
// CHECK-SAME: ttl.dfb_reconfiguration_plan = {
// CHECK-SAME: boundary_ordinals = array<i64: 0>
// CHECK-SAME: block_count = 2 : i32
// CHECK-SAME: byte_offset = 0
// CHECK-SAME: byte_size = 4096
// CHECK-SAME: block_count = 1 : i32
// CHECK-SAME: entry_reconfiguration = 0 : i64
// CHECK-SAME: byte_offset = 2048
// CHECK-SAME: byte_size = 2048
// CHECK-LABEL: func.func @compute
// CHECK-SAME: ttl.base_cta_index = 2 : i32

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @compute() attributes {
    ttl.base_cta_index = 2 : i32,
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2}
        {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "first" dfb_dependencies(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    ttl.dfb_reconfiguration #boundary

    %second = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 2048, byte_size = 2048>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.opaque_call "second" dfb_dependencies(%second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>, #ttl.dfb_protocol_effect<push, 0, 1>, #ttl.dfb_protocol_effect<wait, 0, 1>, #ttl.dfb_protocol_effect<pop, 0, 1>] () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @read() attributes {
    ttl.base_cta_index = 2 : i32,
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    ttl.dfb_reconfiguration #boundary
    return
  }

  func.func @write() attributes {
    ttl.base_cta_index = 2 : i32,
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    ttl.dfb_reconfiguration #boundary
    return
  }
}

// -----

// Verifies that the static descriptor covers a core first used after the
// reconfiguration boundary without installing its future tensor alias early.
#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

// CHECK: ttl.dfb_allocations = [
// CHECK-SAME: dfb_index = 0 : i32
// CHECK-SAME: storage_segments = [
// CHECK-SAME: nodes = {{\[\[0, 0\]\]}}
// CHECK-SAME: tensor_backing = #ttl.tensor_backing<tensor_index = 0
// CHECK-SAME: nodes = {{\[\[1, 0\]\]}}
// CHECK-NOT: tensor_backing
// CHECK-SAME: ttl.dfb_reconfiguration_plan = {
// CHECK-SAME: entry_reconfiguration = 0 : i64
// CHECK-SAME: nodes = {{\[\[1, 0\]\]}}
// CHECK-SAME: tensor_backing = #ttl.tensor_backing<tensor_index = 0
module attributes {ttl.launch_grid = [2, 1]} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index, tensor_backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %node_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %first_node = arith.cmpi eq, %node_x, %zero : index
    %second_node = arith.cmpi ne, %node_x, %zero : index
    scf.if %first_node {
      ttl.opaque_call "first" dfb_dependencies(
          %first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                       #ttl.dfb_protocol_effect<push, 0, 1>,
                       #ttl.dfb_protocol_effect<wait, 0, 1>,
                       #ttl.dfb_protocol_effect<pop, 0, 1>]
          () {header = "effects.hpp"} : () -> ()
    }
    ttl.dfb_reconfiguration #boundary
    scf.if %second_node {
      ttl.opaque_call "second" dfb_dependencies(
          %second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                       #ttl.dfb_protocol_effect<push, 0, 1>,
                       #ttl.dfb_protocol_effect<wait, 0, 1>,
                       #ttl.dfb_protocol_effect<pop, 0, 1>]
          () {header = "effects.hpp"} : () -> ()
    }
    return
  }

  func.func @read() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    ttl.dfb_reconfiguration #boundary
    return
  }

  func.func @write() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    ttl.dfb_reconfiguration #boundary
    return
  }
}
