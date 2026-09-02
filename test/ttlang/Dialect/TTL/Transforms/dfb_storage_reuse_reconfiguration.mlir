// Summary: Reconfigured descriptors retain dedicated runtime storage.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

// The two lifetimes are ordered and use different physical descriptors. The
// second descriptor requires runtime reconfiguration storage, so the static
// descriptor cannot share that allocation.

// CHECK: ttl.dfb_allocations = [
// CHECK-SAME: {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 1 : i32},
// CHECK-SAME: {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 1 : i32, dfb_index = 1 : i32, element_type = !ttcore.tile<32x32, f32>, num_tiles = 1 : i32, page_size = 4096 : i32, storage_index = 0 : i32}

module attributes {ttl.launch_grid = [1, 1]} {
  func.func @compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.opaque_call "first" dfb_dependencies(
        %first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>,
                     #ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
    ttl.dfb_reconfiguration #boundary
    %second = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    ttl.opaque_call "second" dfb_dependencies(
        %second : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>,
                     #ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
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

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

// Hidden reconfiguration backing on one launch node does not prevent a static
// descriptor on another launch node from using the same storage index.

// CHECK: ttl.dfb_allocations = [
// CHECK-SAME: {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 0 : i32},
// CHECK-SAME: {allocation_nodes = {{\[\[1, 0\]\]}}, block_count = 1 : i32, dfb_index = 1 : i32, element_type = !ttcore.tile<32x32, f32>, num_tiles = 1 : i32, page_size = 4096 : i32, storage_index = 0 : i32}

module attributes {ttl.launch_grid = [2, 1]} {
  func.func @compute_disjoint_nodes() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %on_first_node = arith.cmpi eq, %core_x, %zero : index
    %on_second_node = arith.cmpi ne, %core_x, %zero : index
    scf.if %on_first_node {
      ttl.opaque_call "first" dfb_dependencies(
          %first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                       #ttl.dfb_protocol_effect<push, 0, 1>,
                       #ttl.dfb_protocol_effect<wait, 0, 1>,
                       #ttl.dfb_protocol_effect<pop, 0, 1>]
          () {header = "effects.hpp"} : () -> ()
    }
    ttl.dfb_reconfiguration #boundary
    scf.if %on_second_node {
      ttl.opaque_call "second" dfb_dependencies(
          %second : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
          dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                       #ttl.dfb_protocol_effect<push, 0, 1>,
                       #ttl.dfb_protocol_effect<wait, 0, 1>,
                       #ttl.dfb_protocol_effect<pop, 0, 1>]
          () {header = "effects.hpp"} : () -> ()
    }
    return
  }

  func.func @read_disjoint_nodes() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    ttl.dfb_reconfiguration #boundary
    return
  }

  func.func @write_disjoint_nodes() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    ttl.dfb_reconfiguration #boundary
    return
  }
}
