// Tests DFB address-scope propagation into physical allocation metadata.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s

// Local DFBs on disjoint nodes may share backing storage. A remote-uniform DFB
// receives one distinct allocation covering every node where it is used.
// CHECK: ttl.dfb_allocations = [
// CHECK-SAME: {allocation_nodes = {{\[\[0, 0\]\]}}, block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 0 : i32},
// CHECK-SAME: {allocation_nodes = {{\[\[1, 0\]\]}}, block_count = 1 : i32, dfb_index = 1 : i32, element_type = !ttcore.tile<32x32, f32>, num_tiles = 1 : i32, page_size = 4096 : i32, storage_index = 0 : i32},
// CHECK-SAME: {address_scope = "remote_uniform", allocation_nodes = {{\[\[0, 0\], \[1, 0\]\]}}, block_count = 1 : i32, dfb_index = 2 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 1 : i32}

module attributes {ttl.launch_grid = [2, 1]} {
  func.func @per_node_address_scopes()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32} {
    %default_local = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %explicit_local = ttl.bind_cb {cb_index = 1, block_count = 1}
        {address_scope = "local", dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %remote_uniform = ttl.bind_cb {cb_index = 2, block_count = 1}
        {address_scope = "remote_uniform", dfb_id = 2 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %node_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %first_node = arith.cmpi eq, %node_x, %zero : index
    %second_node = arith.cmpi ne, %node_x, %zero : index
    scf.if %first_node {
      ttl.opaque_call "use_default_local" dfb_dependencies(
          %default_local : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
          () {header = "effects.hpp"} : () -> ()
    }
    scf.if %second_node {
      ttl.opaque_call "use_explicit_local" dfb_dependencies(
          %explicit_local : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>)
          () {header = "effects.hpp"} : () -> ()
    }
    ttl.opaque_call "use_remote_uniform" dfb_dependencies(
        %remote_uniform : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        () {header = "effects.hpp"} : () -> ()
    return
  }
}

// -----

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#reconfiguration = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

// Both logical DFBs reuse physical index zero in disjoint configuration epochs.
// The initial and replacement descriptors retain the remote-uniform contract.
// CHECK: ttl.dfb_allocations = [{address_scope = "remote_uniform",
// CHECK-SAME: dfb_index = 0 : i32
// CHECK: ttl.dfb_reconfiguration_plan = {
// CHECK-SAME: configurations = [{address_scope = "remote_uniform",
// CHECK-SAME: {address_scope = "remote_uniform",
// CHECK-SAME: entry_reconfiguration = 0 : i64
// CHECK-SAME: dfb_index = 0 : i32
// CHECK-LABEL: func.func @reconfigured_compute
// CHECK-COUNT-2: ttl.bind_cb{cb_index = 0,

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @reconfigured_compute() attributes {
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1}
        {address_scope = "remote_uniform", dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.opaque_call "use_first" dfb_dependencies(
        %first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>,
                     #ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
    ttl.dfb_reconfiguration #reconfiguration
    %second = ttl.bind_cb {cb_index = 1, block_count = 1}
        {address_scope = "remote_uniform", dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.opaque_call "use_second" dfb_dependencies(
        %second : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
        dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 1>,
                     #ttl.dfb_protocol_effect<push, 0, 1>,
                     #ttl.dfb_protocol_effect<wait, 0, 1>,
                     #ttl.dfb_protocol_effect<pop, 0, 1>]
        () {header = "effects.hpp"} : () -> ()
    return
  }

  func.func @reconfigured_reader() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    ttl.dfb_reconfiguration #reconfiguration
    return
  }

  func.func @reconfigured_writer() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    ttl.dfb_reconfiguration #reconfiguration
    return
  }
}

// -----

// Omitting the scope and spelling local explicitly are equivalent declarations
// of one logical DFB.
module attributes {ttl.launch_grid = [1, 1]} {
  func.func @implicit_local() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.noc_index = 0 : i32
  } {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }

  func.func @explicit_local() attributes {
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.noc_index = 1 : i32
  } {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {address_scope = "local", dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
