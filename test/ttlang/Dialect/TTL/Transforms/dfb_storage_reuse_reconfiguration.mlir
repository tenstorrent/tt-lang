// Summary: Reconfigured descriptors reuse non-overlapping runtime storage.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{memory-model=compiler-l1 reuse-user-dfbs=true})' | FileCheck %s

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "operation">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>

// The two lifetimes are ordered and use different physical descriptors. Their
// byte-addressed storage can be shared because the first lifetime completes
// before the second configuration becomes active.

// CHECK: ttl.dfb_allocations = [
// CHECK-SAME: dfb_index = 0 : i32
// CHECK-SAME: l1_payload_offset = [[SHARED_PAYLOAD:[0-9]+]] : i64
// CHECK-SAME: dfb_index = 1 : i32
// CHECK-SAME: l1_payload_offset = [[SHARED_PAYLOAD]] : i64

module attributes {ttl.launch_grid = [1, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
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
// CHECK-SAME: dfb_index = 0 : i32
// CHECK-SAME: l1_payload_offset = [[DISJOINT_PAYLOAD:[0-9]+]] : i64
// CHECK-SAME: dfb_index = 1 : i32
// CHECK-SAME: l1_payload_offset = [[DISJOINT_PAYLOAD]] : i64

module attributes {ttl.launch_grid = [2, 1], ttl.target_arch = #ttcore.arch<blackhole>} {
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
