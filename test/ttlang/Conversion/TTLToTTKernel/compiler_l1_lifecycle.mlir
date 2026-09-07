// Verifies address-based selected reset, reset-all, and reconfiguration lowering.
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false})' | FileCheck %s
// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false l1-budget-override=4224})' | FileCheck %s

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reset_test">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reset_test">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reset_test">
#selected = #ttl.synchronized_dfb_reset<0, participants[#compute, #reader, #writer]>
#all = #ttl.synchronized_dfb_reset<1, participants[#compute, #reader, #writer]>

// CHECK-LABEL: module attributes {
// CHECK-SAME: ttl.dfb_reset_count = 2 : i64
// CHECK-SAME: ttl.pipe_sram_scratch_bytes = 32 : i64
// CHECK-LABEL: func.func @compute
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[SCRATCH0:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: ttkernel.opaque_call "experimental::reset_dfb_interfaces"(%[[SCRATCH0]], %[[ZERO]], %[[ZERO]]) {{.*}}dfb_resource_indices = array<i32: 0>
// CHECK-NEXT: %[[SELECTED:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-NEXT: ttkernel.opaque_call "ttlang::l1::resetState"(%[[SELECTED]]) {{.*}}dfb_resource_indices = array<i32: 0>
// CHECK-NEXT: ttkernel.opaque_call "experimental::reset_dfb_interfaces"(%[[SCRATCH0]], %[[ZERO]], %[[ZERO]]) {{.*}}dfb_resource_indices = array<i32: 0>
// CHECK: %[[SCRATCH_BASE:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: %[[SCRATCH1:.*]] = arith.addi %[[SCRATCH_BASE]], %{{.*}} : i32
// CHECK-NEXT: ttkernel.opaque_call "experimental::reset_dfb_interfaces"(%[[SCRATCH1]], %[[ZERO]], %[[ZERO]]) {{.*}}dfb_resource_indices = array<i32: 0, 1>
// CHECK-NEXT: %[[ALL0:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-NEXT: ttkernel.opaque_call "ttlang::l1::resetState"(%[[ALL0]]) {{.*}}dfb_resource_indices = array<i32: 0>
// CHECK-NEXT: %[[ALL1:.*]] = ttkernel.get_compile_time_arg_val(1)
// CHECK-NEXT: ttkernel.opaque_call "ttlang::l1::resetState"(%[[ALL1]]) {{.*}}dfb_resource_indices = array<i32: 1>
// CHECK-NEXT: ttkernel.opaque_call "experimental::reset_dfb_interfaces"(%[[SCRATCH1]], %[[ZERO]], %[[ZERO]]) {{.*}}dfb_resource_indices = array<i32: 0, 1>
// CHECK-NOT: ttl.reset
module attributes {ttl.dfb_allocations = [{block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 0 : i32}, {block_count = 1 : i32, dfb_index = 1 : i32, element_type = !ttcore.tile<32x32, bf16>, l1_allocation_bytes = 2048 : i64, l1_offset = 8 : i64, l1_payload_offset = 2112 : i64, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 1 : i32}], ttl.l1_arena_bytes = 4160 : i64, ttl.launch_grid = [1, 1], ttl.memory_model = "compiler-l1", ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {ttl.base_cta_index = 1 : i32, ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #compute} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs #selected(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    ttl.reset_all_dfbs #all
    return
  }
  func.func @read() attributes {ttl.base_cta_index = 1 : i32, ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #reader, ttl.noc_index = 0 : i32} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs #selected(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    ttl.reset_all_dfbs #all
    return
  }
  func.func @write() attributes {ttl.base_cta_index = 1 : i32, ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #writer, ttl.noc_index = 1 : i32} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %second = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.reset_dfbs #selected(%first : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>)
    ttl.reset_all_dfbs #all
    return
  }
}

// -----

#reconfig_compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "reconfiguration_test">
#reconfig_reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "reconfiguration_test">
#reconfig_writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "reconfiguration_test">
#boundary = #ttl.dfb_reconfiguration<0, participants[#reconfig_compute, #reconfig_reader, #reconfig_writer]>

// CHECK-LABEL: module attributes {
// CHECK-NOT: ttl.dfb_reset_count
// CHECK-SAME: ttl.pipe_sram_scratch_bytes = 32 : i64
// CHECK-LABEL: func.func @compute
// CHECK: %[[RECONFIG_ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[RECONFIG_SCRATCH:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: ttkernel.opaque_call "experimental::reset_dfb_interfaces"(%[[RECONFIG_SCRATCH]], %[[RECONFIG_ZERO]], %[[RECONFIG_ZERO]]) {{.*}}dfb_resource_indices = array<i32: 0>
// CHECK-NEXT: %[[ENDED:.*]] = ttkernel.get_compile_time_arg_val(0)
// CHECK-NEXT: ttkernel.opaque_call "ttlang::l1::resetState"(%[[ENDED]]) {{.*}}dfb_resource_indices = array<i32: 0>
// CHECK-NEXT: ttkernel.opaque_call "experimental::reset_dfb_interfaces"(%[[RECONFIG_SCRATCH]], %[[RECONFIG_ZERO]], %[[RECONFIG_ZERO]]) {{.*}}dfb_resource_indices = array<i32: 0>
// CHECK-NOT: ttl.dfb_reconfiguration
module attributes {ttl.compiler_l1_reconfiguration_resets = [{dfb_indices = array<i32: 0>, ordinal = 0 : i64}], ttl.dfb_allocations = [{block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 0 : i32}, {block_count = 1 : i32, dfb_index = 1 : i32, element_type = !ttcore.tile<32x32, bf16>, l1_allocation_bytes = 2048 : i64, l1_offset = 8 : i64, l1_payload_offset = 2112 : i64, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 1 : i32}], ttl.l1_arena_bytes = 4160 : i64, ttl.launch_grid = [1, 1], ttl.memory_model = "compiler-l1", ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute() attributes {ttl.base_cta_index = 1 : i32, ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #reconfig_compute} {
    %ended = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %preserved = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.dfb_reconfiguration #boundary
    return
  }
  func.func @read() attributes {ttl.base_cta_index = 1 : i32, ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #reconfig_reader, ttl.noc_index = 0 : i32} {
    %ended = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %preserved = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.dfb_reconfiguration #boundary
    return
  }
  func.func @write() attributes {ttl.base_cta_index = 1 : i32, ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #reconfig_writer, ttl.noc_index = 1 : i32} {
    %ended = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %preserved = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.dfb_reconfiguration #boundary
    return
  }
}

// -----

#preserve_compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "preserve_all">
#preserve_reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "preserve_all">
#preserve_writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "preserve_all">
#preserve_boundary = #ttl.dfb_reconfiguration<0, participants[#preserve_compute, #preserve_reader, #preserve_writer]>

// A boundary with no terminal DFB state requires one synchronization barrier and no state writes.
// CHECK-LABEL: func.func @compute_without_terminal_dfb
// CHECK: %[[PRESERVE_ZERO:.*]] = arith.constant 0 : i32
// CHECK: %[[PRESERVE_SCRATCH:.*]] = ttkernel.get_common_arg_val
// CHECK-NEXT: ttkernel.opaque_call "experimental::reset_dfb_interfaces"(%[[PRESERVE_SCRATCH]], %[[PRESERVE_ZERO]], %[[PRESERVE_ZERO]]) {header = "<cstdint>"
// CHECK-NEXT: return
module attributes {ttl.dfb_allocations = [{block_count = 1 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, l1_allocation_bytes = 2048 : i64, l1_offset = 0 : i64, l1_payload_offset = 64 : i64, num_tiles = 1 : i32, page_size = 2048 : i32, storage_index = 0 : i32}], ttl.l1_arena_bytes = 2112 : i64, ttl.launch_grid = [1, 1], ttl.memory_model = "compiler-l1", ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute_without_terminal_dfb() attributes {ttl.base_cta_index = 1 : i32, ttl.kernel_thread = #ttkernel.thread<compute>, ttl.logical_kernel = #preserve_compute} {
    %preserved = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.dfb_reconfiguration #preserve_boundary
    return
  }
  func.func @read_without_terminal_dfb() attributes {ttl.base_cta_index = 1 : i32, ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #preserve_reader, ttl.noc_index = 0 : i32} {
    %preserved = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.dfb_reconfiguration #preserve_boundary
    return
  }
  func.func @write_without_terminal_dfb() attributes {ttl.base_cta_index = 1 : i32, ttl.kernel_thread = #ttkernel.thread<noc>, ttl.logical_kernel = #preserve_writer, ttl.noc_index = 1 : i32} {
    %preserved = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.dfb_reconfiguration #preserve_boundary
    return
  }
}
