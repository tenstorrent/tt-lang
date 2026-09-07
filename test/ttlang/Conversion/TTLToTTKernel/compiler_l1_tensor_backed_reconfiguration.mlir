// Summary: Verifies tensor-independent state reset for a tensor-backed DFB.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(convert-ttl-to-ttkernel{pipe-computed-addresses=false},ttkernel-finalize-tensor-runtime-args,convert-ttkernel-to-emitc)' | FileCheck %s

#compute = #ttl.logical_kernel<kind = compute, identity = "compute", operation = "tensor_backed_reconfiguration">
#reader = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "tensor_backed_reconfiguration">
#writer = #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "tensor_backed_reconfiguration">
#boundary = #ttl.dfb_reconfiguration<0, participants[#compute, #reader, #writer]>
#backing = #ttl.tensor_backing<tensor_index = 0, byte_offset = 0, byte_size = 2048>

// A synchronization-only kernel obtains the DFB state from the arena without
// retaining the tensor payload argument.
// CHECK-LABEL: func.func @compute
// CHECK-SAME: ttl.crta_indices = []
// CHECK-NOT: ttlang::l1::Buffer
// CHECK: %[[ARENA_INDEX:.*]] = emitc.literal "get_compile_time_arg_val(0)"
// CHECK-NEXT: %[[STATE:.*]] = emitc.call_opaque "get_common_arg_val"(%[[ARENA_INDEX]])
// CHECK: emitc.call_opaque "ttlang::l1::resetState"(%[[STATE]])
// CHECK-LABEL: func.func @read
// CHECK-SAME: ttl.crta_indices = [0 : i32]
module attributes {
  ttl.compiler_l1_reconfiguration_resets = [{dfb_indices = array<i32: 0>, ordinal = 0 : i64}],
  ttl.dfb_allocations = [{
    allocation_nodes = [[0, 0]],
    block_count = 1 : i32,
    dfb_index = 0 : i32,
    element_type = !ttcore.tile<32x32, bf16>,
    l1_offset = 0 : i64,
    num_tiles = 1 : i32,
    page_size = 2048 : i32,
    storage_capacity_pages = 1 : i32,
    storage_index = 0 : i32,
    storage_segments = [{nodes = [[0, 0]], tensor_backing = #backing}]
  }],
  ttl.l1_arena_bytes = 64 : i64,
  ttl.launch_grid = [1, 1],
  ttl.memory_model = "compiler-l1",
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @compute() attributes {
    ttl.base_cta_index = 1 : i32,
    ttl.crta_indices = [],
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #compute
  } {
    ttl.dfb_reconfiguration #boundary
    return
  }

  func.func @read() attributes {
    ttl.base_cta_index = 1 : i32,
    ttl.crta_indices = [0 : i32],
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #reader,
    ttl.noc_index = 0 : i32
  } {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index, tensor_backing = #backing}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.dfb_reconfiguration #boundary
    return
  }

  func.func @write() attributes {
    ttl.base_cta_index = 1 : i32,
    ttl.crta_indices = [],
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #writer,
    ttl.noc_index = 1 : i32
  } {
    ttl.dfb_reconfiguration #boundary
    return
  }
}
