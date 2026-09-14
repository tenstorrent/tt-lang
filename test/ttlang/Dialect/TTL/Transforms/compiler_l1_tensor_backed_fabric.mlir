// Summary: Verifies compiler-managed tensor-backed receivers with generated fabric routes.
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline='memory-model=compiler-l1 l1-allocation-strategy=first-fit-decreasing' --convert-ttkernel-to-emitc | FileCheck %s
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline='memory-model=compiler-l1 l1-allocation-strategy=best-fit-decreasing' --convert-ttkernel-to-emitc | FileCheck %s

// Generated fabric transfers use the tensor runtime argument as the stable
// receiver base while retaining the compiler-SRAM arena for interface state.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.l1_arena_bytes = 4160 : i64
// CHECK-SAME: ttl.memory_model = "compiler-l1"
// CHECK-LABEL: func.func @tensor_backed_fabric
// CHECK-SAME: ttl.fabric_routes = [
// CHECK-SAME: ttl.fabric_runtime_arg_base_common_index = 4 : i64
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK: emitc.verbatim "ttlang::l1::Buffer<4096, 1, 1, 1, 0, 0> cb_ctarg_1({});"
// CHECK: call_opaque "experimental::routing_plane_atomic_inc"
// CHECK: call_opaque "experimental::routing_plane_fused_write_atomic_inc"
// CHECK-NOT: call_opaque "noc_inline_dw_write"

#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#transfer = #ttl.device_transfer<
    domain = #domain,
    edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {
  ttl.launch_grid = array<i64: 2, 1>,
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @tensor_backed_fabric(
      %tensor: tensor<1x1x!ttcore.tile<32x32, f32>>)
      attributes {
        ttl.kernel_thread = #ttkernel.thread<noc>,
        ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement,
            identity = "reader", operation = "tensor_backed_fabric">,
        ttl.noc_index = 0 : i32, ttl.base_cta_index = 1 : i32,
        ttl.crta_indices = [0 : i32]
      } {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %destination = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index,
         tensor_backing = #ttl.tensor_backing<tensor_index = 0,
             byte_offset = 0, byte_size = 4096>}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
        {deviceTransfer = #transfer}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %reserved = ttl.cb_reserve %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %receive = ttl.copy %pipe, %reserved
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
      ttl.cb_push %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0> {
      %send = ttl.copy %source, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}
