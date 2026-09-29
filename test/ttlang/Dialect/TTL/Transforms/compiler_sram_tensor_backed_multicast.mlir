// Summary: Verifies every tensor-backed multicast receiver is retained for
// runtime address validation in per-node SRAM mode.
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline='memory-model=compiler-sram sram-allocation-strategy=first-fit-decreasing sram-allocation-mode=per-node' --convert-ttkernel-to-emitc | FileCheck %s

// Both destination nodes receive one sender-computed physical base.

// CHECK-LABEL: module attributes
// CHECK-SAME: ttl.memory_model = "compiler-sram"
// CHECK-LABEL: func.func @tensor_backed_fabric
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK-SAME: ttl.sram_receiver_targets = [{device = array<i64: 1>, dfb_index = 1 : i64, node = array<i64: 1, 1>, receivers = [{device = array<i64: 1>, node = array<i64: 1, 0>}, {device = array<i64: 1>, node = array<i64: 1, 1>}]}]

#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#transfer = #ttl.device_transfer<
    domain = #domain,
    edge = <source = <coordinates = [0]>, destination = <coordinates = [1]>>>

module attributes {
  ttl.launch_grid = array<i64: 2, 2>,
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
    %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 1) net 0
        {deviceTransfer = #transfer}
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0>
    ttl.if_dst %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0> {
      %reserved = ttl.cb_reserve %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, f32>>
      %receive = ttl.copy %pipe, %reserved
          : (!ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0>,
             tensor<1x1x!ttcore.tile<32x32, f32>>)
          -> !ttl.receive_request
      ttl.wait %receive : !ttl.receive_request
      ttl.cb_push %destination
          : <[1, 1], !ttcore.tile<32x32, f32>, 1>
    }
    ttl.if_src %pipe
        : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0> {
      %send = ttl.copy %source, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
             !ttl.pipe<src(0, 0) dst(1, 0) to(1, 1) net 0>)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
    }
    func.return
  }
}
