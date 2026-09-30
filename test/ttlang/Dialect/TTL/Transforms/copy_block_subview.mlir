// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Verify that a tensor copy adds the DFB subview tile offset to the block
// pointer while preserving the tensor coordinates.

// CHECK-LABEL: func.func @write_second_tile
// CHECK: %[[DFB_BASE:.*]] = ttkernel.get_read_ptr
// CHECK: %[[DFB_BASE_INDEX:.*]] = arith.index_cast %[[DFB_BASE]]
// CHECK: %[[DFB_ADDRESS_INDEX:.*]] = arith.addi %[[DFB_BASE_INDEX]], %c4096
// CHECK: %[[DFB_ADDRESS:.*]] = arith.index_cast %[[DFB_ADDRESS_INDEX]]
// CHECK: ttkernel.noc_async_write_tile(%{{.*}}, %{{.*}}, %[[DFB_ADDRESS]]

#layout = #ttl.layout<
    shape = [1, 1], element_type = !ttcore.tile<32x32, f32>,
    buffer = dram, grid = [1, 1], memory = interleaved>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @write_second_tile(
      %output: tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
      attributes {
        ttl.base_cta_index = 0 : i32,
        ttl.crta_indices = [0],
        ttl.kernel_thread = #ttkernel.thread<noc>
      } {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>
    %waited = ttl.cb_wait %dfb
        : <[1, 2], !ttcore.tile<32x32, f32>, 1>
        -> tensor<1x2x!ttcore.tile<32x32, f32>>
    %block = ttl.attach_cb %waited, %dfb
        : (tensor<1x2x!ttcore.tile<32x32, f32>>,
           !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, f32>>
    %view = tensor.extract_slice %block[0, 1] [1, 1] [1, 1]
        : tensor<1x2x!ttcore.tile<32x32, f32>>
        to tensor<1x1x!ttcore.tile<32x32, f32>>
    %zero = arith.constant 0 : index
    %output_tile = ttl.tensor_slice %output[%zero, %zero]
        : tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
        -> tensor<1x1x!ttcore.tile<32x32, f32>, #layout>
    %transfer = ttl.copy %view, %output_tile
        : (tensor<1x1x!ttcore.tile<32x32, f32>>,
           tensor<1x1x!ttcore.tile<32x32, f32>, #layout>)
        -> !ttl.transfer_handle<write>
    ttl.wait %transfer : !ttl.transfer_handle<write>
    ttl.cb_pop %dfb : <[1, 2], !ttcore.tile<32x32, f32>, 1>
    func.return
  }
}
