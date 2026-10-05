// Summary: Verify schedule analysis matches graph callback wait-any candidates
// with their sends.
// RUN: ttlang-opt %s --ttl-verify-pipenet-schedule | FileCheck %s

#records = #ttl.pipenet_records<net 0 name "gather" mappings
  <graph = <domain = <components = <name = "device", extent = [2]>>,
    kind = gather, componentName = "device",
    properties = {root = #ttl.device_ref<coordinates = [0]>}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>]>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  // CHECK-LABEL: func.func @send
  // CHECK: ttl.pipenet_foreach_src
  func.func @send() attributes {
      ttl.base_cta_index = 2 : i32,
      ttl.crta_indices = [],
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
      ttl.noc_index = 0 : i32} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %reserved = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      %available = ttl.cb_wait %dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %send = ttl.copy %dfb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      ttl.yield
    }
    func.return
  }

  // CHECK-LABEL: func.func @receive
  // CHECK: ttl.wait_any
  func.func @receive() attributes {
      ttl.base_cta_index = 2 : i32,
      ttl.crta_indices = [],
      ttl.kernel_thread = #ttkernel.thread<noc>,
      ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
      ttl.noc_index = 1 : i32} {
    %dfb = ttl.bind_cb {cb_index = 1, block_count = 1} {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %start = arith.constant 0 : index
    ttl.pipenet_foreach_dst attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %block = ttl.cb_reserve %dfb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %request = ttl.copy %pipe, %block
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, bf16>>) -> !ttl.receive_request
      %ready = ttl.wait_any %request start %start
          : (!ttl.receive_request, index) -> !ttl.ready_receive
      ttl.wait %request : !ttl.receive_request
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      ttl.yield
    }
    func.return
  }
}
