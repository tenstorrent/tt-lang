// Summary: Verify graph callbacks reserve resources for every concrete
// destination without materializing graph-edge and node-pipe pairs.
// RUN: ttlang-opt %s --ttl-form-pipe-transports='group-size=1' | FileCheck %s

#records = #ttl.pipenet_records<net 0 name "all-to-all" mappings
  <graph = <domain = <components = <name = "device", extent = [4]>>,
    kind = all_to_all, componentName = "device", properties = {}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>]>>

// Twelve device edges and one send per callback require twelve conservative
// resource units. The pass accounts for them without creating twelve records.
// CHECK: module attributes
// CHECK-SAME: ttl.launch_grid = array<i64: 1, 1>
// CHECK-SAME: ttl.pipe_conservative_l1_bytes = 2688 : i64
module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @graph_callback()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %send = ttl.copy %dfb, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.yield
    }
    func.return
  }
}
