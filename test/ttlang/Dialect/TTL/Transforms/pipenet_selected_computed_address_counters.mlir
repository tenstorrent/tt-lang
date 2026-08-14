// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Selected matching edges execute the same sender and receiver callbacks on
// disjoint devices. Their repeated transfers therefore use the same local
// slot-counter index while retaining distinct per-device counter storage.

// CHECK-LABEL: func.func @sender
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK: memref.alloca() : memref<1xi32>
// CHECK: ttkernel.experimental.constant_table_lookup {{.*}}, [0, 0] : index
// CHECK-LABEL: func.func @receiver
// CHECK: ttkernel.experimental.constant_table_lookup {{.*}}, [0, 0] : index

#domain = #ttl.device_domain<components = <name = "device", extent = [4]>>
#records = #ttl.pipenet_records<net 0 name "disjoint_matching" pipes [
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [1]>>>>,
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #domain,
        edge = <source = <coordinates = [2]>,
                destination = <coordinates = [3]>>>>
]>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @sender() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %one = arith.constant 1 : index
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      scf.for %iteration = %zero to %two step %one {
        %send = ttl.copy %source, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
               !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
      ttl.yield
    }
    func.return
  }

  func.func @receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %one = arith.constant 1 : index
    ttl.pipenet_foreach_dst attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      scf.for %iteration = %zero to %two step %one {
        %reserved = ttl.cb_reserve %destination
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %post = ttl.copy %pipe, %reserved
            : (!ttl.selected_pipe_dst,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.transfer_handle
        ttl.wait %post : !ttl.transfer_handle
        ttl.cb_push %destination
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.yield
    }
    func.return
  }
}

// -----

// Repeated fanout records share one source device but use distinct local
// incident ordinals. They require independent slot counters in that sender.

// CHECK-LABEL: func.func @fanout_sender
// CHECK-SAME: ttl.pipe_computed_address_dfb_indices = array<i32: 1>
// CHECK: memref.alloca() : memref<2xi32>
// CHECK: ttkernel.experimental.constant_table_lookup {{.*}}, [0, 1] : index

#fanout_domain = #ttl.device_domain<components = <name = "device", extent = [3]>>
#fanout_records = #ttl.pipenet_records<net 1 name "repeated_fanout" pipes [
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #fanout_domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [1]>>>>,
  #ttl.pipe_record<
      srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
      dstEndX = 0, dstEndY = 0,
      deviceTransfer = <
        domain = #fanout_domain,
        edge = <source = <coordinates = [0]>,
                destination = <coordinates = [2]>>>>
]>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func @fanout_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1} {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %one = arith.constant 1 : index
    ttl.pipenet_foreach_src attributes {records = #fanout_records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      scf.for %iteration = %zero to %two step %one {
        %send = ttl.copy %source, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>,
               !ttl.selected_pipe_src) -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
      }
      ttl.yield
    }
    func.return
  }

  func.func @fanout_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %one = arith.constant 1 : index
    ttl.pipenet_foreach_dst attributes {records = #fanout_records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      scf.for %iteration = %zero to %two step %one {
        %reserved = ttl.cb_reserve %destination
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
            -> tensor<1x1x!ttcore.tile<32x32, f32>>
        %post = ttl.copy %pipe, %reserved
            : (!ttl.selected_pipe_dst,
               tensor<1x1x!ttcore.tile<32x32, f32>>)
            -> !ttl.transfer_handle
        ttl.wait %post : !ttl.transfer_handle
        ttl.cb_push %destination
            : <[1, 1], !ttcore.tile<32x32, f32>, 2>
      }
      ttl.yield
    }
    func.return
  }
}
