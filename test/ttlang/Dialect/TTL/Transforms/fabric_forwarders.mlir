// RUN: ttlang-opt %s --split-input-file -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verify Blackhole fabric forwarder selection and conservative
// fallback decisions.

#domain = #ttl.device_domain<components = <name = "device", extent = [4]>>
#records = #ttl.pipenet_records<net 0 name "disjoint" mappings
  <graph = <domain = #domain, kind = explicit,
            properties = {edges = [#ttl.transfer_edge<
              source = <coordinates = [0]>,
              destination = <coordinates = [1]>>,
              #ttl.transfer_edge<source = <coordinates = [2]>,
              destination = <coordinates = [3]>>,
              #ttl.transfer_edge<source = <coordinates = [1]>,
              destination = <coordinates = [0]>>,
              #ttl.transfer_edge<source = <coordinates = [3]>,
              destination = <coordinates = [2]>>]}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>,
         <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0,
          dstEndX = 1, dstEndY = 0>,
         <srcX = 2, srcY = 0, dstStartX = 2, dstStartY = 0,
          dstEndX = 2, dstEndY = 0>,
         <srcX = 3, srcY = 0, dstStartX = 3, dstStartY = 0,
          dstEndX = 3, dstEndY = 0>]>>
#outer_records = #ttl.pipenet_records<net 1 name "uniform-control" mappings
  <graph = <domain = #domain, kind = explicit,
            properties = {edges = [#ttl.transfer_edge<
              source = <coordinates = [0]>,
              destination = <coordinates = [1]>>,
              #ttl.transfer_edge<source = <coordinates = [2]>,
              destination = <coordinates = [3]>>,
              #ttl.transfer_edge<source = <coordinates = [1]>,
              destination = <coordinates = [0]>>,
              #ttl.transfer_edge<source = <coordinates = [3]>,
              destination = <coordinates = [2]>>]}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>,
         <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0,
          dstEndX = 1, dstEndY = 0>,
         <srcX = 2, srcY = 0, dstStartX = 2, dstStartY = 0,
          dstEndX = 2, dstEndY = 0>,
         <srcX = 3, srcY = 0, dstStartX = 3, dstStartY = 0,
          dstEndX = 3, dstEndY = 0>]>>

// Two bidirectional device pairs use the same four-worker protocol. A static
// loop executes the uniform outer callback twice. The sender's one-worker and
// three-worker operations select forwarders 0 and 1, while the receiver's
// four-worker operation selects forwarders 0 and 2. Manager ownership remains
// independent because the physical connection owners differ.
// CHECK-LABEL: module @disjoint_domains attributes
// CHECK-SAME: ttl.pipe_sram_scratch_bytes = 8288 : i64
// CHECK-SAME: ttl.pipe_sync_semaphore_count = 0 : i64
// CHECK-LABEL: func.func @disjoint_sender
// CHECK-SAME: ttl.fabric_manager_intervals = [#ttl.fabric_manager_interval<
// CHECK-SAME: interferingIntervals = ["generated.1"]>]
// CHECK-SAME: ttl.fabric_routes = [{
// CHECK-SAME: source_nodes = [array<i64: 0, 0>, array<i64: 1, 0>]
// CHECK: ttkernel.noc_async_write
// CHECK-NEXT: ttkernel.noc_async_write_barrier
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.noc_async_write
// CHECK-NEXT: ttkernel.noc_async_write_barrier
// CHECK: ttkernel.routing_plane.fused_write_atomic_inc
// CHECK-LABEL: func.func @disjoint_receiver
// CHECK-SAME: ttl.fabric_manager_intervals = [#ttl.fabric_manager_interval<
// CHECK-SAME: interferingIntervals = ["generated.0"]>]
// CHECK-SAME: ttl.fabric_routes = [{
// CHECK-SAME: source_nodes = [array<i64: 0, 0>, array<i64: 2, 0>]
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental.semaphore_wait_min
// CHECK: ttkernel.routing_plane.atomic_inc
// CHECK: ttkernel.noc_semaphore_inc
// CHECK: ttkernel.experimental.semaphore_wait_min
module @disjoint_domains attributes {
  ttl.launch_grid = [4, 1],
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @disjoint_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.pipenet_foreach_src attributes {records = #outer_records} {
      ^bb0(%outer_pipe: !ttl.selected_pipe_src):
        %node_x = ttl.core_x : index
        %c1 = arith.constant 1 : index
        %left = arith.cmpi slt, %node_x, %c1 : index
        scf.if %left {
          ttl.pipenet_foreach_src attributes {records = #records} {
          ^bb0(%pipe: !ttl.selected_pipe_src):
            %send = ttl.copy %source, %pipe
                : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
                   !ttl.selected_pipe_src)
                -> !ttl.transfer_handle<write>
            ttl.wait %send : !ttl.transfer_handle<write>
            ttl.yield
          }
        }
        %right = arith.cmpi sge, %node_x, %c1 : index
        scf.if %right {
          ttl.pipenet_foreach_src attributes {records = #records} {
          ^bb0(%pipe: !ttl.selected_pipe_src):
            %send = ttl.copy %source, %pipe
                : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
                   !ttl.selected_pipe_src)
                -> !ttl.transfer_handle<write>
            ttl.wait %send : !ttl.transfer_handle<write>
            ttl.yield
          }
        }
        ttl.yield
      }
    }
    func.return
  }

  func.func @disjoint_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.pipenet_foreach_dst attributes {records = #outer_records} {
      ^bb0(%outer_pipe: !ttl.selected_pipe_dst):
        ttl.pipenet_foreach_dst attributes {records = #records} {
        ^bb0(%pipe: !ttl.selected_pipe_dst):
          %reserved = ttl.cb_reserve %destination
              : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
              -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %post = ttl.copy %pipe, %reserved
              : (!ttl.selected_pipe_dst,
                 tensor<1x1x!ttcore.tile<32x32, bf16>>)
              -> !ttl.receive_request
          ttl.wait %post : !ttl.receive_request
          ttl.cb_push %destination
              : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          ttl.yield
        }
        ttl.yield
      }
    }
    func.return
  }
}

// -----

#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#records = #ttl.pipenet_records<net 0 name "overlap" mappings
  <graph = <domain = #domain, kind = explicit,
            properties = {edges = [#ttl.transfer_edge<
              source = <coordinates = [0]>,
              destination = <coordinates = [1]>>]}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>,
         <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0,
          dstEndX = 1, dstEndY = 0>,
         <srcX = 2, srcY = 0, dstStartX = 2, dstStartY = 0,
          dstEndX = 2, dstEndY = 0>,
         <srcX = 3, srcY = 0, dstStartX = 3, dstStartY = 0,
          dstEndX = 3, dstEndY = 0>]>>

// The sender retains one direct connection per worker because the two
// operation-local counters would alias on workers 1 and 2. Only those
// overlapping receiver workers require readiness connections.
// CHECK-LABEL: module @overlapping_domains attributes
// CHECK-NOT: ttl.pipe_sram_scratch_bytes
// CHECK-LABEL: func.func @overlapping_sender
// CHECK-SAME: ttl.fabric_routes = [{
// CHECK-SAME: source_nodes = [array<i64: 0, 0>, array<i64: 1, 0>, array<i64: 2, 0>, array<i64: 3, 0>]
// CHECK-LABEL: func.func @overlapping_receiver
// CHECK-SAME: ttl.fabric_routes = [{
// CHECK-SAME: source_nodes = [array<i64: 1, 0>, array<i64: 2, 0>]
module @overlapping_domains attributes {
  ttl.launch_grid = [4, 1],
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @overlapping_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %node_x = ttl.core_x : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %left = arith.cmpi slt, %node_x, %c3 : index
    scf.if %left {
      ttl.pipenet_foreach_src attributes {records = #records} {
      ^bb0(%pipe: !ttl.selected_pipe_src):
        %send = ttl.copy %source, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
               !ttl.selected_pipe_src)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        ttl.yield
      }
    }
    %right = arith.cmpi sge, %node_x, %c1 : index
    scf.if %right {
      ttl.pipenet_foreach_src attributes {records = #records} {
      ^bb0(%pipe: !ttl.selected_pipe_src):
        %send = ttl.copy %source, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
               !ttl.selected_pipe_src)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        ttl.yield
      }
    }
    func.return
  }

  func.func @overlapping_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %node_x = ttl.core_x : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %left = arith.cmpi slt, %node_x, %c3 : index
    scf.if %left {
      ttl.pipenet_foreach_dst attributes {records = #records} {
      ^bb0(%pipe: !ttl.selected_pipe_dst):
        %reserved = ttl.cb_reserve %destination
            : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %post = ttl.copy %pipe, %reserved
            : (!ttl.selected_pipe_dst,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.receive_request
        ttl.wait %post : !ttl.receive_request
        ttl.cb_push %destination : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        ttl.yield
      }
    }
    %right = arith.cmpi sge, %node_x, %c1 : index
    scf.if %right {
      ttl.pipenet_foreach_dst attributes {records = #records} {
      ^bb0(%pipe: !ttl.selected_pipe_dst):
        %reserved = ttl.cb_reserve %destination
            : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %post = ttl.copy %pipe, %reserved
            : (!ttl.selected_pipe_dst,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.receive_request
        ttl.wait %post : !ttl.receive_request
        ttl.cb_push %destination : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        ttl.yield
      }
    }
    func.return
  }
}

// -----

#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#inner_records = #ttl.pipenet_records<net 0 name "nested-fabric" mappings
  <graph = <domain = #domain, kind = explicit,
            properties = {edges = [#ttl.transfer_edge<
              source = <coordinates = [0]>,
              destination = <coordinates = [1]>>]}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>,
         <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0,
          dstEndX = 1, dstEndY = 0>,
         <srcX = 2, srcY = 0, dstStartX = 2, dstStartY = 0,
          dstEndX = 2, dstEndY = 0>,
         <srcX = 3, srcY = 0, dstStartX = 3, dstStartY = 0,
          dstEndX = 3, dstEndY = 0>]>>
#outer_records = #ttl.pipenet_records<net 1 name "nested-control" mappings
  <graph = <domain = #domain, kind = explicit,
            properties = {edges = [#ttl.transfer_edge<
              source = <coordinates = [0]>,
              destination = <coordinates = [1]>>]}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>,
         <srcX = 0, srcY = 0, dstStartX = 2, dstStartY = 0,
          dstEndX = 2, dstEndY = 0>,
         <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0,
          dstEndX = 1, dstEndY = 0>,
         <srcX = 2, srcY = 0, dstStartX = 2, dstStartY = 0,
          dstEndX = 2, dstEndY = 0>,
         <srcX = 2, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>,
         <srcX = 3, srcY = 0, dstStartX = 3, dstStartY = 0,
          dstEndX = 3, dstEndY = 0>]>>

// The outer callback executes twice on workers 0 and 2, but once on workers 1
// and 3. Sender connections remain direct because their execution counts
// differ; only the repeated receiver workers require readiness connections.
// CHECK-LABEL: module @nonuniform_nested_callbacks attributes
// CHECK-NOT: ttl.pipe_sram_scratch_bytes
// CHECK-LABEL: func.func @nested_sender
// CHECK-SAME: ttl.fabric_routes = [{
// CHECK-SAME: source_nodes = [array<i64: 0, 0>, array<i64: 1, 0>, array<i64: 2, 0>, array<i64: 3, 0>]
// CHECK-LABEL: func.func @nested_receiver
// CHECK-SAME: ttl.fabric_routes = [{
// CHECK-SAME: source_nodes = [array<i64: 0, 0>, array<i64: 2, 0>]
module @nonuniform_nested_callbacks attributes {
  ttl.launch_grid = [4, 1],
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @nested_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.pipenet_foreach_src attributes {records = #outer_records} {
    ^bb0(%outer_pipe: !ttl.selected_pipe_src):
      ttl.pipenet_foreach_src attributes {records = #inner_records} {
      ^bb0(%pipe: !ttl.selected_pipe_src):
        %send = ttl.copy %source, %pipe
            : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
               !ttl.selected_pipe_src)
            -> !ttl.transfer_handle<write>
        ttl.wait %send : !ttl.transfer_handle<write>
        ttl.yield
      }
      ttl.yield
    }
    func.return
  }

  func.func @nested_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.pipenet_foreach_dst attributes {records = #outer_records} {
    ^bb0(%outer_pipe: !ttl.selected_pipe_dst):
      ttl.pipenet_foreach_dst attributes {records = #inner_records} {
      ^bb0(%pipe: !ttl.selected_pipe_dst):
        %reserved = ttl.cb_reserve %destination
            : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
            -> tensor<1x1x!ttcore.tile<32x32, bf16>>
        %post = ttl.copy %pipe, %reserved
            : (!ttl.selected_pipe_dst,
               tensor<1x1x!ttcore.tile<32x32, bf16>>)
            -> !ttl.receive_request
        ttl.wait %post : !ttl.receive_request
        ttl.cb_push %destination : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
        ttl.yield
      }
      ttl.yield
    }
    func.return
  }
}

// -----

#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#records0 = #ttl.pipenet_records<net 0 name "first-operation" mappings
  <graph = <domain = #domain, kind = explicit,
            properties = {edges = [#ttl.transfer_edge<
              source = <coordinates = [0]>,
              destination = <coordinates = [1]>>]}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>]>>
#records1 = #ttl.pipenet_records<net 1 name "second-operation" mappings
  <graph = <domain = #domain, kind = explicit,
            properties = {edges = [#ttl.transfer_edge<
              source = <coordinates = [0]>,
              destination = <coordinates = [1]>>]}>,
   pipes[<srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0,
          dstEndX = 1, dstEndY = 0>]>>
#records2 = #ttl.pipenet_records<net 2 name "remaining-operation" mappings
  <graph = <domain = #domain, kind = explicit,
            properties = {edges = [#ttl.transfer_edge<
              source = <coordinates = [0]>,
              destination = <coordinates = [1]>>]}>,
   pipes[<srcX = 2, srcY = 0, dstStartX = 2, dstStartY = 0,
          dstEndX = 2, dstEndY = 0>,
         <srcX = 3, srcY = 0, dstStartX = 3, dstStartY = 0,
          dstEndX = 3, dstEndY = 0>]>>

// Three independently executing source operations require three forwarders.
// The two-forwarder limit therefore retains all four direct source workers.
// CHECK-LABEL: module @too_many_operations attributes
// CHECK-LABEL: func.func @three_operation_sender
// CHECK-SAME: ttl.fabric_routes = [{
// CHECK-SAME: source_nodes = [array<i64: 0, 0>, array<i64: 1, 0>, array<i64: 2, 0>, array<i64: 3, 0>]
module @too_many_operations attributes {
  ttl.launch_grid = [4, 1],
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @three_operation_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.pipenet_foreach_src attributes {records = #records0} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %send = ttl.copy %source, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.yield
    }
    ttl.pipenet_foreach_src attributes {records = #records1} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %send = ttl.copy %source, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.yield
    }
    ttl.pipenet_foreach_src attributes {records = #records2} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %send = ttl.copy %source, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.yield
    }
    func.return
  }

  func.func @three_operation_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.pipenet_foreach_dst attributes {records = #records0} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %reserved = ttl.cb_reserve %destination
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %post = ttl.copy %pipe, %reserved
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      ttl.wait %post : !ttl.receive_request
      ttl.cb_push %destination : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      ttl.yield
    }
    ttl.pipenet_foreach_dst attributes {records = #records1} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %reserved = ttl.cb_reserve %destination
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %post = ttl.copy %pipe, %reserved
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      ttl.wait %post : !ttl.receive_request
      ttl.cb_push %destination : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      ttl.yield
    }
    ttl.pipenet_foreach_dst attributes {records = #records2} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %reserved = ttl.cb_reserve %destination
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %post = ttl.copy %pipe, %reserved
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      ttl.wait %post : !ttl.receive_request
      ttl.cb_push %destination : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      ttl.yield
    }
    func.return
  }
}

// -----

#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#records = #ttl.pipenet_records<net 0 name "shared-source" mappings
  <graph = <domain = #domain, kind = explicit,
            properties = {edges = [#ttl.transfer_edge<
              source = <coordinates = [0]>,
              destination = <coordinates = [1]>>]}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>,
         <srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0,
          dstEndX = 1, dstEndY = 0>,
         <srcX = 1, srcY = 0, dstStartX = 2, dstStartY = 0,
          dstEndX = 2, dstEndY = 0>,
         <srcX = 2, srcY = 0, dstStartX = 3, dstStartY = 0,
          dstEndX = 3, dstEndY = 0>]>>

// Worker 0 executes two source records in each callback. Sender connections
// remain direct so the transfers retain independent payload slots. One-shot
// receiver records require no readiness manager.
// CHECK-LABEL: module @repeated_local_record attributes
// CHECK-LABEL: func.func @shared_source_sender
// CHECK-SAME: ttl.fabric_routes = [{
// CHECK-SAME: source_nodes = [array<i64: 0, 0>, array<i64: 1, 0>, array<i64: 2, 0>]
// CHECK-LABEL: func.func @shared_source_receiver
// CHECK-NOT: ttl.fabric_manager_intervals
// CHECK-SAME: ttl.fabric_routes = [{
// CHECK-SAME: source_nodes = []
// CHECK-NOT: ttkernel.routing_plane.create_connection_manager
module @repeated_local_record attributes {
  ttl.launch_grid = [4, 1],
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @shared_source_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %send = ttl.copy %source, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.yield
    }
    func.return
  }

  func.func @shared_source_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.pipenet_foreach_dst attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %reserved = ttl.cb_reserve %destination
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %post = ttl.copy %pipe, %reserved
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      ttl.wait %post : !ttl.receive_request
      ttl.cb_push %destination : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      ttl.yield
    }
    func.return
  }
}

// -----

#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#records = #ttl.pipenet_records<net 0 name "staggered" mappings
  <graph = <domain = #domain, kind = explicit,
            properties = {edges = [#ttl.transfer_edge<
              source = <coordinates = [0]>,
              destination = <coordinates = [1]>>]}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>,
         <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0,
          dstEndX = 1, dstEndY = 0>,
         <srcX = 2, srcY = 0, dstStartX = 2, dstStartY = 0,
          dstEndX = 2, dstEndY = 0>,
         <srcX = 3, srcY = 0, dstStartX = 3, dstStartY = 0,
          dstEndX = 3, dstEndY = 0>]>>

// Every worker executes the transfer once, but a different loop iteration
// selects each worker. Sender connections remain direct to avoid a rendezvous
// across different iterations; one-shot receivers need no reverse connection.
// CHECK-LABEL: module @staggered_iterations attributes
// CHECK-NOT: ttl.pipe_sram_scratch_bytes
// CHECK-LABEL: func.func @staggered_sender
// CHECK-SAME: ttl.fabric_routes = [{
// CHECK-SAME: source_nodes = [array<i64: 0, 0>, array<i64: 1, 0>, array<i64: 2, 0>, array<i64: 3, 0>]
// CHECK-LABEL: func.func @staggered_receiver
// CHECK-SAME: ttl.fabric_routes = [{
// CHECK-SAME: source_nodes = []
module @staggered_iterations attributes {
  ttl.launch_grid = [4, 1],
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @staggered_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %node_x = ttl.core_x : index
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %active = arith.cmpi eq, %iteration, %node_x : index
      scf.if %active {
        ttl.pipenet_foreach_src attributes {records = #records} {
        ^bb0(%pipe: !ttl.selected_pipe_src):
          %send = ttl.copy %source, %pipe
              : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
                 !ttl.selected_pipe_src)
              -> !ttl.transfer_handle<write>
          ttl.wait %send : !ttl.transfer_handle<write>
          ttl.yield
        }
      }
    }
    func.return
  }

  func.func @staggered_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %node_x = ttl.core_x : index
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      %active = arith.cmpi eq, %iteration, %node_x : index
      scf.if %active {
        ttl.pipenet_foreach_dst attributes {records = #records} {
        ^bb0(%pipe: !ttl.selected_pipe_dst):
          %reserved = ttl.cb_reserve %destination
              : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
              -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %post = ttl.copy %pipe, %reserved
              : (!ttl.selected_pipe_dst,
                 tensor<1x1x!ttcore.tile<32x32, bf16>>)
              -> !ttl.receive_request
          ttl.wait %post : !ttl.receive_request
          ttl.cb_push %destination
              : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          ttl.yield
        }
      }
    }
    func.return
  }
}

// -----

#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#records = #ttl.pipenet_records<net 0 name "wormhole" mappings
  <graph = <domain = #domain, kind = explicit,
            properties = {edges = [#ttl.transfer_edge<
              source = <coordinates = [0]>,
              destination = <coordinates = [1]>>]}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>,
         <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0,
          dstEndX = 1, dstEndY = 0>,
         <srcX = 2, srcY = 0, dstStartX = 2, dstStartY = 0,
          dstEndX = 2, dstEndY = 0>,
         <srcX = 3, srcY = 0, dstStartX = 3, dstStartY = 0,
          dstEndX = 3, dstEndY = 0>]>>

// Wormhole retains direct fabric connections because the forwarder protocol
// uses Blackhole-only synchronization and routing-plane operations.
// CHECK-LABEL: module @wormhole_direct attributes
// CHECK-NOT: ttl.pipe_sram_scratch_bytes
// CHECK-LABEL: func.func @wormhole_sender
// CHECK-SAME: ttl.fabric_routes = [{
// CHECK-SAME: source_nodes = [array<i64: 0, 0>, array<i64: 1, 0>, array<i64: 2, 0>, array<i64: 3, 0>]
module @wormhole_direct attributes {
  ttl.launch_grid = [4, 1],
  ttl.target_arch = #ttcore.arch<wormhole_b0>
} {
  func.func @wormhole_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %send = ttl.copy %source, %pipe
          : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
             !ttl.selected_pipe_src)
          -> !ttl.transfer_handle<write>
      ttl.wait %send : !ttl.transfer_handle<write>
      ttl.yield
    }
    func.return
  }

  func.func @wormhole_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    ttl.pipenet_foreach_dst attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %reserved = ttl.cb_reserve %destination
          : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %post = ttl.copy %pipe, %reserved
          : (!ttl.selected_pipe_dst,
             tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> !ttl.receive_request
      ttl.wait %post : !ttl.receive_request
      ttl.cb_push %destination : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
      ttl.yield
    }
    func.return
  }
}

// -----

#domain = #ttl.device_domain<components = <name = "device", extent = [2]>>
#records = #ttl.pipenet_records<net 0 name "counter-overflow" mappings
  <graph = <domain = #domain, kind = explicit,
            properties = {edges = [#ttl.transfer_edge<
              source = <coordinates = [0]>,
              destination = <coordinates = [1]>>]}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>,
         <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0,
          dstEndX = 1, dstEndY = 0>,
         <srcX = 2, srcY = 0, dstStartX = 2, dstStartY = 0,
          dstEndX = 2, dstEndY = 0>,
         <srcX = 3, srcY = 0, dstStartX = 3, dstStartY = 0,
          dstEndX = 3, dstEndY = 0>]>>
#outer_records = #ttl.pipenet_records<net 1 name "repeat-twice" mappings
  <graph = <domain = #domain, kind = explicit,
            properties = {edges = [#ttl.transfer_edge<
              source = <coordinates = [0]>,
              destination = <coordinates = [1]>>]}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>,
         <srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0,
          dstEndX = 1, dstEndY = 0>,
         <srcX = 1, srcY = 0, dstStartX = 1, dstStartY = 0,
          dstEndX = 1, dstEndY = 0>,
         <srcX = 1, srcY = 0, dstStartX = 2, dstStartY = 0,
          dstEndX = 2, dstEndY = 0>,
         <srcX = 2, srcY = 0, dstStartX = 2, dstStartY = 0,
          dstEndX = 2, dstEndY = 0>,
         <srcX = 2, srcY = 0, dstStartX = 3, dstStartY = 0,
          dstEndX = 3, dstEndY = 0>,
         <srcX = 3, srcY = 0, dstStartX = 3, dstStartY = 0,
          dstEndX = 3, dstEndY = 0>,
         <srcX = 3, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>]>>

// The outer callback executes each selected fabric record twice. Two
// forwarders split four workers into groups of two, so 2^30 loop iterations
// would increment each arrival counter 2^32 times. Direct connections avoid
// that overflow.
// CHECK-LABEL: module @counter_overflow_direct attributes
// CHECK-NOT: ttl.pipe_sram_scratch_bytes
// CHECK-LABEL: func.func @counter_overflow_sender
// CHECK-SAME: ttl.fabric_routes = [{
// CHECK-SAME: source_nodes = [array<i64: 0, 0>, array<i64: 1, 0>, array<i64: 2, 0>, array<i64: 3, 0>]
// CHECK-LABEL: func.func @counter_overflow_receiver
// CHECK-SAME: ttl.fabric_routes = [{
// CHECK-SAME: source_nodes = [array<i64: 0, 0>, array<i64: 1, 0>, array<i64: 2, 0>, array<i64: 3, 0>]
module @counter_overflow_direct attributes {
  ttl.launch_grid = [4, 1],
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  func.func @counter_overflow_sender()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %source = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 1073741824 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.pipenet_foreach_src attributes {records = #outer_records} {
      ^bb0(%outer_pipe: !ttl.selected_pipe_src):
        ttl.pipenet_foreach_src attributes {records = #records} {
        ^bb0(%pipe: !ttl.selected_pipe_src):
          %send = ttl.copy %source, %pipe
              : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>,
                 !ttl.selected_pipe_src)
              -> !ttl.transfer_handle<write>
          ttl.wait %send : !ttl.transfer_handle<write>
          ttl.yield
        }
        ttl.yield
      }
    }
    func.return
  }

  func.func @counter_overflow_receiver()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    %destination = ttl.bind_cb {cb_index = 1, block_count = 1}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
    %lower = arith.constant 0 : index
    %upper = arith.constant 1073741824 : index
    %step = arith.constant 1 : index
    scf.for %iteration = %lower to %upper step %step {
      ttl.pipenet_foreach_dst attributes {records = #outer_records} {
      ^bb0(%outer_pipe: !ttl.selected_pipe_dst):
        ttl.pipenet_foreach_dst attributes {records = #records} {
        ^bb0(%pipe: !ttl.selected_pipe_dst):
          %reserved = ttl.cb_reserve %destination
              : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
              -> tensor<1x1x!ttcore.tile<32x32, bf16>>
          %post = ttl.copy %pipe, %reserved
              : (!ttl.selected_pipe_dst,
                 tensor<1x1x!ttcore.tile<32x32, bf16>>)
              -> !ttl.receive_request
          ttl.wait %post : !ttl.receive_request
          ttl.cb_push %destination : <[1, 1], !ttcore.tile<32x32, bf16>, 1>
          ttl.yield
        }
        ttl.yield
      }
    }
    func.return
  }
}
