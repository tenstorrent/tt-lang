// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s

// Summary: Verifies one graph relation is planned independently for each
// enclosing launch grid.

#records = #ttl.pipenet_records<net 0 mappings
  <graph = <domain = <components = <name = "device", extent = [2]>>,
    kind = all_to_all, componentName = "device", properties = {}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>]>>

module {
  module @single_node attributes {ttl.launch_grid = array<i64: 1, 1>} {
    func.func private @consume(index)

    // A 1x1 grid makes the declared node pipe an identity relation, so every
    // incident edge executes without a node-coordinate condition.
    // CHECK-LABEL: func.func @single_node_source
    // CHECK: scf.for
    // CHECK-NOT: scf.if
    // CHECK: func.call @consume
    func.func @single_node_source()
        attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
      ttl.pipenet_foreach_src attributes {records = #records} {
      ^bb0(%pipe: !ttl.selected_pipe_src):
        %destination = ttl.selected_pipe_destination_device_index %pipe
            : !ttl.selected_pipe_src
        func.call @consume(%destination) : (index) -> ()
        ttl.yield
      }
      func.return
    }
  }

  module @two_nodes attributes {ttl.launch_grid = array<i64: 2, 1>} {
    func.func private @consume(index)

    // The same node pipe covers only x=0 in a 2x1 grid, so lowering retains
    // the node-coordinate condition for this operation.
    // CHECK-LABEL: func.func @two_node_source
    // CHECK: scf.for
    // CHECK: scf.if
    // CHECK: func.call @consume
    func.func @two_node_source()
        attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
      ttl.pipenet_foreach_src attributes {records = #records} {
      ^bb0(%pipe: !ttl.selected_pipe_src):
        %destination = ttl.selected_pipe_destination_device_index %pipe
            : !ttl.selected_pipe_src
        func.call @consume(%destination) : (index) -> ()
        ttl.yield
      }
      func.return
    }
  }
}
