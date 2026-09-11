// RUN: ttlang-opt %s -convert-ttl-to-ttkernel | FileCheck %s --implicit-check-not='array<12xi64>'

// Summary: Verifies every structured graph kind lowers incident-edge
// iteration without an expanded device-edge table.

#axis_records = #ttl.pipenet_records<net 0 mappings
  <graph = <domain = <components = <name = "device", extent = [4]>>,
    kind = axis_neighbor, componentName = "device",
    properties = {axis = 0 : i64, offset = 1 : i64, wrap = false}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>]>>

#stencil_records = #ttl.pipenet_records<net 1 mappings
  <graph = <domain = <components = <name = "device", extent = [2, 2]>>,
    kind = stencil, componentName = "device",
    properties = {offsets = [array<i64: 0, 1>, array<i64: 1, 0>],
                  wrap = false}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>]>>

#gather_records = #ttl.pipenet_records<net 2 mappings
  <graph = <domain = <components = <name = "device", extent = [4]>>,
    kind = gather, componentName = "device",
    properties = {root = #ttl.device_ref<coordinates = [0]>}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>]>>

#scatter_records = #ttl.pipenet_records<net 3 mappings
  <graph = <domain = <components = <name = "device", extent = [4]>>,
    kind = scatter, componentName = "device",
    properties = {source = #ttl.device_ref<coordinates = [0]>}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>]>>

#explicit_records = #ttl.pipenet_records<net 4 mappings
  <graph = <domain = <components = <name = "device", extent = [4]>>,
    kind = explicit, properties = {
      edges = [#ttl.transfer_edge<source = <coordinates = [0]>,
                                  destination = <coordinates = [1]>>,
               #ttl.transfer_edge<source = <coordinates = [2]>,
                                  destination = <coordinates = [3]>>]}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>]>>

#all_to_all_records = #ttl.pipenet_records<net 5 mappings
  <graph = <domain = <components = <name = "device", extent = [4]>>,
    kind = all_to_all, componentName = "device", properties = {}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 0, dstStartY = 0,
          dstEndX = 0, dstEndY = 0>]>>

module attributes {ttl.launch_grid = array<i64: 1, 1>} {
  func.func private @consume(index, index)

  // A non-wrapping axis relation predicates the source loop at the upper
  // boundary and computes the adjacent destination from the current device.
  // CHECK-LABEL: func.func @axis_source
  // CHECK: %[[AXIS_DEVICE_I32:.*]] = ttkernel.get_common_arg_val
  // CHECK: %[[AXIS_DEVICE:.*]] = arith.index_cast %[[AXIS_DEVICE_I32]]
  // CHECK: %[[AXIS_COORD:.*]] = arith.remsi %[[AXIS_DEVICE]], %{{.*}}
  // CHECK: %[[AXIS_NORMALIZED:.*]] = arith.remsi %[[AXIS_COORD]], %{{.*}}
  // CHECK: %[[AXIS_VALID:.*]] = arith.cmpi slt, %[[AXIS_NORMALIZED]], %{{.*}}
  // CHECK: %[[AXIS_COUNT:.*]] = arith.select %[[AXIS_VALID]], %{{.*}}, %{{.*}}
  // CHECK: scf.for %{{.*}} = %{{.*}} to %[[AXIS_COUNT]] step
  // CHECK: func.call @consume(%[[AXIS_DESTINATION:.*]], %[[AXIS_DESTINATION]])
  func.func @axis_source()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    ttl.pipenet_foreach_src attributes {records = #axis_records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %destination = ttl.selected_pipe_destination_device_index %pipe
          : !ttl.selected_pipe_src
      func.call @consume(%destination, %destination) : (index, index) -> ()
      ttl.yield
    }
    func.return
  }

  // Stencil destination iteration derives each possible source, filters
  // invalid boundary offsets, and preserves source-major order.
  // CHECK-LABEL: func.func @stencil_destination
  // CHECK: %[[STENCIL_X:.*]] = ttkernel.get_common_arg_val
  // CHECK: %[[STENCIL_Y:.*]] = ttkernel.get_common_arg_val
  // CHECK: %[[STENCIL_ROW:.*]] = arith.muli %[[STENCIL_X]], %{{.*}}
  // CHECK: %[[STENCIL_DEVICE_I32:.*]] = arith.addi %[[STENCIL_ROW]], %[[STENCIL_Y]]
  // CHECK: %[[STENCIL_DEVICE:.*]] = arith.index_cast %[[STENCIL_DEVICE_I32]]
  // CHECK: arith.cmpi
  // CHECK: arith.andi
  // CHECK: scf.for
  // CHECK: func.call @consume(%[[STENCIL_SOURCE:.*]], %[[STENCIL_SOURCE]])
  func.func @stencil_destination()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    ttl.pipenet_foreach_dst attributes {records = #stencil_records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %source = ttl.selected_pipe_source_device_index %pipe
          : !ttl.selected_pipe_dst
      func.call @consume(%source, %source) : (index, index) -> ()
      ttl.yield
    }
    func.return
  }

  // The gather root iterates the three non-root sources; other devices have
  // no destination callback iterations.
  // CHECK-LABEL: func.func @gather_destination
  // CHECK: %[[GATHER_DEVICE_I32:.*]] = ttkernel.get_common_arg_val
  // CHECK: %[[GATHER_DEVICE:.*]] = arith.index_cast %[[GATHER_DEVICE_I32]]
  // CHECK: %[[IS_ROOT:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}}
  // CHECK: %[[GATHER_COUNT:.*]] = arith.select %[[IS_ROOT]], %{{.*}}, %{{.*}}
  // CHECK: scf.for %{{.*}} = %{{.*}} to %[[GATHER_COUNT]] step
  // CHECK: func.call @consume(%[[GATHER_SOURCE:.*]], %[[GATHER_SOURCE]])
  func.func @gather_destination()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    ttl.pipenet_foreach_dst attributes {records = #gather_records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %source = ttl.selected_pipe_source_device_index %pipe
          : !ttl.selected_pipe_dst
      func.call @consume(%source, %source) : (index, index) -> ()
      ttl.yield
    }
    func.return
  }

  // The scatter source iterates the three non-source destinations; other
  // devices have no source callback iterations.
  // CHECK-LABEL: func.func @scatter_source
  // CHECK: %[[SCATTER_DEVICE_I32:.*]] = ttkernel.get_common_arg_val
  // CHECK: %[[SCATTER_DEVICE:.*]] = arith.index_cast %[[SCATTER_DEVICE_I32]]
  // CHECK: %[[IS_SOURCE:.*]] = arith.cmpi eq, %{{.*}}, %{{.*}}
  // CHECK: %[[SCATTER_COUNT:.*]] = arith.select %[[IS_SOURCE]], %{{.*}}, %{{.*}}
  // CHECK: scf.for %{{.*}} = %{{.*}} to %[[SCATTER_COUNT]] step
  // CHECK: func.call @consume(%[[SCATTER_DESTINATION:.*]], %[[SCATTER_DESTINATION]])
  func.func @scatter_source()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    ttl.pipenet_foreach_src attributes {records = #scatter_records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %destination = ttl.selected_pipe_destination_device_index %pipe
          : !ttl.selected_pipe_src
      func.call @consume(%destination, %destination) : (index, index) -> ()
      ttl.yield
    }
    func.return
  }

  // Explicit graphs use compact device-indexed adjacency tables instead of a
  // device-edge by node-pipe table.
  // CHECK-LABEL: func.func @explicit_source
  // CHECK: %[[EXPLICIT_DEVICE_I32:.*]] = ttkernel.get_common_arg_val
  // CHECK: %[[EXPLICIT_DEVICE:.*]] = arith.index_cast %[[EXPLICIT_DEVICE_I32]]
  // CHECK: ttkernel.experimental.constant_table_lookup %[[EXPLICIT_DEVICE]], [1, 0, 1, 0]
  // CHECK: scf.for
  // CHECK: func.call @consume(%[[EXPLICIT_DESTINATION:.*]], %[[EXPLICIT_DESTINATION]])
  func.func @explicit_source()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    ttl.pipenet_foreach_src attributes {records = #explicit_records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %destination = ttl.selected_pipe_destination_device_index %pipe
          : !ttl.selected_pipe_src
      func.call @consume(%destination, %destination) : (index, index) -> ()
      ttl.yield
    }
    func.return
  }

  // The opposite axis role computes the adjacent source for a destination.
  // CHECK-LABEL: func.func @axis_destination
  // CHECK: scf.for
  // CHECK: func.call @consume(%[[AXIS_SOURCE:.*]], %[[AXIS_SOURCE]])
  func.func @axis_destination()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    ttl.pipenet_foreach_dst attributes {records = #axis_records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %source = ttl.selected_pipe_source_device_index %pipe
          : !ttl.selected_pipe_dst
      func.call @consume(%source, %source) : (index, index) -> ()
      ttl.yield
    }
    func.return
  }

  // The stencil source role computes every valid destination in offset order.
  // CHECK-LABEL: func.func @stencil_source
  // CHECK: scf.for
  // CHECK: func.call @consume(%[[STENCIL_DESTINATION:.*]], %[[STENCIL_DESTINATION]])
  func.func @stencil_source()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    ttl.pipenet_foreach_src attributes {records = #stencil_records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %destination = ttl.selected_pipe_destination_device_index %pipe
          : !ttl.selected_pipe_src
      func.call @consume(%destination, %destination) : (index, index) -> ()
      ttl.yield
    }
    func.return
  }

  // Each non-root gather source has one destination: the declared root.
  // CHECK-LABEL: func.func @gather_source
  // CHECK: scf.for
  // CHECK: func.call @consume(%[[GATHER_DESTINATION:.*]], %[[GATHER_DESTINATION]])
  func.func @gather_source()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    ttl.pipenet_foreach_src attributes {records = #gather_records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %destination = ttl.selected_pipe_destination_device_index %pipe
          : !ttl.selected_pipe_src
      func.call @consume(%destination, %destination) : (index, index) -> ()
      ttl.yield
    }
    func.return
  }

  // Each non-source scatter destination has one source: the declared source.
  // CHECK-LABEL: func.func @scatter_destination
  // CHECK: scf.for
  // CHECK: func.call @consume(%[[SCATTER_SOURCE:.*]], %[[SCATTER_SOURCE]])
  func.func @scatter_destination()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    ttl.pipenet_foreach_dst attributes {records = #scatter_records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %source = ttl.selected_pipe_source_device_index %pipe
          : !ttl.selected_pipe_dst
      func.call @consume(%source, %source) : (index, index) -> ()
      ttl.yield
    }
    func.return
  }

  // All-to-all source and destination roles each iterate the other devices.
  // CHECK-LABEL: func.func @all_to_all_source
  // CHECK: scf.for
  // CHECK: func.call @consume(%[[ALL_DESTINATION:.*]], %[[ALL_DESTINATION]])
  func.func @all_to_all_source()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    ttl.pipenet_foreach_src attributes {records = #all_to_all_records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      %destination = ttl.selected_pipe_destination_device_index %pipe
          : !ttl.selected_pipe_src
      func.call @consume(%destination, %destination) : (index, index) -> ()
      ttl.yield
    }
    func.return
  }

  // CHECK-LABEL: func.func @all_to_all_destination
  // CHECK: scf.for
  // CHECK: func.call @consume(%[[ALL_SOURCE:.*]], %[[ALL_SOURCE]])
  func.func @all_to_all_destination()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    ttl.pipenet_foreach_dst attributes {records = #all_to_all_records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      %source = ttl.selected_pipe_source_device_index %pipe
          : !ttl.selected_pipe_dst
      func.call @consume(%source, %source) : (index, index) -> ()
      ttl.yield
    }
    func.return
  }
}
