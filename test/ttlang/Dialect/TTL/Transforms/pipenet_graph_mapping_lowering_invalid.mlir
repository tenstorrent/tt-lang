// RUN: ttlang-opt %s --convert-ttl-to-ttkernel --verify-diagnostics --split-input-file

// Graph callback lowering validates every launch-node relation before rewriting.

// A graph callback needs a launch grid to interpret its node coordinates.
#records = #ttl.pipenet_records<net 0 mappings
  <graph = <domain = <components = <name = "device", extent = [2]>>,
    kind = all_to_all, componentName = "device", properties = {}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0,
          dstEndX = 1, dstEndY = 0>]>>
module {
  func.func @missing_launch_grid()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{graph PipeNet callback requires a valid ttl.launch_grid with two positive integer extents}}
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      ttl.yield
    }
    func.return
  }
}

// -----

// Endpoint diagnostics include the invalid coordinates and actual grid.
#records = #ttl.pipenet_records<net 0 mappings
  <graph = <domain = <components = <name = "device", extent = [2]>>,
    kind = all_to_all, componentName = "device", properties = {}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 3, dstStartY = 0,
          dstEndX = 3, dstEndY = 0>]>>
module attributes {ttl.launch_grid = array<i64: 3, 2>} {
  func.func @destination_outside_launch_grid()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{PipeNet record 0 has endpoint range core_x=3..3, core_y=0..0 outside the local launch grid (3, 2); increase the launch grid or correct the PipeNet endpoint coordinates}}
    ttl.pipenet_foreach_dst attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_dst):
      ttl.yield
    }
    func.return
  }
}

// -----

// An overflowing grid fails before graph callback IR is changed.
#records = #ttl.pipenet_records<net 0 mappings
  <graph = <domain = <components = <name = "device", extent = [2]>>,
    kind = all_to_all, componentName = "device", properties = {}>,
   pipes[<srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0,
          dstEndX = 1, dstEndY = 0>]>>
module attributes {ttl.launch_grid = array<i64: 9223372036854775807, 2>} {
  func.func @launch_grid_index_overflow()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{graph PipeNet launch-node relation for grid (9223372036854775807, 2) and 1 node-pipe records exceeds the signed 64-bit indexing limit; reduce the launch grid or split the PipeNet}}
    ttl.pipenet_foreach_src attributes {records = #records} {
    ^bb0(%pipe: !ttl.selected_pipe_src):
      ttl.yield
    }
    func.return
  }
}
