// RUN: ttlang-opt %s --convert-ttl-to-ttkernel --verify-diagnostics --split-input-file

// Direct conversion reports invalid local role-query inputs before emitting IR.

// A serialized role query without a launch grid needs a concrete correction.
#records = #ttl.pipenet_records<net 0 pipes [
  <srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0,
   dstEndX = 1, dstEndY = 0>
]>
module {
  func.func @missing_launch_grid()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-error @below {{local PipeNet role query requires a valid ttl.launch_grid with two positive integer extents; set the operation's launch grid to include its PipeNet endpoints}}
    %is_source = ttl.is_src {pipe_net_id = 0 : i64, records = #records}
    "ttl.dprint"(%is_source) {fmt = "source={}", mode = "scalar"} : (i1) -> ()
    func.return
  }
}

// -----

// Destination diagnostics include the invalid coordinates and actual grid.
#records = #ttl.pipenet_records<net 0 pipes [
  <srcX = 0, srcY = 0, dstStartX = 3, dstStartY = 0,
   dstEndX = 3, dstEndY = 0>
]>
module attributes {ttl.launch_grid = array<i64: 3, 2>} {
  func.func @destination_outside_launch_grid()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-error @below {{PipeNet record 0 has endpoint range core_x=3..3, core_y=0..0 outside the local launch grid (3, 2); increase the launch grid or correct the PipeNet endpoint coordinates}}
    %is_destination = ttl.is_dst {pipe_net_id = 0 : i64, records = #records}
    "ttl.dprint"(%is_destination) {fmt = "destination={}", mode = "scalar"} : (i1) -> ()
    func.return
  }
}

// -----

// A grid whose area overflows an index must fail before allocating its table.
#records = #ttl.pipenet_records<net 0 pipes [
  <srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0,
   dstEndX = 1, dstEndY = 0>
]>
module attributes {ttl.launch_grid = array<i64: 9223372036854775807, 2>} {
  func.func @launch_grid_index_overflow()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    // expected-error @below {{local PipeNet table for launch grid (9223372036854775807, 2) and 1 records exceeds the signed 64-bit indexing limit; reduce the launch grid or split the PipeNet}}
    %is_active = ttl.is_active {pipe_net_id = 0 : i64, records = #records}
    "ttl.dprint"(%is_active) {fmt = "active={}", mode = "scalar"} : (i1) -> ()
    func.return
  }
}
