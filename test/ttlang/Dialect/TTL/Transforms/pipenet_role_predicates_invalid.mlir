// RUN: ttlang-opt %s --verify-diagnostics --split-input-file -ttl-verify-pipenet-guards

// Verify role-query record identity and both source and destination grid bounds.

// A source query and its record table must identify the same PipeNet.

#records = #ttl.pipenet_records<net 7 name "mismatched" pipes [
  <srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0,
   dstEndX = 1, dstEndY = 0>
]>

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @mismatched_is_src() {
    // expected-error @below {{PipeNet 7, but pipe_net_id is 3}}
    %predicate = ttl.is_src {pipe_net_id = 3 : i64, records = #records}
    func.return
  }
}

// -----

// Destination queries enforce the same PipeNet identity requirement.

#records = #ttl.pipenet_records<net 7 name "mismatched" pipes [
  <srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0,
   dstEndX = 1, dstEndY = 0>
]>

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @mismatched_is_dst() {
    // expected-error @below {{PipeNet 7, but pipe_net_id is 3}}
    %predicate = ttl.is_dst {pipe_net_id = 3 : i64, records = #records}
    func.return
  }
}

// -----

// Active queries enforce the same PipeNet identity requirement.

#records = #ttl.pipenet_records<net 7 name "mismatched" pipes [
  <srcX = 0, srcY = 0, dstStartX = 1, dstStartY = 0,
   dstEndX = 1, dstEndY = 0>
]>

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @mismatched_is_active() {
    // expected-error @below {{PipeNet 7, but pipe_net_id is 3}}
    %predicate = ttl.is_active {pipe_net_id = 3 : i64, records = #records}
    func.return
  }
}

// -----

// Destination record endpoints must belong to the module launch grid.

#records = #ttl.pipenet_records<net 7 name "outside_grid" pipes [
  <srcX = 0, srcY = 0, dstStartX = 2, dstStartY = 0,
   dstEndX = 2, dstEndY = 0>
]>

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @outside_grid_is_dst() {
    // expected-error @below {{declares destination range core_x=2..2, core_y=0..0 outside the module `ttl.launch_grid`}}
    %predicate = ttl.is_dst {pipe_net_id = 7 : i64, records = #records}
    func.return
  }
}

// -----

// A valid destination cannot hide a source outside the launch grid's Y extent.

#records = #ttl.pipenet_records<net 7 name "source_outside_grid" pipes [
  <srcX = 0, srcY = 2, dstStartX = 1, dstStartY = 0,
   dstEndX = 1, dstEndY = 0>
]>

module attributes {ttl.launch_grid = array<i64: 3, 2>} {
  func.func @outside_grid_is_src() {
    // expected-error @below {{declares source core_x=0, core_y=2 outside the module `ttl.launch_grid`}}
    %is_source = ttl.is_src {pipe_net_id = 7 : i64, records = #records}
    func.return
  }
}
