// RUN: ttlang-opt %s --verify-diagnostics --split-input-file -ttl-verify-pipenet-guards

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
