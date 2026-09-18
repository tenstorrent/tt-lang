// RUN: ttlang-opt %s --split-input-file -ttl-verify-transfer-provenance | FileCheck %s

// This file tests valid transfer provenance through unreachable and merged
// control flow.

// Provenance validation ignores operations in a statically unreachable loop
// body.
// CHECK-LABEL: func.func @zero_trip_post
func.func @zero_trip_post() {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %transfer = builtin.unrealized_conversion_cast to !ttl.pipe_transfer
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  scf.for %index = %zero to %zero step %one {
    %dst = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %token = ttl.pipe_transfer.post %transfer, %dst
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 0>
  }
  func.return
}

// -----

// A selected token may come from different dynamic posts in the same PipeNet.
// CHECK-LABEL: func.func @selected_same_net_token
func.func @selected_same_net_token(%condition: i1) {
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %transfer = ttl.pipe_transfer.create %pipe {
      kind = #ttl.pipe_transfer_kind<point_to_point>}
      : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
      -> !ttl.pipe_transfer
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %token = scf.if %condition -> (!ttl.pipe_token<net 0>) {
    %dst = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %then_token = ttl.pipe_transfer.post %transfer, %dst
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 0>
    scf.yield %then_token : !ttl.pipe_token<net 0>
  } else {
    %dst = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, f32>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, f32>>
    %else_token = ttl.pipe_transfer.post %transfer, %dst
        : (!ttl.pipe_transfer, tensor<1x1x!ttcore.tile<32x32, f32>>)
        -> !ttl.pipe_token<net 0>
    scf.yield %else_token : !ttl.pipe_token<net 0>
  }
  ttl.pipe_transfer.wait %token : !ttl.pipe_token<net 0>
  func.return
}
