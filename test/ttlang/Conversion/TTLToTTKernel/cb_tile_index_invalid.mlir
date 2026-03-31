// RUN: ttlang-opt --convert-ttl-to-ttkernel --verify-diagnostics --split-input-file %s

// Tests for tile_store index validation in convert-ttl-to-ttkernel.
// tile_store requires pre-materialized multi-dimensional indices from
// ttl-lower-to-loops. These tests verify error handling when indices
// are missing.

// -----

// tile_store with empty indices outside compute body: the AffineLinearizeIndexOp
// folds to constant 0, producing a valid pack_tile at index 0.
module {
  func.func @tile_store_outside_compute(
      %tile: !ttcore.tile<32x32, bf16>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %fake_cb = builtin.unrealized_conversion_cast to !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
    %view = builtin.unrealized_conversion_cast %fake_cb : !ttkernel.cb<4, !ttcore.tile<32x32, bf16>> to tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.tile_store %tile, %view[] : !ttcore.tile<32x32, bf16>, tensor<2x2x!ttcore.tile<32x32, bf16>>
    func.return
  }
}

// -----

// Non-zero lower bound on a tile loop enclosing a tile_bcast
module {
  func.func @nonzero_lb_tile_loop() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %cb_in = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>
    %cb_out = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 1>
    %in = ttl.cb_wait %cb_in : <[2, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    %in_cb = ttl.attach_cb %in, %cb_in : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
    %view = ttl.cb_reserve %cb_out : <[2, 3], !ttcore.tile<32x32, bf16>, 1> -> tensor<2x3x!ttcore.tile<32x32, bf16>>
    %view_cb = ttl.attach_cb %view, %cb_out : (tensor<2x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 1>) -> tensor<2x3x!ttcore.tile<32x32, bf16>>
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    scf.for %row = %c1 to %c2 step %c1 {
      scf.for %col = %c1 to %c3 step %c1 {
        %tile = tensor.extract %in_cb[%row, %c1] : tensor<2x1x!ttcore.tile<32x32, bf16>>
        %out_tile = tensor.extract %view_cb[%row, %col] : tensor<2x3x!ttcore.tile<32x32, bf16>>
        ttl.tile_regs_acquire
        // expected-error @below {{'ttl.tile_bcast' op enclosing tile loop has non-zero lower bound (1)}}
        // expected-error @below {{failed to legalize operation 'ttl.tile_bcast'}}
        %bcast = ttl.tile_bcast %tile, %out_tile 1 : i32
            {dst_idx = 0 : i32, ttl.bcast_output_cb_index = 2 : index}
            : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>)
            -> !ttcore.tile<32x32, bf16>
        ttl.tile_store %bcast, %view[%row, %col] : !ttcore.tile<32x32, bf16>, tensor<2x3x!ttcore.tile<32x32, bf16>>
        ttl.tile_regs_commit
        ttl.tile_regs_wait
        ttl.tile_regs_release
      } {ttl.tile_loop_stride = 1 : index}
    } {ttl.tile_loop_stride = 3 : index}
    func.return
  }
}
