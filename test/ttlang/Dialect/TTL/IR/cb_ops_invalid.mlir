// RUN: ttlang-opt --verify-diagnostics --split-input-file %s

// -----

// cb_reserve result shape must match CB shape.
module {
  func.func @cb_reserve_shape_mismatch(%cb: !ttl.cb<[1, 1], f32, 2>) -> tensor<2x2xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{result tensor shape dimension 0 (2) must match CB shape dimension (1)}}
    %view = ttl.cb_reserve %cb : <[1, 1], f32, 2> -> tensor<2x2xf32>
    func.return %view : tensor<2x2xf32>
  }
}

// -----

// cb_reserve result element type must match CB element type.
module {
  func.func @cb_reserve_element_mismatch(%cb: !ttl.cb<[1, 1], f32, 2>) -> tensor<1x1xf16> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{result tensor element type ('f16') must match CB element type ('f32')}}
    %view = ttl.cb_reserve %cb : <[1, 1], f32, 2> -> tensor<1x1xf16>
    func.return %view : tensor<1x1xf16>
  }
}

// -----

// cb_reserve result rank must match CB shape rank.
module {
  func.func @cb_reserve_rank_mismatch(%cb: !ttl.cb<[1, 1], f32, 2>) -> tensor<1xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{result tensor rank (1) must match CB shape rank (2)}}
    %view = ttl.cb_reserve %cb : <[1, 1], f32, 2> -> tensor<1xf32>
    func.return %view : tensor<1xf32>
  }
}

// -----

// cb_wait result shape must match CB shape.
module {
  func.func @cb_wait_shape_mismatch(%cb: !ttl.cb<[1, 1], f32, 2>) -> tensor<2x2xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{result tensor shape dimension 0 (2) must match CB shape dimension (1)}}
    %view = ttl.cb_wait %cb : <[1, 1], f32, 2> -> tensor<2x2xf32>
    func.return %view : tensor<2x2xf32>
  }
}

// -----

// cb_wait result element type must match CB element type.
module {
  func.func @cb_wait_element_mismatch(%cb: !ttl.cb<[1, 1], f32, 2>) -> tensor<1x1xbf16> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{result tensor element type ('bf16') must match CB element type ('f32')}}
    %view = ttl.cb_wait %cb : <[1, 1], f32, 2> -> tensor<1x1xbf16>
    func.return %view : tensor<1x1xbf16>
  }
}

// -----

// cb_wait result rank must match CB shape rank.
module {
  func.func @cb_wait_rank_mismatch(%cb: !ttl.cb<[1, 1], f32, 2>) -> tensor<1x1x1xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{result tensor rank (3) must match CB shape rank (2)}}
    %view = ttl.cb_wait %cb : <[1, 1], f32, 2> -> tensor<1x1x1xf32>
    func.return %view : tensor<1x1x1xf32>
  }
}

// -----

// cb_reserve with tile element type, wrong result element type.
module {
  func.func @cb_reserve_tile_element_mismatch(%cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{result tensor element type ('!ttcore.tile<32x32, f32>') must match CB element type ('!ttcore.tile<32x32, bf16>')}}
    %view = ttl.cb_reserve %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    func.return %view : tensor<1x1x!ttcore.tile<32x32, f32>>
  }
}

// -----

// cb_reserve second dimension mismatch.
module {
  func.func @cb_reserve_shape_dim1_mismatch(%cb: !ttl.cb<[2, 3], f32, 2>) -> tensor<2x4xf32> attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
    // expected-error @below {{result tensor shape dimension 1 (4) must match CB shape dimension (3)}}
    %view = ttl.cb_reserve %cb : <[2, 3], f32, 2> -> tensor<2x4xf32>
    func.return %view : tensor<2x4xf32>
  }
}

// -----

// ttl.store tile operand must be !ttcore.tile.
module {
  func.func @store_non_tile(%val: f32, %view: tensor<1x1xf32>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    // expected-error @below {{tile operand must be !ttcore.tile, got 'f32'}}
    ttl.store %val, %view[%c0] : f32, tensor<1x1xf32>
    func.return
  }
}

// -----

// ttl.store view element type must match tile type.
module {
  func.func @store_type_mismatch(%tile: !ttcore.tile<32x32, bf16>, %view: tensor<1x1x!ttcore.tile<32x32, f32>>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    // expected-error @below {{view element type ('!ttcore.tile<32x32, f32>') must match tile type ('!ttcore.tile<32x32, bf16>')}}
    ttl.store %tile, %view[%c0] : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, f32>>
    func.return
  }
}
