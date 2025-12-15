// RUN: ttlang-opt --split-input-file --verify-diagnostics %s
// Summary: Test TTL compute operation verifiers (error cases).

// -----

// TileAddOp: operand types must match (different datatypes).
module {
  func.func @tile_add_type_mismatch(%lhs_view: tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                     %rhs_view: tensor<1x1x!ttcore.tile<32x32, f32>>) -> !ttcore.tile<32x32, bf16> {
    %c0 = arith.constant 0 : index
    %lhs = tensor.extract %lhs_view[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %rhs = tensor.extract %rhs_view[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, f32>>
    // expected-error @+1 {{operand types must match}}
    %result = ttl.tile_add %lhs, %rhs : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, bf16>
    return %result : !ttcore.tile<32x32, bf16>
  }
}

// -----

// TileAddOp: result type must match operand type.
module {
  func.func @tile_add_result_mismatch(%lhs_view: tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                       %rhs_view: tensor<1x1x!ttcore.tile<32x32, bf16>>) -> !ttcore.tile<32x32, f32> {
    %c0 = arith.constant 0 : index
    %lhs = tensor.extract %lhs_view[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %rhs = tensor.extract %rhs_view[%c0, %c0] : tensor<1x1x!ttcore.tile<32x32, bf16>>
    // expected-error @+1 {{result type must match operand type}}
    %result = ttl.tile_add %lhs, %rhs : !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, f32>
    return %result : !ttcore.tile<32x32, f32>
  }
}

// -----

// StoreOp: destination element type must be a tile type.
module {
  func.func @store_dest_not_tile(%tile: !ttcore.tile<32x32, bf16>,
                                  %dest: tensor<1x1xf32>) {
    // expected-error @+1 {{destination tensor element type must be a tile type}}
    ttl.store %tile, %dest : !ttcore.tile<32x32, bf16>, tensor<1x1xf32>
    return
  }
}

// -----

// StoreOp: value type must match destination element type.
module {
  func.func @store_type_mismatch(%cb: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
                                  %tile: !ttcore.tile<32x32, bf16>) {
    %c1 = arith.constant 1 : i32
    %view = ttl.cb_reserve %cb, %c1 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
    // expected-error @+1 {{value type must match destination element type}}
    ttl.store %tile, %view : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, f32>>
    return
  }
}
