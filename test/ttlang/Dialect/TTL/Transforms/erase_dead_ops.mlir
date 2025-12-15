// RUN: ttlang-opt --ttl-erase-dead-ops --split-input-file %s | FileCheck %s
// Summary: Test ttl-erase-dead-ops pass erases dead conversion artifacts.

// -----

// Dead ttcore.get_global ops are erased.
// CHECK-LABEL: func.func @dead_get_global()
// CHECK-NOT: ttcore.get_global
// CHECK: return
#l1_1 = #ttnn.buffer_type<l1>
#ttnn_layout_1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1_1>, <height_sharded>>
module {
  ttcore.global @tensor = tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_1> [0]
  func.func @dead_get_global() {
    %unused = ttcore.get_global @tensor : tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_1>
    return
  }
}

// -----

// Dead unrealized_conversion_cast ops are erased.
// CHECK-LABEL: func.func @dead_cast()
// CHECK-NOT: unrealized_conversion_cast
// CHECK: return
module {
  func.func @dead_cast() {
    %c0 = arith.constant 0 : i32
    %unused = builtin.unrealized_conversion_cast %c0 : i32 to !ttl.cb<[1, 1], f32, 2>
    return
  }
}

// -----

// Cascading dead ops: get_global feeding dead cast is erased.
// CHECK-LABEL: func.func @cascading_dead_ops()
// CHECK-NOT: ttcore.get_global
// CHECK-NOT: unrealized_conversion_cast
// CHECK: arith.constant 42
// CHECK: return
#l1_3 = #ttnn.buffer_type<l1>
#ttnn_layout_3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1_3>, <height_sharded>>
module {
  ttcore.global @tensor = tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_3> [0]
  func.func @cascading_dead_ops() {
    %t = ttcore.get_global @tensor : tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_3>
    %dead_cast = builtin.unrealized_conversion_cast %t : tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_3> to i32
    %live = arith.constant 42 : i32
    return
  }
}

// -----

// Live ops are preserved.
// CHECK-LABEL: func.func @live_ops
// CHECK: %[[T:.*]] = ttcore.get_global @tensor
// CHECK: return %[[T]]
#l1_4 = #ttnn.buffer_type<l1>
#ttnn_layout_4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1_4>, <height_sharded>>
module {
  ttcore.global @tensor = tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_4> [0]
  func.func @live_ops() -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_4> {
    %t = ttcore.get_global @tensor : tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_4>
    return %t : tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_4>
  }
}

// -----

// Mixed live and dead: only dead ops erased.
// CHECK-LABEL: func.func @mixed_live_dead
// CHECK: %[[LIVE:.*]] = ttcore.get_global @live_tensor
// CHECK-NOT: ttcore.get_global @dead_tensor
// CHECK: return %[[LIVE]]
#l1_5 = #ttnn.buffer_type<l1>
#ttnn_layout_5 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1_5>, <height_sharded>>
module {
  ttcore.global @live_tensor = tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_5> [0]
  ttcore.global @dead_tensor = tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_5> [1]
  func.func @mixed_live_dead() -> tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_5> {
    %live = ttcore.get_global @live_tensor : tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_5>
    %dead = ttcore.get_global @dead_tensor : tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_5>
    return %live : tensor<1x1x!ttcore.tile<32x32, bf16>, #ttnn_layout_5>
  }
}

// -----

// Multiple iterations: chain of dead casts.
// CHECK-LABEL: func.func @chain_of_dead_casts()
// CHECK-NOT: unrealized_conversion_cast
// CHECK: return
module {
  func.func @chain_of_dead_casts() {
    %c0 = arith.constant 0 : i32
    %cast1 = builtin.unrealized_conversion_cast %c0 : i32 to !ttl.cb<[1, 1], f32, 2>
    %cast2 = builtin.unrealized_conversion_cast %cast1 : !ttl.cb<[1, 1], f32, 2> to i64
    %cast3 = builtin.unrealized_conversion_cast %cast2 : i64 to f32
    return
  }
}
