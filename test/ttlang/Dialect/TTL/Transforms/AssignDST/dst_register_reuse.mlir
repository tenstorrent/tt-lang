// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=4}))' --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=8 separate-output-region=1}))' --split-input-file | FileCheck %s --check-prefix=SEPARATE
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=4}))' -debug-only=ttl-assign-dst --split-input-file 2>&1 | FileCheck %s --check-prefix=DEBUG

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: verify copy_tile insertion, dst token + tile results, and that tile
// ops consume the copied tiles with dst_idx annotations.
// DEBUG: Max DST usage: 2 / 4 registers
// CHECK: #[[IDXMAP:.*]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK-LABEL: func.func @simple_add
func.func @simple_add(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                      %b: tensor<2x2x!ttcore.tile<32x32, f32>>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>

  // Bind circular buffers.
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>

  // Attach CBs to tensors.
  %a_cb = ttl.attach_cb %a, %cb0 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_cb = ttl.attach_cb %b, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_cb = ttl.attach_cb %init, %cb2 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // Simple add operation that fits in DST (needs ~3 registers: 2 inputs + 1 result).
// CHECK: ttl.compute
// CHECK: ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT: %[[LINIDX0:.*]] = ttl.linearized_index #{{.*}} : index
// CHECK-NEXT: %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT: %[[DTOK0:.*]], %[[DTILE0:.*]] = ttl.copy_tile %[[A]], %[[LINIDX0]], %[[C0]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NEXT: %[[LINIDX1:.*]] = ttl.linearized_index #{{.*}} : index
// CHECK-NEXT: %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT: %[[DTOK1:.*]], %[[DTILE1:.*]] = ttl.copy_tile %[[B]], %[[LINIDX1]], %[[C1]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NOT:   ttl.copy_tile
// CHECK-NEXT: %[[ADD:.*]] = ttl.tile_add %[[DTILE0]], %[[DTILE1]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// SEPARATE: ttl.tile_add {{.*}} {dst_idx = 2 : i32}
// CHECK-NEXT: ttl.yield %[[ADD]] : !ttcore.tile<32x32, f32>
// CHECK: }
  %result = ttl.compute
      ins(%a_cb, %b_cb : tensor<2x2x!ttcore.tile<32x32, f32>>,
                         tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_cb : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [#map, #map, #map],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%a_tile: !ttcore.tile<32x32, f32>,
       %b_tile: !ttcore.tile<32x32, f32>,
       %out_tile: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %a_tile, %b_tile : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>

  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Capacity is 4.
// We chain 5 adds (3 inputs). With capacity 4, reuse must succeed.
// DEBUG: Max DST usage: 2 / 4 registers

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @chain_reuse
// CHECK: ttl.compute
// CHECK: ^bb0(%[[ARG0:.*]]: !ttcore.tile<32x32, f32>, %[[ARG1:.*]]: !ttcore.tile<32x32, f32>, %[[ARG2:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:   %[[LIN_IDX_0:.*]] = ttl.linearized_index #{{.*}} : index
// CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[DST0:.*]], %[[TILE0:.*]] = ttl.copy_tile %[[ARG0]], %[[LIN_IDX_0]], %[[C0]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[LIN_IDX_1:.*]] = ttl.linearized_index #{{.*}} : index
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[DST1:.*]], %[[TILE1:.*]] = ttl.copy_tile %[[ARG1]], %[[LIN_IDX_1]], %[[C1]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NOT:    ttl.copy_tile
// CHECK-NEXT:   %[[X0:.*]] = ttl.tile_add %[[TILE0]], %[[TILE1]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[LIN_IDX_2:.*]] = ttl.linearized_index #{{.*}} : index
// CHECK-NEXT:   %[[C1_2:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[DST2:.*]], %[[TILE2:.*]] = ttl.copy_tile %[[ARG2]], %[[LIN_IDX_2]], %[[C1_2]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NOT:    ttl.copy_tile
// CHECK-NEXT:   %[[X1:.*]] = ttl.tile_add %[[X0]], %[[TILE2]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[X2:.*]] = ttl.tile_add %[[X1]], %[[TILE2]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[X3:.*]] = ttl.tile_add %[[X2]], %[[TILE2]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[X4:.*]] = ttl.tile_add %[[X3]], %[[TILE2]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// SEPARATE: ttl.tile_add {{.*}} {dst_idx = 2 : i32}
// CHECK-NEXT:   ttl.yield %[[X4]]

func.func @chain_reuse(%i0: tensor<32x32xf32>, %i1: tensor<32x32xf32>,
                       %i2: tensor<32x32xf32>)
    -> tensor<32x32xf32> {
  %init = tensor.empty() : tensor<32x32xf32>

  // Bind CBs (omitted for brevity, just attach)
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>

  %t0 = ttl.attach_cb %i0, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t1 = ttl.attach_cb %i1, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t2 = ttl.attach_cb %i2, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t_init = ttl.attach_cb %init, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>

  %res = ttl.compute
    ins(%t0, %t1, %t2 :
        tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%t_init : tensor<32x32xf32>)
    {indexing_maps = [#map, #map, #map, #map],
     iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>,
       %arg2: !ttcore.tile<32x32, f32>,
       %out: !ttcore.tile<32x32, f32>):

    %x0 = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    %x1 = ttl.tile_add %x0, %arg2 : !ttcore.tile<32x32, f32>
    %x2 = ttl.tile_add %x1, %arg2 : !ttcore.tile<32x32, f32>
    %x3 = ttl.tile_add %x2, %arg2 : !ttcore.tile<32x32, f32>
    %x4 = ttl.tile_add %x3, %arg2 : !ttcore.tile<32x32, f32>

    ttl.yield %x4 : !ttcore.tile<32x32, f32>
  } -> tensor<32x32xf32>

  func.return %res : tensor<32x32xf32>
}

// -----

// Test that multiple uses of the same block argument share a single copy_tile operation.
// DEBUG: Max DST usage: 2 / 4 registers

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @block_arg_multi_use
// CHECK: ttl.compute
// CHECK: ^bb0(%[[ARG0:.*]]: !ttcore.tile<32x32, f32>, %[[ARG1:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NEXT:   %[[LIN_IDX_0:.*]] = ttl.linearized_index #{{.*}} : index
// CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[COPY0TOK:.*]], %[[COPY0:.*]] = ttl.copy_tile %[[ARG0]], %[[LIN_IDX_0]], %[[C0]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[LIN_IDX_1:.*]] = ttl.linearized_index #{{.*}} : index
// CHECK-NEXT:   %[[C1:.*]] = arith.constant 1 : index
// CHECK-NEXT:   %[[COPY1TOK:.*]], %[[COPY1:.*]] = ttl.copy_tile %[[ARG1]], %[[LIN_IDX_1]], %[[C1]] : !ttcore.tile<32x32, f32>, index, index -> !ttl.dst, !ttcore.tile<32x32, f32>
// CHECK-NOT:    ttl.copy_tile
// CHECK-NEXT:   %[[ADD0:.*]] = ttl.tile_add %[[COPY0]], %[[COPY1]] {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[ADD1:.*]] = ttl.tile_add %[[COPY0]], %[[ADD0]] {dst_idx = 1 : i32} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:   %[[ADD2:.*]] = ttl.tile_add %[[COPY0]], %[[ADD1]] {dst_idx = 0 : i32} : !ttcore.tile<32x32, f32>
// SEPARATE: ttl.tile_add {{.*}} {dst_idx = 2 : i32}
// CHECK-NEXT:   ttl.yield %[[ADD2]]

func.func @block_arg_multi_use(%i0: tensor<32x32xf32>, %i1: tensor<32x32xf32>)
    -> tensor<32x32xf32> {
  %init = tensor.empty() : tensor<32x32xf32>

  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>

  %t0 = ttl.attach_cb %i0, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t1 = ttl.attach_cb %i1, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t_init = ttl.attach_cb %init, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>

  %res = ttl.compute
    ins(%t0, %t1 : tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%t_init : tensor<32x32xf32>)
    {indexing_maps = [#map, #map, #map],
     iterator_types = ["parallel", "parallel"]} {
  ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>,
       %out: !ttcore.tile<32x32, f32>):

    // arg0 is used 3 times - should share the same copy_tile
    %x0 = ttl.tile_add %arg0, %arg1 : !ttcore.tile<32x32, f32>
    %x1 = ttl.tile_add %arg0, %x0 : !ttcore.tile<32x32, f32>
    %x2 = ttl.tile_add %arg0, %x1 : !ttcore.tile<32x32, f32>

    ttl.yield %x2 : !ttcore.tile<32x32, f32>
  } -> tensor<32x32xf32>

  func.return %res : tensor<32x32xf32>
}

// -----

// Test SiLU pattern: x * sigmoid(x) where x is a block arg with 2 consumers.
// This requires copy insertion to prevent sigmoid from clobbering x.
// DEBUG: Phase 1: Inserted copy_tile for consumer 0 of block arg
// DEBUG: Max DST usage: 2 / 4 registers

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @silu_pattern
// CHECK: ttl.compute
// CHECK: ^bb0(%[[X:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// Two copy_tile ops for the same block arg (one for sigmoid, one for mul)
// CHECK:       %{{.*}}, %[[XCOPY1:.*]] = ttl.copy_tile %[[X]]
// CHECK:       %{{.*}}, %[[XCOPY2:.*]] = ttl.copy_tile %[[X]]
// CHECK-NOT:   ttl.copy_tile
// CHECK:       %[[SIG:.*]] = ttl.tile_sigmoid %{{.*}} {dst_idx =
// CHECK:       %[[MUL:.*]] = ttl.tile_mul %{{.*}}, %[[SIG]] {dst_idx =
// CHECK:       ttl.yield %[[MUL]]

func.func @silu_pattern(%i0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %init = tensor.empty() : tensor<32x32xf32>

  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>

  %t0 = ttl.attach_cb %i0, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t_init = ttl.attach_cb %init, %cb : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>

  %res = ttl.compute
    ins(%t0 : tensor<32x32xf32>)
    outs(%t_init : tensor<32x32xf32>)
    {indexing_maps = [#map, #map],
     iterator_types = ["parallel", "parallel"]} {
  ^bb0(%x: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
    // x is used by both sigmoid AND mul -> multi-consumer with unary
    // Phase 1 should insert copy_tile so sigmoid doesn't clobber x
    %sig = ttl.tile_sigmoid %x : !ttcore.tile<32x32, f32>
    %prod = ttl.tile_mul %x, %sig : !ttcore.tile<32x32, f32>
    ttl.yield %prod : !ttcore.tile<32x32, f32>
  } -> tensor<32x32xf32>

  func.return %res : tensor<32x32xf32>
}

// -----

// Test: Block arg used by one unary op and two binary ops.
// Pattern:
//   %abs = tile_abs %x    // Unary - overwrites x in-place
//   %add = tile_add %x, %y // Binary uses same x
//   %mul = tile_mul %x, %y // Binary uses same x
//
// The unary op (abs) would clobber x before the binary ops can use it.
// Phase 1 inserts copy_tile only for abs (last unary consumer).
// The binary consumers (add, mul) share a single copy created later.
// Result: 2 copies of x total (one for abs, one shared by add/mul).

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @unary_and_binary_consumers
// CHECK: ttl.compute
// CHECK: ^bb0(%[[X:.*]]: !ttcore.tile<32x32, f32>, %[[Y:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// Two copy_tile ops for x (one for abs, one shared by add/mul), one for y
// CHECK:       %{{.*}}, %[[XCOPY1:.*]] = ttl.copy_tile %[[X]]
// CHECK:       %{{.*}}, %[[XCOPY2:.*]] = ttl.copy_tile %[[X]]
// CHECK-NOT:   ttl.copy_tile %[[X]]
// CHECK:       %[[ABS:.*]] = ttl.tile_abs %{{.*}} {dst_idx =
// CHECK:       %{{.*}}, %[[YCOPY:.*]] = ttl.copy_tile %[[Y]]
// CHECK-NOT:   ttl.copy_tile
// CHECK:       %[[ADD:.*]] = ttl.tile_add %{{.*}}, %{{.*}} {dst_idx =
// CHECK:       %[[MUL:.*]] = ttl.tile_mul %{{.*}}, %{{.*}} {dst_idx =
// CHECK:       ttl.yield

func.func @unary_and_binary_consumers(%i0: tensor<32x32xf32>,
                                       %i1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %init = tensor.empty() : tensor<32x32xf32>

  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
  %cb_out = ttl.bind_cb {cb_index = 16, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>

  %t0 = ttl.attach_cb %i0, %cb0 : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t1 = ttl.attach_cb %i1, %cb1 : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>
  %t_init = ttl.attach_cb %init, %cb_out : (tensor<32x32xf32>, !ttl.cb<[1, 1], f32, 2>) -> tensor<32x32xf32>

  %res = ttl.compute
    ins(%t0, %t1 : tensor<32x32xf32>, tensor<32x32xf32>)
    outs(%t_init : tensor<32x32xf32>)
    {indexing_maps = [#map, #map, #map],
     iterator_types = ["parallel", "parallel"]} {
  ^bb0(%x: !ttcore.tile<32x32, f32>, %y: !ttcore.tile<32x32, f32>,
       %out: !ttcore.tile<32x32, f32>):
    // x is used by abs (unary), add (binary), and mul (binary)
    // Phase 1 inserts copy_tile for abs and add; mul uses original
    %abs = ttl.tile_abs %x : !ttcore.tile<32x32, f32>
    %add = ttl.tile_add %x, %y : !ttcore.tile<32x32, f32>
    %mul = ttl.tile_mul %x, %y : !ttcore.tile<32x32, f32>
    // Combine results
    %tmp = ttl.tile_add %abs, %add : !ttcore.tile<32x32, f32>
    %result = ttl.tile_add %tmp, %mul : !ttcore.tile<32x32, f32>
    ttl.yield %result : !ttcore.tile<32x32, f32>
  } -> tensor<32x32xf32>

  func.return %res : tensor<32x32xf32>
}
