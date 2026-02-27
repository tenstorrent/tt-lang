// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=4}))' --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=4 separate-output-region=1}))' --split-input-file | FileCheck %s --check-prefix=SEPARATE
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-assign-dst{dst-capacity=4}))' -debug-only=ttl-assign-dst --split-input-file 2>&1 | FileCheck %s --check-prefix=DEBUG

// Verify no placeholder copies remain in final IR (they should all be replaced)
// CHECK-NOT: placeholder

#map = affine_map<(d0, d1) -> (d0, d1)>

// Purpose: verify FPU binary detection for simple add (both operands are block
// args). No copy_tile needed since FPU reads from CB, not DST.
// DEBUG: Max DST usage: 1 / 4 registers
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

  // FPU binary: both operands are block args, no copies needed.
// CHECK:           ttl.compute
// CHECK-NEXT:      ^bb0(%[[A:.*]]: !ttcore.tile<32x32, f32>, %[[B:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// CHECK-NOT:       ttl.copy_tile
// CHECK-NEXT:      %[[ADD:.*]] = ttl.tile_add %[[A]], %[[B]] {dst_idx = 0 : i32, ttl.fpu_binary} : !ttcore.tile<32x32, f32>
// CHECK-NEXT:      ttl.yield %[[ADD]] : !ttcore.tile<32x32, f32>
// SEPARATE:        ttl.tile_add {{.*}} {dst_idx = 0 : i32, ttl.fpu_binary}
// SEPARATE-NEXT:   ttl.yield

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
// We chain 5 adds (3 inputs). First add is FPU (both block args), rest SFPU.
// DEBUG: Max DST usage: 2 / 4 registers

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @chain_reuse
// CHECK:           ttl.compute
// CHECK-NEXT:      ^bb0(%[[ARG0:.*]]: !ttcore.tile<32x32, f32>, %[[ARG1:.*]]: !ttcore.tile<32x32, f32>, %[[ARG2:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// First add is FPU binary (no copies for ARG0/ARG1). ARG2 copied for SFPU adds.
// CHECK-NEXT:      %[[ADD0:.*]] = ttl.tile_add %[[ARG0]], %[[ARG1]] {dst_idx = 0 : i32, ttl.fpu_binary}
// CHECK:           %{{.*}}, %[[COPY:.*]] = ttl.copy_tile %[[ARG2]], %{{.*}}, %{{.*}} {dst_idx = 1 : i32}
// CHECK-NEXT:      %{{.*}} = ttl.tile_add %[[ADD0]], %[[COPY]] {dst_idx = 0 : i32}
// CHECK-NEXT:      %{{.*}} = ttl.tile_add %{{.*}}, %[[COPY]] {dst_idx = 0 : i32}
// CHECK-NEXT:      %{{.*}} = ttl.tile_add %{{.*}}, %[[COPY]] {dst_idx = 0 : i32}
// CHECK-NEXT:      %[[X4:.*]] = ttl.tile_add %{{.*}}, %[[COPY]] {dst_idx = 0 : i32}
// CHECK-NEXT:      ttl.yield %[[X4]]
// SEPARATE-LABEL: func.func @chain_reuse
// SEPARATE:      %[[ADD0S:.*]] = ttl.tile_add {{.*}} {dst_idx = 0 : i32, ttl.fpu_binary}
// SEPARATE:      %{{.*}}, %[[COPYS:.*]] = ttl.copy_tile {{.*}} {dst_idx = 1 : i32}
// SEPARATE-NEXT: %{{.*}} = ttl.tile_add %[[ADD0S]], %[[COPYS]] {dst_idx = 0 : i32}
// With separate-output-region, last add (output) gets dst_idx = 2
// SEPARATE:      %[[X4S:.*]] = ttl.tile_add %{{.*}}, %[[COPYS]] {dst_idx = 2 : i32}
// SEPARATE-NEXT: ttl.yield %[[X4S]]

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

// Test multi-use block arg: first add is FPU (both block args), rest need SFPU.
// arg0 still needs copy_tile for the SFPU adds that use it with DST results.
// DEBUG: Max DST usage: 2 / 4 registers

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @block_arg_multi_use
// CHECK:           ttl.compute
// CHECK-NEXT:      ^bb0(%[[ARG0:.*]]: !ttcore.tile<32x32, f32>, %[[ARG1:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// First add is FPU binary (no copies for ARG0/ARG1).
// CHECK-NEXT:      %[[ADD0:.*]] = ttl.tile_add %[[ARG0]], %[[ARG1]] {dst_idx = 0 : i32, ttl.fpu_binary}
// ARG0 copied for subsequent SFPU adds that use it with DST results.
// CHECK:           %{{.*}}, %[[COPY0:.*]] = ttl.copy_tile %[[ARG0]], %{{.*}}, %{{.*}} {dst_idx = 1 : i32}
// CHECK-NEXT:      %[[ADD1:.*]] = ttl.tile_add %[[COPY0]], %[[ADD0]] {dst_idx = 0 : i32}
// CHECK-NEXT:      %[[ADD2:.*]] = ttl.tile_add %[[COPY0]], %[[ADD1]] {dst_idx = 0 : i32}
// CHECK-NEXT:      ttl.yield %[[ADD2]]
// SEPARATE-LABEL: func.func @block_arg_multi_use
// SEPARATE:      %[[ADD0S:.*]] = ttl.tile_add {{.*}} {dst_idx = 0 : i32, ttl.fpu_binary}
// SEPARATE:      %{{.*}}, %[[COPY0S:.*]] = ttl.copy_tile {{.*}} {dst_idx = 1 : i32}
// SEPARATE-NEXT: %{{.*}} = ttl.tile_add %[[COPY0S]], %[[ADD0S]] {dst_idx = 0 : i32}
// With separate-output-region, last add (output) gets dst_idx = 2
// SEPARATE-NEXT: %[[ADD2S:.*]] = ttl.tile_add %[[COPY0S]], %{{.*}} {dst_idx = 2 : i32}

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

    // arg0 is used 3 times: first add is FPU (both block args),
    // subsequent adds use arg0 with DST results
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
// DEBUG: Phase 1: Inserted placeholder copy_tile for consumer 0 of block arg
// DEBUG: Max DST usage: 2 / 4 registers

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func.func @silu_pattern
// CHECK:           ttl.compute
// CHECK-NEXT:      ^bb0(%[[X:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
// CHECK:           %{{.*}}, %[[XCOPY_SIG:.*]] = ttl.copy_tile %[[X]], %{{.*}}, %{{.*}} {dst_idx = 0 : i32}
// CHECK-NEXT:      %[[SIG:.*]] = ttl.tile_sigmoid %[[XCOPY_SIG]] {dst_idx = 0 : i32}
// CHECK:           %{{.*}}, %[[XCOPY_MUL:.*]] = ttl.copy_tile %[[X]], %{{.*}}, %{{.*}} {dst_idx = 1 : i32}
// CHECK-NEXT:      %[[MUL:.*]] = ttl.tile_mul %[[XCOPY_MUL]], %[[SIG]] {dst_idx = 0 : i32}
// CHECK-NEXT:      ttl.yield %[[MUL]]
// SEPARATE-LABEL: func.func @silu_pattern
// SEPARATE:      %{{.*}}, %[[XCOPY_SIGS:.*]] = ttl.copy_tile {{.*}} {dst_idx = 0 : i32}
// SEPARATE-NEXT: %[[SIGS:.*]] = ttl.tile_sigmoid %[[XCOPY_SIGS]] {dst_idx = 0 : i32}
// SEPARATE:      %{{.*}}, %[[XCOPY_MULS:.*]] = ttl.copy_tile {{.*}} {dst_idx = 1 : i32}
// With separate-output-region, mul (output) gets dst_idx = 2
// SEPARATE-NEXT: %[[MULS:.*]] = ttl.tile_mul %[[XCOPY_MULS]], %[[SIGS]] {dst_idx = 2 : i32}
// SEPARATE-NEXT: ttl.yield %[[MULS]]

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
