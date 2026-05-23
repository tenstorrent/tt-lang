// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(convert-ttl-to-compute),cse,canonicalize)' | FileCheck %s

// Summary: Tests for ttl.block.broadcast lowering to ttl.compute with
// tile_bcast. The block op carries dims/shape attributes; the lowering
// derives the hardware tile-level BcastType (Col=1, Row=2, Scalar=3) from
// whether the innermost two dims are listed in `dims`.

// Row broadcast: (1,N) -> (M,N). Broadcasts row of tiles M times.
// dims=[-2] => Row (the second-innermost dim broadcasts), tile_bcast type 2.
// CHECK-LABEL: func.func @bcast_row
func.func @bcast_row(%arg0: tensor<1x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<1x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x2x!ttcore.tile<32x32, f32>>

  // CHECK: %[[IN:.*]] = ttl.attach_cb %arg0
  // CHECK: %[[OUT:.*]] = ttl.cb_reserve
  // CHECK: %[[INIT:.*]] = ttl.attach_cb {{.*}}
  // CHECK: %[[COMPUTE:.*]] = ttl.compute ins(%[[IN]]
  // CHECK-SAME: outs(%[[INIT]]
  // CHECK-NEXT: ^bb0(%[[IN_TILE:.*]]: !ttcore.tile<32x32, f32>, %[[OUT_TILE:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT: %[[ROW:.*]] = ttl.iter_index 0
  // CHECK-NEXT: %[[COL:.*]] = ttl.iter_index 1
  // CHECK-NEXT: %[[BCASTED:.*]] = ttl.tile_bcast %[[IN_TILE]], %[[OUT_TILE]] 2 : i32
  // CHECK-NEXT: ttl.tile_store %[[BCASTED]], %[[OUT]][%[[ROW]], %[[COL]]] from dst
  // CHECK-NEXT: ttl.yield
  // CHECK: return %[[COMPUTE]]
  %reserve = ttl.cb_reserve %cb1 : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.block.broadcast %arg0_cb dims = [-2], shape = [2, 2] : tensor<1x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %result, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Column broadcast: (M,1) -> (M,N). Broadcasts column of tiles N times.
// dims=[-1] => Col (the innermost dim broadcasts), tile_bcast type 1.
// CHECK-LABEL: func.func @bcast_col
func.func @bcast_col(%arg0: tensor<2x1x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<2x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x1x!ttcore.tile<32x32, f32>>

  // CHECK: %[[IN:.*]] = ttl.attach_cb %arg0
  // CHECK: %[[OUT:.*]] = ttl.cb_reserve
  // CHECK: %[[INIT:.*]] = ttl.attach_cb {{.*}}
  // CHECK: %[[COMPUTE:.*]] = ttl.compute ins(%[[IN]]
  // CHECK-SAME: outs(%[[INIT]]
  // CHECK-NEXT: ^bb0(%[[IN_TILE:.*]]: !ttcore.tile<32x32, f32>, %[[OUT_TILE:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT: %[[ROW:.*]] = ttl.iter_index 0
  // CHECK-NEXT: %[[COL:.*]] = ttl.iter_index 1
  // CHECK-NEXT: %[[BCASTED:.*]] = ttl.tile_bcast %[[IN_TILE]], %[[OUT_TILE]] 1 : i32
  // CHECK-NEXT: ttl.tile_store %[[BCASTED]], %[[OUT]][%[[ROW]], %[[COL]]] from dst
  // CHECK-NEXT: ttl.yield
  // CHECK: return %[[COMPUTE]]
  %reserve = ttl.cb_reserve %cb1 : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.block.broadcast %arg0_cb dims = [-1], shape = [2, 2] : tensor<2x1x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %result, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Scalar broadcast: (1,1) -> (M,N). Broadcasts single tile to all positions.
// dims=[-1, -2] => Scalar, tile_bcast type 3.
// CHECK-LABEL: func.func @bcast_scalar
func.func @bcast_scalar(%arg0: tensor<1x1x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<1x1x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x1x!ttcore.tile<32x32, f32>>

  // CHECK: %[[IN:.*]] = ttl.attach_cb %arg0
  // CHECK: %[[OUT:.*]] = ttl.cb_reserve
  // CHECK: %[[INIT:.*]] = ttl.attach_cb {{.*}}
  // CHECK: %[[COMPUTE:.*]] = ttl.compute ins(%[[IN]]
  // CHECK-SAME: outs(%[[INIT]]
  // CHECK-NEXT: ^bb0(%[[IN_TILE:.*]]: !ttcore.tile<32x32, f32>, %[[OUT_TILE:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT: %[[ROW:.*]] = ttl.iter_index 0
  // CHECK-NEXT: %[[COL:.*]] = ttl.iter_index 1
  // CHECK-NEXT: %[[BCASTED:.*]] = ttl.tile_bcast %[[IN_TILE]], %[[OUT_TILE]] 3 : i32
  // CHECK-NEXT: ttl.tile_store %[[BCASTED]], %[[OUT]][%[[ROW]], %[[COL]]] from dst
  // CHECK-NEXT: ttl.yield
  // CHECK: return %[[COMPUTE]]
  %reserve = ttl.cb_reserve %cb1 : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %result = ttl.block.broadcast %arg0_cb dims = [-1, -2], shape = [2, 2] : tensor<1x1x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %result, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %result : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Broadcast feeding an elementwise op should be handled by fused lowering.
// The broadcast has no direct store user, so standalone broadcast lowering
// defers and buildFusedCompute emits tile_bcast before tile_add.
// CHECK-LABEL: func.func @bcast_row_fused_add
func.func @bcast_row_fused_add(%arg0: tensor<1x2x!ttcore.tile<32x32, f32>>, %arg1: tensor<2x2x!ttcore.tile<32x32, f32>>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %cb2 = ttl.bind_cb {cb_index = 16, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>
  %arg0_cb = ttl.attach_cb %arg0, %cb0 : (tensor<1x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<1x2x!ttcore.tile<32x32, f32>>
  %arg1_cb = ttl.attach_cb %arg1, %cb1 : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>

  // CHECK: %[[BCAST_CB:.*]] = ttl.attach_cb %arg0
  // CHECK: %[[RHS_CB:.*]] = ttl.attach_cb %arg1
  // CHECK: %[[OUT:.*]] = ttl.cb_reserve
  // CHECK: %[[INIT:.*]] = ttl.attach_cb {{.*}}
  // CHECK: %[[COMPUTE:.*]] = ttl.compute ins(%[[BCAST_CB]], %[[RHS_CB]]
  // CHECK-SAME: outs(%[[INIT]]
  // CHECK-NEXT: ^bb0(%[[BCAST_TILE:.*]]: !ttcore.tile<32x32, f32>, %[[RHS_TILE:.*]]: !ttcore.tile<32x32, f32>, %[[OUT_TILE:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT: %[[ROW:.*]] = ttl.iter_index 0
  // CHECK-NEXT: %[[COL:.*]] = ttl.iter_index 1
  // CHECK-NEXT: %[[BCASTED:.*]] = ttl.tile_bcast %[[BCAST_TILE]], %[[OUT_TILE]] 2 : i32
  // CHECK-NEXT: %[[ADDED:.*]] = ttl.tile_add %[[BCASTED]], %[[RHS_TILE]]
  // CHECK-NEXT: ttl.tile_store %[[ADDED]], %[[OUT]][%[[ROW]], %[[COL]]] from dst
  // CHECK-NEXT: ttl.yield
  // CHECK: return %[[COMPUTE]]
  %reserve = ttl.cb_reserve %cb2 : !ttl.cb<[2, 2], !ttcore.tile<32x32, f32>, 2> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %bcast = ttl.block.broadcast %arg0_cb dims = [-2], shape = [2, 2] : tensor<1x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %add = ttl.add %bcast, %arg1_cb : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>> -> tensor<2x2x!ttcore.tile<32x32, f32>>
  ttl.store %add, %reserve : tensor<2x2x!ttcore.tile<32x32, f32>>, tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %add : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// Column reduce -> row broadcast. REDUCE_COL leaves valid data in row 0, so
// the consuming broadcast must use BcastType::Row (2).
// CHECK-LABEL: func.func @bcast_row_after_col_reduce
func.func @bcast_row_after_col_reduce() {
  %inp_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %sc_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %red_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %out_cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>

  %inp_wait = ttl.cb_wait %inp_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %inp_a = ttl.attach_cb %inp_wait, %inp_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %sc_wait = ttl.cb_wait %sc_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sc_a = ttl.attach_cb %sc_wait, %sc_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %red_res = ttl.cb_reserve %red_cb : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %reduced = ttl.reduce %inp_a, %sc_a 0 : i32 [0] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.store %reduced, %red_res : tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %red_cb : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>

  %red_wait = ttl.cb_wait %red_cb : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %red_a = ttl.attach_cb %red_wait, %red_cb : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %out_res = ttl.cb_reserve %out_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  // CHECK: %[[RED_CB:.*]] = ttl.bind_cb{{.*}}cb_index = 2
  // CHECK: %[[OUT_CB:.*]] = ttl.bind_cb{{.*}}cb_index = 3
  // CHECK: ttl.cb_push %[[RED_CB]]
  // CHECK: %[[RED_WAIT:.*]] = ttl.cb_wait %[[RED_CB]]
  // CHECK: %[[RED_IN:.*]] = ttl.attach_cb %[[RED_WAIT]], %[[RED_CB]]
  // CHECK: %[[OUT:.*]] = ttl.cb_reserve %[[OUT_CB]]
  // CHECK: %[[INIT:.*]] = ttl.attach_cb {{.*}}, %[[OUT_CB]]
  // CHECK: %[[COMPUTE:.*]] = ttl.compute ins(%[[RED_IN]]
  // CHECK-SAME: outs(%[[INIT]]
  // CHECK-NEXT: ^bb0(%[[IN_TILE:.*]]: !ttcore.tile<32x32, bf16>, %[[OUT_TILE:.*]]: !ttcore.tile<32x32, bf16>):
  // CHECK-NEXT: %[[ROW:.*]] = ttl.iter_index 0
  // CHECK-NEXT: %[[COL:.*]] = ttl.iter_index 1
  // CHECK-NEXT: %[[BCASTED:.*]] = ttl.tile_bcast %[[IN_TILE]], %[[OUT_TILE]] 2 : i32
  // CHECK-NEXT: ttl.tile_store %[[BCASTED]], %[[OUT]][%[[ROW]], %[[COL]]] from dst
  // CHECK-NEXT: ttl.yield
  // CHECK: ttl.cb_push %[[OUT_CB]]
  %bcast = ttl.block.broadcast %red_a dims = [-2], shape = [2, 2] : tensor<1x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.store %bcast, %out_res : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %out_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %red_cb : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %sc_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %inp_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Row reduce -> column broadcast. REDUCE_ROW leaves valid data in column 0,
// so the consuming broadcast must use BcastType::Col (1).
// CHECK-LABEL: func.func @bcast_col_after_row_reduce
func.func @bcast_col_after_row_reduce() {
  %inp_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %sc_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %red_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>
  %out_cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>

  %inp_wait = ttl.cb_wait %inp_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %inp_a = ttl.attach_cb %inp_wait, %inp_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %sc_wait = ttl.cb_wait %sc_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sc_a = ttl.attach_cb %sc_wait, %sc_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %red_res = ttl.cb_reserve %red_cb : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %reduced = ttl.reduce %inp_a, %sc_a 0 : i32 [1] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  ttl.store %reduced, %red_res : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %red_cb : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>

  %red_wait = ttl.cb_wait %red_cb : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %red_a = ttl.attach_cb %red_wait, %red_cb : (tensor<2x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  %out_res = ttl.cb_reserve %out_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>

  // CHECK: %[[RED_CB:.*]] = ttl.bind_cb{{.*}}cb_index = 2
  // CHECK: %[[OUT_CB:.*]] = ttl.bind_cb{{.*}}cb_index = 3
  // CHECK: ttl.cb_push %[[RED_CB]]
  // CHECK: %[[RED_WAIT:.*]] = ttl.cb_wait %[[RED_CB]]
  // CHECK: %[[RED_IN:.*]] = ttl.attach_cb %[[RED_WAIT]], %[[RED_CB]]
  // CHECK: %[[OUT:.*]] = ttl.cb_reserve %[[OUT_CB]]
  // CHECK: %[[INIT:.*]] = ttl.attach_cb {{.*}}, %[[OUT_CB]]
  // CHECK: %[[COMPUTE:.*]] = ttl.compute ins(%[[RED_IN]]
  // CHECK-SAME: outs(%[[INIT]]
  // CHECK-NEXT: ^bb0(%[[IN_TILE:.*]]: !ttcore.tile<32x32, bf16>, %[[OUT_TILE:.*]]: !ttcore.tile<32x32, bf16>):
  // CHECK-NEXT: %[[ROW:.*]] = ttl.iter_index 0
  // CHECK-NEXT: %[[COL:.*]] = ttl.iter_index 1
  // CHECK-NEXT: %[[BCASTED:.*]] = ttl.tile_bcast %[[IN_TILE]], %[[OUT_TILE]] 1 : i32
  // CHECK-NEXT: ttl.tile_store %[[BCASTED]], %[[OUT]][%[[ROW]], %[[COL]]] from dst
  // CHECK-NEXT: ttl.yield
  // CHECK: ttl.cb_push %[[OUT_CB]]
  %bcast = ttl.block.broadcast %red_a dims = [-1], shape = [2, 2] : tensor<2x1x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.store %bcast, %out_res : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %out_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %red_cb : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %sc_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %inp_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Scalar reduce -> SCALAR broadcast with dims=[-2, -1]. Verifies reduce-dim
// tracing still accepts the derived bcast type matching the producing reduce.
// CHECK-LABEL: func.func @bcast_scalar_after_scalar_reduce
func.func @bcast_scalar_after_scalar_reduce() {
  %inp_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %sc_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %red_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %out_cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>

  %inp_wait = ttl.cb_wait %inp_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %inp_a = ttl.attach_cb %inp_wait, %inp_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %sc_wait = ttl.cb_wait %sc_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sc_a = ttl.attach_cb %sc_wait, %sc_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %red_res = ttl.cb_reserve %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %reduced = ttl.reduce %inp_a, %sc_a 0 : i32 [0, 1] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %reduced, %red_res : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %red_wait = ttl.cb_wait %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %red_a = ttl.attach_cb %red_wait, %red_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_res = ttl.cb_reserve %out_cb : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  // BcastType::Scalar = 3
  // CHECK: %[[RED_CB:.*]] = ttl.bind_cb{{.*}}cb_index = 2
  // CHECK: %[[OUT_CB:.*]] = ttl.bind_cb{{.*}}cb_index = 3
  // CHECK: ttl.cb_push %[[RED_CB]]
  // CHECK: %[[RED_WAIT:.*]] = ttl.cb_wait %[[RED_CB]]
  // CHECK: %[[RED_IN:.*]] = ttl.attach_cb %[[RED_WAIT]], %[[RED_CB]]
  // CHECK: %[[OUT:.*]] = ttl.cb_reserve %[[OUT_CB]]
  // CHECK: %[[INIT:.*]] = ttl.attach_cb {{.*}}, %[[OUT_CB]]
  // CHECK: %[[COMPUTE:.*]] = ttl.compute ins(%[[RED_IN]]
  // CHECK-SAME: outs(%[[INIT]]
  // CHECK-NEXT: ^bb0(%[[IN_TILE:.*]]: !ttcore.tile<32x32, bf16>, %[[OUT_TILE:.*]]: !ttcore.tile<32x32, bf16>):
  // CHECK-NEXT: %[[ROW:.*]] = ttl.iter_index 0
  // CHECK-NEXT: %[[COL:.*]] = ttl.iter_index 1
  // CHECK-NEXT: %[[BCASTED:.*]] = ttl.tile_bcast %[[IN_TILE]], %[[OUT_TILE]] 3 : i32
  // CHECK-NEXT: ttl.tile_store %[[BCASTED]], %[[OUT]][%[[ROW]], %[[COL]]] from dst
  // CHECK-NEXT: ttl.yield
  // CHECK: ttl.cb_push %[[OUT_CB]]
  %bcast = ttl.block.broadcast %red_a dims = [-2, -1], shape = [2, 1] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  ttl.store %bcast, %out_res : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %out_cb : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %sc_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %inp_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  func.return
}

// -----

// Scalar reduce -> SCALAR broadcast with reduce+store inside a nested region
// (ttl.dst_section). Verifies getInputReduceDim traces across nested blocks.
// CHECK-LABEL: func.func @bcast_scalar_after_scalar_reduce_nested
func.func @bcast_scalar_after_scalar_reduce_nested() {
  %inp_cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %sc_cb = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %red_cb = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %out_cb = ttl.bind_cb {cb_index = 3, block_count = 2} : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>

  %inp_wait = ttl.cb_wait %inp_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %inp_a = ttl.attach_cb %inp_wait, %inp_cb : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %sc_wait = ttl.cb_wait %sc_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %sc_a = ttl.attach_cb %sc_wait, %sc_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>

  %red_res = ttl.cb_reserve %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.dst_section {
    %reduced = ttl.reduce %inp_a, %sc_a 0 : i32 [0, 1] : (tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %reduced, %red_res : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield
  }
  ttl.cb_push %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>

  %red_wait = ttl.cb_wait %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %red_a = ttl.attach_cb %red_wait, %red_cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out_res = ttl.cb_reserve %out_cb : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x1x!ttcore.tile<32x32, bf16>>

  // BcastType::Scalar = 3
  // CHECK: %[[RED_CB:.*]] = ttl.bind_cb{{.*}}cb_index = 2
  // CHECK: %[[OUT_CB:.*]] = ttl.bind_cb{{.*}}cb_index = 3
  // CHECK: ttl.dst_section
  // CHECK: ttl.cb_push %[[RED_CB]]
  // CHECK: %[[RED_WAIT:.*]] = ttl.cb_wait %[[RED_CB]]
  // CHECK: %[[RED_IN:.*]] = ttl.attach_cb %[[RED_WAIT]], %[[RED_CB]]
  // CHECK: %[[OUT:.*]] = ttl.cb_reserve %[[OUT_CB]]
  // CHECK: %[[INIT:.*]] = ttl.attach_cb {{.*}}, %[[OUT_CB]]
  // CHECK: %[[COMPUTE:.*]] = ttl.compute ins(%[[RED_IN]]
  // CHECK-SAME: outs(%[[INIT]]
  // CHECK-NEXT: ^bb0(%[[IN_TILE:.*]]: !ttcore.tile<32x32, bf16>, %[[OUT_TILE:.*]]: !ttcore.tile<32x32, bf16>):
  // CHECK-NEXT: %[[ROW:.*]] = ttl.iter_index 0
  // CHECK-NEXT: %[[COL:.*]] = ttl.iter_index 1
  // CHECK-NEXT: %[[BCASTED:.*]] = ttl.tile_bcast %[[IN_TILE]], %[[OUT_TILE]] 3 : i32
  // CHECK-NEXT: ttl.tile_store %[[BCASTED]], %[[OUT]][%[[ROW]], %[[COL]]] from dst
  // CHECK-NEXT: ttl.yield
  // CHECK: ttl.cb_push %[[OUT_CB]]
  %bcast = ttl.block.broadcast %red_a dims = [-2, -1], shape = [2, 1] : tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<2x1x!ttcore.tile<32x32, bf16>>
  ttl.store %bcast, %out_res : tensor<2x1x!ttcore.tile<32x32, bf16>>, tensor<2x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %out_cb : !ttl.cb<[2, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %red_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %sc_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  ttl.cb_pop %inp_cb : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  func.return
}
