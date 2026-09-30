// RUN: ttlang-opt %s --split-input-file --pass-pipeline='builtin.module(func.func(ttl-lower-topk,ttl-verify-topk-epoch))' | FileCheck %s

// Summary: ttl-lower-topk decisions. stable selects fused keys. largest selects
// the order and the bitonic direction. k = 64 flips that direction. k below
// one tile runs the 32-wide network. Tiles a stage does not update are copied
// from their own column. The row count is the scf.for bound. The sequence
// stays in the surrounding region and is emitted at the earlier result store.
// The kernel fp32_dest_acc_en policy is left untouched.

// CHECK-LABEL: func.func @topk_stable
// CHECK-NOT: fp32_dest_acc_en
// Both key banks record the sort order beside the payload.
// CHECK: ttl.bind_cb{{.*}} {ttl.compiler_allocated, ttl.topk_order = #ttl.topk_order<descending>, ttl.topk_payload = #ttl.topk_payload<fused_keys>}
// CHECK: ttl.bind_cb{{.*}} {ttl.compiler_allocated, ttl.topk_order = #ttl.topk_order<descending>, ttl.topk_payload = #ttl.topk_payload<fused_keys>}
// CHECK: scf.for
// The value transpose keeps the bf16 DST type; its output names the u32 key
// buffer the section packs into.
// CHECK: ttl.tile_transpose {{.*}} : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, u32>) -> !ttcore.tile<32x32, bf16>
// CHECK: ttl.tile_topk_fuse
// CHECK: ttl.tile_topk_local_sort
// CHECK-SAME: fused = true
// CHECK-NOT: ttl.tile_topk_defuse
// CHECK: ttl.tile_topk_merge{{.*}} {fused = true, order = #ttl.topk_order<descending>}
// CHECK-NOT: ttl.tile_topk_fuse
// CHECK-NOT: ttl.tile_topk_defuse
// The one selected tile is rebuilt with skip_second set.
// CHECK: %[[SKIP:.*]] = arith.constant 1 : i32
// CHECK-NEXT: ttl.tile_topk_rebuild{{.*}} skip_second = %[[SKIP]]
// CHECK-SAME: fused = true
// CHECK: ttl.tile_topk_defuse
// CHECK: ttl.tile_transpose
// CHECK: ttl.tile_topk_uint16_move_dest_tile_to_pack_half
// CHECK-NOT: ttl.topk
func.func @topk_stable(%values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
                       %indices: tensor<1x2x!ttcore.tile<32x32, u16>>) {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 1>
  %values_attached = ttl.attach_cb %values, %values_cb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %indices_attached = ttl.attach_cb %indices, %indices_cb
      : (tensor<1x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %values_view = ttl.cb_reserve %out_values_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<1x1x!ttcore.tile<32x32, u16>>
  %out_values, %out_indices = ttl.topk %values_attached, %indices_attached
      k = 32 dim = -1 {stable = true}
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  ttl.store %out_values, %values_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %out_indices, %indices_view
      : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
  ttl.cb_push %out_values_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  ttl.cb_push %out_indices_cb : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 1>
  return
}

// -----

// CHECK-LABEL: func.func @topk_unstable
// CHECK-NOT: fp32_dest_acc_en
// CHECK-NOT: ttl.topk_payload
// CHECK-NOT: ttl.tile_topk_fuse
// CHECK-NOT: ttl.tile_topk_defuse
// CHECK: ttl.tile_topk_local_sort
// CHECK-NOT: ttl.tile_topk_fuse
// CHECK-NOT: ttl.tile_topk_defuse
// Index tiles stay in their own buffer and are copied with the values.
// CHECK: ttl.copy_tile {{.*}}!ttcore.tile<32x32, u16>
// CHECK: ttl.tile_topk_merge
// CHECK: ttl.tile_topk_rebuild
// CHECK: ttl.tile_transpose {{.*}}!ttcore.tile<32x32, u16>
// CHECK-NOT: ttl.tile_topk_uint16_move_dest_tile_to_pack_half
// CHECK-NOT: ttl.tile_topk_defuse
// CHECK-NOT: ttl.topk
func.func @topk_unstable(%values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
                         %indices: tensor<1x2x!ttcore.tile<32x32, u16>>) {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 1>
  %values_attached = ttl.attach_cb %values, %values_cb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %indices_attached = ttl.attach_cb %indices, %indices_cb
      : (tensor<1x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %values_view = ttl.cb_reserve %out_values_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<1x1x!ttcore.tile<32x32, u16>>
  %out_values, %out_indices = ttl.topk %values_attached, %indices_attached
      k = 32 dim = -1
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  ttl.store %out_values, %values_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %out_indices, %indices_view
      : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
  return
}

// -----

// largest = false selects ascending order, a local direction of 1, and a true
// merge direction.
// CHECK-LABEL: func.func @topk_smallest
// CHECK: ttl.bind_cb{{.*}} {ttl.compiler_allocated, ttl.topk_order = #ttl.topk_order<ascending>, ttl.topk_payload = #ttl.topk_payload<fused_keys>}
// CHECK: ttl.bind_cb{{.*}} {ttl.compiler_allocated, ttl.topk_order = #ttl.topk_order<ascending>, ttl.topk_payload = #ttl.topk_payload<fused_keys>}
// CHECK: ttl.tile_topk_fuse{{.*}}order = #ttl.topk_order<ascending>
// CHECK: %[[DIR:.*]] = arith.constant 1 : i32
// CHECK-NEXT: ttl.tile_topk_local_sort{{.*}} direction = %[[DIR]]
// CHECK-SAME: order = #ttl.topk_order<ascending>
// CHECK: ttl.tile_topk_merge{{.*}} {direction = true, fused = true, order = #ttl.topk_order<ascending>}
func.func @topk_smallest(%values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
                         %indices: tensor<1x2x!ttcore.tile<32x32, u16>>) {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 1>
  %values_attached = ttl.attach_cb %values, %values_cb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %indices_attached = ttl.attach_cb %indices, %indices_cb
      : (tensor<1x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %values_view = ttl.cb_reserve %out_values_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<1x1x!ttcore.tile<32x32, u16>>
  %out_values, %out_indices = ttl.topk %values_attached, %indices_attached
      k = 32 dim = -1 {largest = false, stable = true}
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  ttl.store %out_values, %values_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %out_indices, %indices_view
      : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
  return
}

// -----

// k below one tile runs the 32-wide network: end_phase 4, k = 32, logk = 5.
// The sorted result tile holds the requested columns first.
// CHECK-LABEL: func.func @topk_k4_runs_tile_network
// CHECK: %[[END:.*]] = arith.constant 4 : i32
// CHECK: %[[K:.*]] = arith.constant 32 : i32
// CHECK: %[[LOGK:.*]] = arith.constant 5 : i32
// CHECK: %[[DIR:.*]] = arith.constant 0 : i32
// CHECK-NEXT: ttl.tile_topk_local_sort{{.*}} direction = %[[DIR]] end_phase = %[[END]]
// CHECK: ttl.copy_tile {{.*}}[%{{[^,]+}}, %c0{{[_0-9]*}}] into dst[%c0]
// CHECK: ttl.copy_tile {{.*}}[%{{[^,]+}}, %c1{{[_0-9]*}}] into dst[%c1]
// CHECK: ttl.tile_topk_merge{{.*}} k = %[[K]] {fused = true, order = #ttl.topk_order<descending>}
// CHECK: ttl.tile_topk_rebuild{{.*}} k = %[[K]] logk = %[[LOGK]]
func.func @topk_k4_runs_tile_network(%values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
                                 %indices: tensor<1x2x!ttcore.tile<32x32, u16>>) {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 1>
  %values_attached = ttl.attach_cb %values, %values_cb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %indices_attached = ttl.attach_cb %indices, %indices_cb
      : (tensor<1x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %values_view = ttl.cb_reserve %out_values_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<1x1x!ttcore.tile<32x32, u16>>
  %out_values, %out_indices = ttl.topk %values_attached, %indices_attached
      k = 4 dim = -1 {stable = true}
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  ttl.store %out_values, %values_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %out_indices, %indices_view
      : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
  return
}

// -----

// A four-tile row keeps one direction for both local sorts. A column the
// rebuild does not update is copied from that column.
// CHECK-LABEL: func.func @topk_wide_copies_unwritten_column
// CHECK: scf.for
// CHECK: %[[DIR0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: ttl.tile_topk_local_sort{{.*}} direction = %[[DIR0]]
// CHECK: %[[DIR1:.*]] = arith.constant 0 : i32
// CHECK-NEXT: ttl.tile_topk_local_sort{{.*}} direction = %[[DIR1]]
// CHECK: ttl.copy_tile {{.*}}[%{{[^,]+}}, %c3{{[_0-9]*}}] into dst[%c0]
// CHECK: ttl.tile_store {{.*}}[%{{[^,]+}}, %c3{{[_0-9]*}}] from dst[%c0]
func.func @topk_wide_copies_unwritten_column(
    %values: tensor<1x4x!ttcore.tile<32x32, bf16>>,
    %indices: tensor<1x4x!ttcore.tile<32x32, u16>>) {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 1>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 4], !ttcore.tile<32x32, u16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 1>
  %values_attached = ttl.attach_cb %values, %values_cb
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x4x!ttcore.tile<32x32, bf16>>
  %indices_attached = ttl.attach_cb %indices, %indices_cb
      : (tensor<1x4x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 4], !ttcore.tile<32x32, u16>, 1>)
        -> tensor<1x4x!ttcore.tile<32x32, u16>>
  %values_view = ttl.cb_reserve %out_values_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<1x1x!ttcore.tile<32x32, u16>>
  %out_values, %out_indices = ttl.topk %values_attached, %indices_attached
      k = 32 dim = -1 {stable = true}
      : (tensor<1x4x!ttcore.tile<32x32, bf16>>, tensor<1x4x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  ttl.store %out_values, %values_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %out_indices, %indices_view
      : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
  return
}

// -----

// k = 64 flips the local-sort and rebuild direction after each pair. The
// result is two tiles, so defuse runs twice.
// CHECK-LABEL: func.func @topk_k64_flips_direction
// CHECK: %[[END:.*]] = arith.constant 5 : i32
// CHECK: %[[LOGK:.*]] = arith.constant 6 : i32
// CHECK: scf.for
// CHECK: %[[DIR0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: ttl.tile_topk_local_sort{{.*}} direction = %[[DIR0]] end_phase = %[[END]]
// CHECK: %[[DIR1:.*]] = arith.constant 1 : i32
// CHECK-NEXT: ttl.tile_topk_local_sort{{.*}} direction = %[[DIR1]] end_phase = %[[END]]
// CHECK: ttl.tile_topk_rebuild{{.*}} direction = %[[DIR1]]{{.*}} logk = %[[LOGK]]
// CHECK: ttl.tile_topk_defuse
// CHECK: ttl.tile_topk_defuse
func.func @topk_k64_flips_direction(
    %values: tensor<1x8x!ttcore.tile<32x32, bf16>>,
    %indices: tensor<1x8x!ttcore.tile<32x32, u16>>) {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 1>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 8], !ttcore.tile<32x32, u16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>
  %values_attached = ttl.attach_cb %values, %values_cb
      : (tensor<1x8x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x8x!ttcore.tile<32x32, bf16>>
  %indices_attached = ttl.attach_cb %indices, %indices_cb
      : (tensor<1x8x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 8], !ttcore.tile<32x32, u16>, 1>)
        -> tensor<1x8x!ttcore.tile<32x32, u16>>
  %values_view = ttl.cb_reserve %out_values_cb
      : <[1, 2], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 2], !ttcore.tile<32x32, u16>, 1> -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %out_values, %out_indices = ttl.topk %values_attached, %indices_attached
      k = 64 dim = -1 {stable = true}
      : (tensor<1x8x!ttcore.tile<32x32, bf16>>, tensor<1x8x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
  ttl.store %out_values, %values_view
      : tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.store %out_indices, %indices_view
      : tensor<1x2x!ttcore.tile<32x32, u16>>, tensor<1x2x!ttcore.tile<32x32, u16>>
  return
}

// -----

// An eight-tile row runs three merge/rebuild iterations. Merge pairs are
// (tile, tile + distance) with distance doubling per iteration; rebuild pairs
// are the surviving heads of each merged pair. The last rebuild has one
// survivor and sets skip_second.
// CHECK-LABEL: func.func @topk_wide_merge_pairs
// CHECK: scf.for
// Iteration 0 merges (0,1) (2,3) (4,5) (6,7).
// CHECK: ttl.copy_tile {{.*}}%c0{{[_0-9]*}}] into dst[%c0]
// CHECK: ttl.copy_tile {{.*}}%c1{{[_0-9]*}}] into dst[%c1]
// CHECK: %[[IT0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: ttl.tile_topk_merge{{.*}} iteration = %[[IT0]]
// CHECK: ttl.copy_tile {{.*}}%c2{{[_0-9]*}}] into dst[%c0]
// CHECK: ttl.copy_tile {{.*}}%c3{{[_0-9]*}}] into dst[%c1]
// CHECK: ttl.tile_topk_merge
// CHECK: ttl.copy_tile {{.*}}%c4{{[_0-9]*}}] into dst[%c0]
// CHECK: ttl.copy_tile {{.*}}%c5{{[_0-9]*}}] into dst[%c1]
// CHECK: ttl.tile_topk_merge
// CHECK: ttl.copy_tile {{.*}}%c6{{[_0-9]*}}] into dst[%c0]
// CHECK: ttl.copy_tile {{.*}}%c7{{[_0-9]*}}] into dst[%c1]
// CHECK: ttl.tile_topk_merge
// Rebuild 0 pairs the heads (0,2) (4,6).
// CHECK: ttl.copy_tile {{.*}}%c0{{[_0-9]*}}] into dst[%c0]
// CHECK: ttl.copy_tile {{.*}}%c2{{[_0-9]*}}] into dst[%c1]
// CHECK: ttl.tile_topk_rebuild{{.*}} skip_second = %c0_i32{{[_0-9]*}}
// CHECK: ttl.copy_tile {{.*}}%c4{{[_0-9]*}}] into dst[%c0]
// CHECK: ttl.copy_tile {{.*}}%c6{{[_0-9]*}}] into dst[%c1]
// CHECK: ttl.tile_topk_rebuild{{.*}} skip_second = %c0_i32{{[_0-9]*}}
// Iteration 1 merges at distance 2: (0,2) (4,6).
// CHECK: ttl.copy_tile {{.*}}%c0{{[_0-9]*}}] into dst[%c0]
// CHECK: ttl.copy_tile {{.*}}%c2{{[_0-9]*}}] into dst[%c1]
// CHECK: %[[IT1:.*]] = arith.constant 1 : i32
// CHECK-NEXT: ttl.tile_topk_merge{{.*}} iteration = %[[IT1]]
// CHECK: ttl.copy_tile {{.*}}%c4{{[_0-9]*}}] into dst[%c0]
// CHECK: ttl.copy_tile {{.*}}%c6{{[_0-9]*}}] into dst[%c1]
// CHECK: ttl.tile_topk_merge
// Rebuild 1 pairs (0,4).
// CHECK: ttl.copy_tile {{.*}}%c0{{[_0-9]*}}] into dst[%c0]
// CHECK: ttl.copy_tile {{.*}}%c4{{[_0-9]*}}] into dst[%c1]
// CHECK: ttl.tile_topk_rebuild{{.*}} skip_second = %c0_i32{{[_0-9]*}}
// Iteration 2 merges at distance 4: (0,4). Its rebuild has one survivor.
// CHECK: ttl.copy_tile {{.*}}%c0{{[_0-9]*}}] into dst[%c0]
// CHECK: ttl.copy_tile {{.*}}%c4{{[_0-9]*}}] into dst[%c1]
// CHECK: %[[IT2:.*]] = arith.constant 2 : i32
// CHECK-NEXT: ttl.tile_topk_merge{{.*}} iteration = %[[IT2]]
// CHECK: ttl.copy_tile {{.*}}%c0{{[_0-9]*}}] into dst[%c0]
// CHECK: %[[SKIP:.*]] = arith.constant 1 : i32
// CHECK-NEXT: ttl.tile_topk_rebuild{{.*}} skip_second = %[[SKIP]]
// CHECK: ttl.tile_topk_defuse
func.func @topk_wide_merge_pairs(
    %values: tensor<1x8x!ttcore.tile<32x32, bf16>>,
    %indices: tensor<1x8x!ttcore.tile<32x32, u16>>) {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 1>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 8], !ttcore.tile<32x32, u16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 1>
  %values_attached = ttl.attach_cb %values, %values_cb
      : (tensor<1x8x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 8], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x8x!ttcore.tile<32x32, bf16>>
  %indices_attached = ttl.attach_cb %indices, %indices_cb
      : (tensor<1x8x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 8], !ttcore.tile<32x32, u16>, 1>)
        -> tensor<1x8x!ttcore.tile<32x32, u16>>
  %values_view = ttl.cb_reserve %out_values_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<1x1x!ttcore.tile<32x32, u16>>
  %out_values, %out_indices = ttl.topk %values_attached, %indices_attached
      k = 32 dim = -1 {stable = true}
      : (tensor<1x8x!ttcore.tile<32x32, bf16>>, tensor<1x8x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  ttl.store %out_values, %values_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %out_indices, %indices_view
      : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
  return
}

// -----

// The lowering does not rewrite the kernel's fp32_dest_acc_en policy. The
// fused stages carry the requirement, and ttl-set-compute-kernel-config
// diagnoses the conflict with an explicit false.
// CHECK-LABEL: func.func @topk_keeps_fp32_policy
// CHECK-SAME: fp32_dest_acc_en = false
// CHECK: ttl.tile_topk_local_sort
// CHECK-SAME: fused = true
func.func @topk_keeps_fp32_policy(%values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
                                  %indices: tensor<1x2x!ttcore.tile<32x32, u16>>)
    attributes {fp32_dest_acc_en = false} {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 1>
  %values_attached = ttl.attach_cb %values, %values_cb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %indices_attached = ttl.attach_cb %indices, %indices_cb
      : (tensor<1x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %values_view = ttl.cb_reserve %out_values_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<1x1x!ttcore.tile<32x32, u16>>
  %out_values, %out_indices = ttl.topk %values_attached, %indices_attached
      k = 32 dim = -1 {stable = true}
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  ttl.store %out_values, %values_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %out_indices, %indices_view
      : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
  return
}

// -----

// The row loop bound is the tensor height.
// CHECK-LABEL: func.func @topk_row_loop
// CHECK: scf.for %{{.*}} = %{{.*}} to %c3{{[_0-9]*}} step %{{.*}}
func.func @topk_row_loop(%values: tensor<3x2x!ttcore.tile<32x32, bf16>>,
                         %indices: tensor<3x2x!ttcore.tile<32x32, u16>>) {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[3, 2], !ttcore.tile<32x32, bf16>, 1>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[3, 2], !ttcore.tile<32x32, u16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[3, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[3, 1], !ttcore.tile<32x32, u16>, 1>
  %values_attached = ttl.attach_cb %values, %values_cb
      : (tensor<3x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<3x2x!ttcore.tile<32x32, bf16>>
  %indices_attached = ttl.attach_cb %indices, %indices_cb
      : (tensor<3x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[3, 2], !ttcore.tile<32x32, u16>, 1>)
        -> tensor<3x2x!ttcore.tile<32x32, u16>>
  %values_view = ttl.cb_reserve %out_values_cb
      : <[3, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<3x1x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[3, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<3x1x!ttcore.tile<32x32, u16>>
  %out_values, %out_indices = ttl.topk %values_attached, %indices_attached
      k = 32 dim = -1 {stable = true}
      : (tensor<3x2x!ttcore.tile<32x32, bf16>>, tensor<3x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<3x1x!ttcore.tile<32x32, bf16>>, tensor<3x1x!ttcore.tile<32x32, u16>>)
  ttl.store %out_values, %values_view
      : tensor<3x1x!ttcore.tile<32x32, bf16>>, tensor<3x1x!ttcore.tile<32x32, bf16>>
  ttl.store %out_indices, %indices_view
      : tensor<3x1x!ttcore.tile<32x32, u16>>, tensor<3x1x!ttcore.tile<32x32, u16>>
  return
}

// -----

// The generated row loop stays inside the region that contains ttl.topk.
// CHECK-LABEL: func.func @topk_inside_scf_if
// CHECK: scf.if
// CHECK: scf.for
// CHECK: ttl.tile_topk_local_sort
func.func @topk_inside_scf_if(%values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
                              %indices: tensor<1x2x!ttcore.tile<32x32, u16>>,
                              %cond: i1) {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 1>
  %values_attached = ttl.attach_cb %values, %values_cb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %indices_attached = ttl.attach_cb %indices, %indices_cb
      : (tensor<1x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, u16>>
  scf.if %cond {
    %values_view = ttl.cb_reserve %out_values_cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %indices_view = ttl.cb_reserve %out_indices_cb
        : <[1, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<1x1x!ttcore.tile<32x32, u16>>
    %out_values, %out_indices = ttl.topk %values_attached, %indices_attached
        k = 32 dim = -1 {stable = true}
        : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
          -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
    ttl.store %out_values, %values_view
        : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %out_indices, %indices_view
        : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
    scf.yield
  }
  return
}

// -----

// Reserves created after ttl.topk still dominate the generated sequence,
// because the sequence is emitted at the result store.
// CHECK-LABEL: func.func @topk_sequence_follows_result_reserve
// CHECK: arith.constant 7 : i32
// CHECK: ttl.cb_reserve
// CHECK: scf.for
// CHECK: ttl.tile_topk_local_sort
func.func @topk_sequence_follows_result_reserve(
    %values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
    %indices: tensor<1x2x!ttcore.tile<32x32, u16>>) {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 1>
  %values_attached = ttl.attach_cb %values, %values_cb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %indices_attached = ttl.attach_cb %indices, %indices_cb
      : (tensor<1x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %out_values, %out_indices = ttl.topk %values_attached, %indices_attached
      k = 32 dim = -1 {stable = true}
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  %marker = arith.constant 7 : i32
  %values_view = ttl.cb_reserve %out_values_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<1x1x!ttcore.tile<32x32, u16>>
  ttl.store %out_values, %values_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.store %out_indices, %indices_view
      : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
  return
}

// -----

// When the index store precedes the value store, the sequence is emitted
// there. The marker between the stores stays after the sequence.
// CHECK-LABEL: func.func @topk_uses_the_earlier_result_store
// CHECK: ttl.tile_topk_defuse
// CHECK: arith.constant 9 : i32
func.func @topk_uses_the_earlier_result_store(
    %values: tensor<1x2x!ttcore.tile<32x32, bf16>>,
    %indices: tensor<1x2x!ttcore.tile<32x32, u16>>) {
  %values_cb = ttl.bind_cb {cb_index = 0, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
  %indices_cb = ttl.bind_cb {cb_index = 1, block_count = 1}
      : !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>
  %out_values_cb = ttl.bind_cb {cb_index = 2, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %out_indices_cb = ttl.bind_cb {cb_index = 3, block_count = 1}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, u16>, 1>
  %values_attached = ttl.attach_cb %values, %values_cb
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  %indices_attached = ttl.attach_cb %indices, %indices_cb
      : (tensor<1x2x!ttcore.tile<32x32, u16>>, !ttl.cb<[1, 2], !ttcore.tile<32x32, u16>, 1>)
        -> tensor<1x2x!ttcore.tile<32x32, u16>>
  %values_view = ttl.cb_reserve %out_values_cb
      : <[1, 1], !ttcore.tile<32x32, bf16>, 1> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %indices_view = ttl.cb_reserve %out_indices_cb
      : <[1, 1], !ttcore.tile<32x32, u16>, 1> -> tensor<1x1x!ttcore.tile<32x32, u16>>
  %out_values, %out_indices = ttl.topk %values_attached, %indices_attached
      k = 32 dim = -1 {stable = true}
      : (tensor<1x2x!ttcore.tile<32x32, bf16>>, tensor<1x2x!ttcore.tile<32x32, u16>>)
        -> (tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, u16>>)
  ttl.store %out_indices, %indices_view
      : tensor<1x1x!ttcore.tile<32x32, u16>>, tensor<1x1x!ttcore.tile<32x32, u16>>
  %marker = arith.constant 9 : i32
  ttl.store %out_values, %values_view
      : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  return
}
