// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// Summary: TopK tile ops parse and print template arguments and optional
// step operands. Default arguments stay omitted.

// CHECK-LABEL: func.func @topk_defaults
// CHECK: ttl.tile_topk_local_sort dst[%[[DST:.*]]] direction = %[[DIR:.*]] end_phase = %[[END:.*]] start_phase = %[[START:.*]] : (index, i32, i32, i32) -> ()
// CHECK: ttl.tile_topk_merge dst[%[[DST]]] iteration = %[[ITER:.*]] k = %[[K:.*]] : (index, i32, i32) -> ()
// CHECK: ttl.tile_topk_rebuild dst[%[[DST]]] direction = %[[DIR]] iteration = %[[ITER]] k = %[[K]] logk = %[[LOGK:.*]] skip_second = %[[SKIP:.*]] : (index, i32, i32, i32, i32, i32) -> ()
func.func @topk_defaults() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 32 : i32
  %logk = arith.constant 5 : i32
  %skip = arith.constant 1 : i32
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start
      : (index, i32, i32, i32) -> ()
  ttl.tile_topk_merge dst[%dst] iteration = %iter k = %k
      : (index, i32, i32) -> ()
  ttl.tile_topk_rebuild dst[%dst] direction = %dir
      iteration = %iter k = %k logk = %logk skip_second = %skip
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @topk_stable
// CHECK: ttl.tile_topk_local_sort {{.*}} {stable_sort = true, tie_order = #ttl.topk_tie_order<descending>}
// CHECK: ttl.tile_topk_merge {{.*}} {direction = true, stable_sort = true, tie_order = #ttl.topk_tie_order<ascending>}
// CHECK: ttl.tile_topk_rebuild {{.*}} {fp32_dest_acc_en = true, stable_sort = true, tie_order = #ttl.topk_tie_order<ascending>}
func.func @topk_stable() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 1 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %iter = arith.constant 1 : i32
  %k = arith.constant 32 : i32
  %logk = arith.constant 5 : i32
  %skip = arith.constant 0 : i32
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start
      {stable_sort = true, tie_order = #ttl.topk_tie_order<descending>}
      : (index, i32, i32, i32) -> ()
  ttl.tile_topk_merge dst[%dst] iteration = %iter k = %k
      {stable_sort = true, tie_order = #ttl.topk_tie_order<ascending>,
       direction = true}
      : (index, i32, i32) -> ()
  ttl.tile_topk_rebuild dst[%dst] direction = %dir
      iteration = %iter k = %k logk = %logk skip_second = %skip
      {fp32_dest_acc_en = true, stable_sort = true,
       tie_order = #ttl.topk_tie_order<ascending>}
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @topk_rank_stamped
// CHECK: ttl.tile_topk_local_sort {{.*}} end_step = %[[END_STEP:.*]] start_step = %[[START_STEP:.*]] {rank_stamped = true, tag_bits = 8 : i32}
// CHECK: ttl.tile_topk_merge {{.*}} {fused = true}
// CHECK: ttl.tile_topk_local_sort {{.*}} end_step = %[[ZERO:.*]]
func.func @topk_rank_stamped() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 4 : i32
  %end_step = arith.constant 6 : i32
  %start_step = arith.constant 4 : i32
  %zero = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 32 : i32
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start
      end_step = %end_step start_step = %start_step
      {rank_stamped = true, tag_bits = 8 : i32}
      : (index, i32, i32, i32, i32, i32) -> ()
  ttl.tile_topk_merge dst[%dst] iteration = %iter k = %k {fused = true}
      : (index, i32, i32) -> ()
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start end_step = %zero
      : (index, i32, i32, i32, i32) -> ()
  return
}
