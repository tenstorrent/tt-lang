// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// Summary: TopK tile ops parse and print the required global order, template
// arguments, and optional step operands. Default arguments stay omitted.

// CHECK-LABEL: func.func @topk_defaults
// CHECK: ttl.tile_topk_local_sort dst[%[[DST:.*]]] direction = %[[DIR:.*]] end_phase = %[[END:.*]] start_phase = %[[START:.*]] {order = #ttl.topk_order<descending>} : (index, i32, i32, i32) -> ()
// CHECK: ttl.tile_topk_merge dst[%[[DST]]] iteration = %[[ITER:.*]] k = %[[K:.*]] {order = #ttl.topk_order<descending>} : (index, i32, i32) -> ()
// CHECK: ttl.tile_topk_rebuild dst[%[[DST]]] direction = %[[DIR]] iteration = %[[ITER]] k = %[[K]] logk = %[[LOGK:.*]] skip_second = %[[SKIP:.*]] {order = #ttl.topk_order<descending>} : (index, i32, i32, i32, i32, i32) -> ()
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
      {order = #ttl.topk_order<descending>}
      : (index, i32, i32, i32) -> ()
  ttl.tile_topk_merge dst[%dst] iteration = %iter k = %k
      {order = #ttl.topk_order<descending>}
      : (index, i32, i32) -> ()
  ttl.tile_topk_rebuild dst[%dst] direction = %dir
      iteration = %iter k = %k logk = %logk skip_second = %skip
      {order = #ttl.topk_order<descending>}
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @topk_stable
// CHECK: ttl.tile_topk_local_sort {{.*}} {order = #ttl.topk_order<descending>, stable_sort = true}
// CHECK: ttl.tile_topk_merge {{.*}} {direction = true, order = #ttl.topk_order<ascending>, stable_sort = true}
// CHECK: ttl.tile_topk_rebuild {{.*}} {fp32_dest_acc_en = true, order = #ttl.topk_order<ascending>, stable_sort = true}
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
      {stable_sort = true, order = #ttl.topk_order<descending>}
      : (index, i32, i32, i32) -> ()
  ttl.tile_topk_merge dst[%dst] iteration = %iter k = %k
      {stable_sort = true, order = #ttl.topk_order<ascending>,
       direction = true}
      : (index, i32, i32) -> ()
  ttl.tile_topk_rebuild dst[%dst] direction = %dir
      iteration = %iter k = %k logk = %logk skip_second = %skip
      {fp32_dest_acc_en = true, stable_sort = true,
       order = #ttl.topk_order<ascending>}
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}

// -----

// CHECK-LABEL: func.func @topk_rank_stamped
// CHECK: ttl.tile_topk_local_sort {{.*}} end_step = %[[END_STEP:.*]] start_step = %[[START_STEP:.*]] {order = #ttl.topk_order<descending>, rank_stamped = true, tag_bits = 8 : i32}
// CHECK: ttl.tile_topk_merge {{.*}} {fused = true, order = #ttl.topk_order<descending>}
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
      {order = #ttl.topk_order<descending>,
       rank_stamped = true, tag_bits = 8 : i32}
      : (index, i32, i32, i32, i32, i32) -> ()
  ttl.tile_topk_merge dst[%dst] iteration = %iter k = %k
      {order = #ttl.topk_order<descending>,
       fused = true}
      : (index, i32, i32) -> ()
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start end_step = %zero
      {order = #ttl.topk_order<descending>}
      : (index, i32, i32, i32, i32) -> ()
  return
}
