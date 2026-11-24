// RUN: ttlang-opt %s | FileCheck %s

// Summary: TTKernel TopK ops accept valid operand boundaries and relationships.

// Verify local-sort direction and phase boundaries are accepted.
// CHECK-LABEL: func.func @topk_local_sort_operand_boundaries
// CHECK: ttkernel.topk_local_sort
// CHECK: ttkernel.topk_local_sort
func.func @topk_local_sort_operand_boundaries() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %lower_dir = arith.constant 0 : i32
  %lower_end = arith.constant 1 : i32
  %lower_start = arith.constant 0 : i32
  %upper_dir = arith.constant 1 : i32
  %upper_end = arith.constant 5 : i32
  %upper_start = arith.constant 5 : i32
  ttkernel.topk_local_sort(%dst, %lower_dir, %lower_end, %lower_start)
      : (index, i32, i32, i32) -> ()
  ttkernel.topk_local_sort(%dst, %upper_dir, %upper_end, %upper_start)
      : (index, i32, i32, i32) -> ()
  return
}

// Verify merge iteration and k boundaries are accepted.
// CHECK-LABEL: func.func @topk_merge_operand_boundaries
// CHECK: ttkernel.topk_merge
// CHECK: ttkernel.topk_merge
func.func @topk_merge_operand_boundaries() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %lower_iter = arith.constant 0 : i32
  %lower_k = arith.constant 4 : i32
  %upper_iter = arith.constant 9 : i32
  %upper_k = arith.constant 64 : i32
  ttkernel.topk_merge(%dst, %lower_iter, %lower_k)
      : (index, i32, i32) -> ()
  ttkernel.topk_merge(%dst, %upper_iter, %upper_k)
      : (index, i32, i32) -> ()
  return
}

// Verify rebuild boundaries, log2 consistency, and both skip modes are accepted.
// CHECK-LABEL: func.func @topk_rebuild_operand_boundaries
// CHECK: ttkernel.topk_rebuild
// CHECK: ttkernel.topk_rebuild
func.func @topk_rebuild_operand_boundaries() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 1 : i32
  %iter = arith.constant 9 : i32
  %k = arith.constant 64 : i32
  %logk = arith.constant 6 : i32
  %skip_zero = arith.constant 0 : i32
  %skip_one = arith.constant 1 : i32
  ttkernel.topk_rebuild(%dst, %dir, %iter, %k, %logk, %skip_zero)
      : (index, i32, i32, i32, i32, i32) -> ()
  ttkernel.topk_rebuild(%dst, %dir, %iter, %k, %logk, %skip_one)
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}

// Verify a supported k and its matching log2 value are accepted.
// CHECK-LABEL: func.func @topk_rebuild_matching_logk
// CHECK: ttkernel.topk_rebuild
func.func @topk_rebuild_matching_logk() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 16 : i32
  %logk = arith.constant 4 : i32
  %skip = arith.constant 0 : i32
  ttkernel.topk_rebuild(%dst, %dir, %iter, %k, %logk, %skip)
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}

// Verify skip_second = 1 is accepted with otherwise valid rebuild operands.
// CHECK-LABEL: func.func @topk_rebuild_skip_second_one
// CHECK: ttkernel.topk_rebuild
func.func @topk_rebuild_skip_second_one() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 16 : i32
  %logk = arith.constant 4 : i32
  %skip = arith.constant 1 : i32
  ttkernel.topk_rebuild(%dst, %dir, %iter, %k, %logk, %skip)
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}
