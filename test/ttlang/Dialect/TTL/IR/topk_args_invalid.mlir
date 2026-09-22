// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// Summary: TopK tile ops reject template arguments and step operands that the
// metal API rejects.

func.func @stable_sort_requires_tie_order() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  // expected-error @below {{stable_sort requires an explicit tie_order}}
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start {stable_sort = true}
      : (index, i32, i32, i32) -> ()
  return
}

// -----

func.func @fused_and_stable_are_exclusive() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  // expected-error @below {{fused and stable_sort are mutually exclusive}}
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start
      {stable_sort = true, fused = true,
       tie_order = #ttl.topk_tie_order<ascending>}
      : (index, i32, i32, i32) -> ()
  return
}

// -----

func.func @rank_stamped_and_stable_are_exclusive() {
  %dst = arith.constant 0 : index
  %iter = arith.constant 0 : i32
  %k = arith.constant 32 : i32
  // expected-error @below {{rank_stamped and stable_sort are mutually exclusive}}
  ttl.tile_topk_merge dst[%dst] iteration = %iter k = %k
      {stable_sort = true, rank_stamped = true,
       tie_order = #ttl.topk_tie_order<descending>}
      : (index, i32, i32) -> ()
  return
}

// -----

func.func @rank_stamped_and_fused_are_exclusive() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 32 : i32
  %logk = arith.constant 5 : i32
  %skip = arith.constant 0 : i32
  // expected-error @below {{rank_stamped and fused are mutually exclusive}}
  ttl.tile_topk_rebuild dst[%dst] direction = %dir
      iteration = %iter k = %k logk = %logk skip_second = %skip
      {fused = true, rank_stamped = true}
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}

// -----

func.func @fused_rejects_fp32_dest_acc_disabled() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  // expected-error @below {{fused and rank_stamped modes require fp32 destination accumulation}}
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start
      {fp32_dest_acc_en = false, fused = true}
      : (index, i32, i32, i32) -> ()
  return
}

// -----

func.func @tag_bits_require_rank_stamped() {
  %dst = arith.constant 0 : index
  %iter = arith.constant 0 : i32
  %k = arith.constant 32 : i32
  // expected-error @below {{tag_bits applies only to rank_stamped mode}}
  ttl.tile_topk_merge dst[%dst] iteration = %iter k = %k {tag_bits = 8 : i32}
      : (index, i32, i32) -> ()
  return
}

// -----

func.func @tag_bits_below_range() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  // expected-error @below {{tag_bits must be in the range [6, 16]}}
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start
      {rank_stamped = true, tag_bits = 5 : i32}
      : (index, i32, i32, i32) -> ()
  return
}

// -----

func.func @tag_bits_above_range() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  // expected-error @below {{tag_bits must be in the range [6, 16]}}
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start
      {rank_stamped = true, tag_bits = 17 : i32}
      : (index, i32, i32, i32) -> ()
  return
}

// -----

func.func @start_step_requires_end_step() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 4 : i32
  %start_step = arith.constant 4 : i32
  // expected-error @below {{start_step requires end_step}}
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start start_step = %start_step
      : (index, i32, i32, i32, i32) -> ()
  return
}

// -----

func.func @end_step_out_of_range() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 4 : i32
  %end_step = arith.constant 3 : i32
  // expected-error @below {{end_step must be 0 or in the range [4, 6]}}
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start end_step = %end_step
      : (index, i32, i32, i32, i32) -> ()
  return
}

// -----

func.func @start_step_out_of_range() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 4 : i32
  %end_step = arith.constant 6 : i32
  %start_step = arith.constant 7 : i32
  // expected-error @below {{start_step must be 0 or in the range [4, 6]}}
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start
      end_step = %end_step start_step = %start_step
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}

// -----

func.func @end_phase_out_of_range() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 0 : i32
  %start = arith.constant 0 : i32
  // expected-error @below {{end_phase must be in the range [1, 5]}}
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start
      : (index, i32, i32, i32) -> ()
  return
}

// -----

func.func @start_phase_exceeds_end_phase() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 2 : i32
  %start = arith.constant 3 : i32
  // expected-error @below {{start_phase must not exceed end_phase}}
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start
      : (index, i32, i32, i32) -> ()
  return
}

// -----

func.func @merge_k_not_supported() {
  %dst = arith.constant 0 : index
  %iter = arith.constant 0 : i32
  %k = arith.constant 3 : i32
  // expected-error @below {{k must be one of {4, 8, 16, 32, 64}}}
  ttl.tile_topk_merge dst[%dst] iteration = %iter k = %k
      : (index, i32, i32) -> ()
  return
}

// -----

func.func @merge_iteration_out_of_range() {
  %dst = arith.constant 0 : index
  %iter = arith.constant 10 : i32
  %k = arith.constant 32 : i32
  // expected-error @below {{merge_iteration must be in the range [0, 9]}}
  ttl.tile_topk_merge dst[%dst] iteration = %iter k = %k
      : (index, i32, i32) -> ()
  return
}

// -----

func.func @rebuild_logk_does_not_match_k() {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 32 : i32
  %logk = arith.constant 4 : i32
  %skip = arith.constant 0 : i32
  // expected-error @below {{logk must equal log2(k)}}
  ttl.tile_topk_rebuild dst[%dst] direction = %dir
      iteration = %iter k = %k logk = %logk skip_second = %skip
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}
