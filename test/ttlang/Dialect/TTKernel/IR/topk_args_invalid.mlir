// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// Summary: TTKernel TopK ops reject illegal template arguments, step operands,
// and out-of-range local-sort phase operands.

func.func @topk_local_sort_stable_requires_tie_order() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  // expected-error @below {{stable_sort requires an explicit tie_order}}
  ttkernel.topk_local_sort(%dst, %dir, %end, %start) {stable_sort = true}
      : (index, i32, i32, i32) -> ()
  return
}

// -----

func.func @topk_merge_tag_bits_require_rank_stamped() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %iter = arith.constant 0 : i32
  %k = arith.constant 32 : i32
  // expected-error @below {{tag_bits applies only to rank_stamped mode}}
  ttkernel.topk_merge(%dst, %iter, %k) {tag_bits = 8 : i32}
      : (index, i32, i32) -> ()
  return
}

// -----

func.func @topk_rebuild_modes_are_exclusive() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 32 : i32
  %logk = arith.constant 5 : i32
  %skip = arith.constant 0 : i32
  // expected-error @below {{rank_stamped and fused are mutually exclusive}}
  ttkernel.topk_rebuild(%dst, %dir, %iter, %k, %logk, %skip)
      {fused = true, rank_stamped = true}
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}

// -----

func.func @topk_init_tag_bits_require_rank_stamped() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  // expected-error @below {{tag_bits applies only to rank_stamped mode}}
  ttkernel.topk_tile_init() {tag_bits = 8 : i32} : () -> ()
  return
}

// -----

func.func @topk_init_modes_are_exclusive() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  // expected-error @below {{rank_stamped and fused are mutually exclusive}}
  ttkernel.topk_tile_init() {fused = true, rank_stamped = true} : () -> ()
  return
}

// -----

func.func @topk_local_sort_end_step_out_of_range() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 4 : i32
  %end_step = arith.constant 3 : i32
  // expected-error @below {{end_step must be 0 or in the range [4, 6]}}
  ttkernel.topk_local_sort(%dst, %dir, %end, %start, %end_step)
      : (index, i32, i32, i32, i32) -> ()
  return
}

// Reject a direction value above its two-value domain.
// -----

func.func @topk_local_sort_idir_out_of_range() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 2 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  // expected-error @below {{idir must be in the range [0, 1]}}
  ttkernel.topk_local_sort(%dst, %dir, %end, %start)
      : (index, i32, i32, i32) -> ()
  return
}

// Reject an end phase below the supported range.
// -----

func.func @topk_local_sort_end_phase_below_range() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 0 : i32
  %start = arith.constant 0 : i32
  // expected-error @below {{i_end_phase must be in the range [1, 5]}}
  ttkernel.topk_local_sort(%dst, %dir, %end, %start)
      : (index, i32, i32, i32) -> ()
  return
}

// Reject an end phase above the supported range.
// -----

func.func @topk_local_sort_end_phase_above_range() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 6 : i32
  %start = arith.constant 0 : i32
  // expected-error @below {{i_end_phase must be in the range [1, 5]}}
  ttkernel.topk_local_sort(%dst, %dir, %end, %start)
      : (index, i32, i32, i32) -> ()
  return
}

// Reject a start phase below the supported range.
// -----

func.func @topk_local_sort_start_phase_below_range() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant -1 : i32
  // expected-error @below {{i_start_phase must be in the range [0, 5]}}
  ttkernel.topk_local_sort(%dst, %dir, %end, %start)
      : (index, i32, i32, i32) -> ()
  return
}

// Reject a start phase above the supported range.
// -----

func.func @topk_local_sort_start_phase_above_range() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 6 : i32
  // expected-error @below {{i_start_phase must be in the range [0, 5]}}
  ttkernel.topk_local_sort(%dst, %dir, %end, %start)
      : (index, i32, i32, i32) -> ()
  return
}

// Reject a start phase that is greater than the end phase even when both are in range.
// -----

func.func @topk_local_sort_start_phase_exceeds_end_phase() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 5 : i32
  // expected-error @below {{i_start_phase must not exceed i_end_phase}}
  ttkernel.topk_local_sort(%dst, %dir, %end, %start)
      : (index, i32, i32, i32) -> ()
  return
}

// Reject a logk value that does not match log2(k).
// -----

func.func @topk_rebuild_mismatched_logk() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 16 : i32
  %logk = arith.constant 3 : i32
  %skip = arith.constant 0 : i32
  // expected-error @below {{logk must equal log2(k)}}
  ttkernel.topk_rebuild(%dst, %dir, %iter, %k, %logk, %skip)
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}
// Reject a logk value below the supported range.
// -----

func.func @topk_rebuild_logk_below_range() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 4 : i32
  %logk = arith.constant 1 : i32
  %skip = arith.constant 0 : i32
  // expected-error @below {{logk must be in the range [2, 6]}}
  ttkernel.topk_rebuild(%dst, %dir, %iter, %k, %logk, %skip)
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}

// Reject a logk value above the supported range.
// -----

func.func @topk_rebuild_logk_above_range() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 64 : i32
  %logk = arith.constant 7 : i32
  %skip = arith.constant 0 : i32
  // expected-error @below {{logk must be in the range [2, 6]}}
  ttkernel.topk_rebuild(%dst, %dir, %iter, %k, %logk, %skip)
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}

// Reject a k value below the supported set.
// -----

func.func @topk_rebuild_k_below_supported_set() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 3 : i32
  %logk = arith.constant 2 : i32
  %skip = arith.constant 0 : i32
  // expected-error @below {{k must be one of {4, 8, 16, 32, 64}}}
  ttkernel.topk_rebuild(%dst, %dir, %iter, %k, %logk, %skip)
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}

// Reject a k value above the supported set.
// -----

func.func @topk_rebuild_k_above_supported_set() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 128 : i32
  %logk = arith.constant 7 : i32
  %skip = arith.constant 0 : i32
  // expected-error @below {{k must be one of {4, 8, 16, 32, 64}}}
  ttkernel.topk_rebuild(%dst, %dir, %iter, %k, %logk, %skip)
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}

// Reject a merge iteration value below the supported range.
// -----

func.func @topk_merge_m_iter_below_range() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %iter = arith.constant -1 : i32
  %k = arith.constant 16 : i32
  // expected-error @below {{m_iter must be in the range [0, 9]}}
  ttkernel.topk_merge(%dst, %iter, %k)
      : (index, i32, i32) -> ()
  return
}

// Reject a merge iteration value above the supported range.
// -----

func.func @topk_merge_m_iter_above_range() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %iter = arith.constant 10 : i32
  %k = arith.constant 16 : i32
  // expected-error @below {{m_iter must be in the range [0, 9]}}
  ttkernel.topk_merge(%dst, %iter, %k)
      : (index, i32, i32) -> ()
  return
}

// Reject an unsupported merge k value.
// -----

func.func @topk_merge_k_unsupported() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %iter = arith.constant 0 : i32
  %k = arith.constant 3 : i32
  // expected-error @below {{k must be one of {4, 8, 16, 32, 64}}}
  ttkernel.topk_merge(%dst, %iter, %k)
      : (index, i32, i32) -> ()
  return
}

// Reject a rebuild direction value outside its two-value domain.
// -----

func.func @topk_rebuild_idir_out_of_range() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 2 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 16 : i32
  %logk = arith.constant 4 : i32
  %skip = arith.constant 0 : i32
  // expected-error @below {{idir must be in the range [0, 1]}}
  ttkernel.topk_rebuild(%dst, %dir, %iter, %k, %logk, %skip)
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}

// Reject a rebuild iteration value above the supported range.
// -----

func.func @topk_rebuild_m_iter_out_of_range() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %iter = arith.constant 10 : i32
  %k = arith.constant 16 : i32
  %logk = arith.constant 4 : i32
  %skip = arith.constant 0 : i32
  // expected-error @below {{m_iter must be in the range [0, 9]}}
  ttkernel.topk_rebuild(%dst, %dir, %iter, %k, %logk, %skip)
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}

// Reject a skip_second value outside its two-value domain.
// -----

func.func @topk_rebuild_skip_second_out_of_range() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 16 : i32
  %logk = arith.constant 4 : i32
  %skip = arith.constant 2 : i32
  // expected-error @below {{skip_second must be in the range [0, 1]}}
  ttkernel.topk_rebuild(%dst, %dir, %iter, %k, %logk, %skip)
      : (index, i32, i32, i32, i32, i32) -> ()
  return
}

// -----

func.func @topk_defuse_num_tiles_out_of_range() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %num_tiles = arith.constant 0 : i32
  // expected-error @below {{num_tiles must be in the range [1, 2]}}
  ttkernel.topk_defuse_tile(%dst, %num_tiles) : (index, i32) -> ()
  return
}

// -----

func.func @topk_stamp_tag_bits_out_of_range() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  // expected-error @below {{tag_bits must be in the range [6, 16]}}
  ttkernel.topk_stamp_local_positions(%dst) {tag_bits = 17 : i32}
      : (index) -> ()
  return
}

// -----

func.func @topk_strip_rejects_fp32_false() attributes {
    ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  // expected-error @below {{strip_rank_tags requires fp32 destination accumulation}}
  ttkernel.topk_strip_rank_tags(%dst) {fp32_dest_acc_en = false} : (index) -> ()
  return
}
