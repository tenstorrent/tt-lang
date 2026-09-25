// RUN: ttlang-opt --convert-ttl-to-ttkernel --ttkernel-insert-inits %s -o %t.ttkernel.mlir
// RUN: FileCheck %s --input-file=%t.ttkernel.mlir --check-prefix=TTKERNEL
// RUN: ttlang-opt --convert-ttkernel-to-emitc %t.ttkernel.mlir -o %t.emitc.mlir
// RUN: ttlang-translate --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// Summary: TopK template arguments and optional steps survive lowering into
// kernel C++. Trailing metal defaults stay omitted, and a changed init
// configuration is re-initialized.

// TTKERNEL-LABEL: func.func @topk_stable
// TTKERNEL: ttkernel.topk_tile_init()
// TTKERNEL: ttkernel.topk_local_sort({{[^)]*}}) {{[{].*}}stable_sort = true{{.*}}tie_order = #ttkernel.topk_tie_order<ascending>
// TTKERNEL: ttkernel.topk_merge({{[^)]*}}) {{[{].*}}direction = true, stable_sort = true, tie_order = #ttkernel.topk_tie_order<ascending>
// TTKERNEL: ttkernel.topk_rebuild({{[^)]*}}) {{[{].*}}stable_sort = true{{.*}}tie_order = #ttkernel.topk_tie_order<ascending>
// CPP-LABEL: void kernel_main()
// CPP: topk_tile_init();
// CPP: topk_local_sort<true, DST_ACCUM_MODE, false, false, TopkTieOrder::Ascending>(
// CPP: topk_merge<true, true, DST_ACCUM_MODE, false, false, TopkTieOrder::Ascending>(
// CPP: topk_rebuild<true, DST_ACCUM_MODE, false, false, TopkTieOrder::Ascending>(
func.func @topk_stable() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 1 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 32 : i32
  %logk = arith.constant 5 : i32
  %skip = arith.constant 0 : i32
  ttl.tile_regs_acquire
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start
      {stable_sort = true, tie_order = #ttl.topk_tie_order<ascending>}
      : (index, i32, i32, i32) -> ()
  ttl.tile_topk_merge dst[%dst] iteration = %iter k = %k
      {stable_sort = true, tie_order = #ttl.topk_tie_order<ascending>,
       direction = true}
      : (index, i32, i32) -> ()
  ttl.tile_topk_rebuild dst[%dst] direction = %dir
      iteration = %iter k = %k logk = %logk skip_second = %skip
      {stable_sort = true, tie_order = #ttl.topk_tie_order<ascending>}
      : (index, i32, i32, i32, i32, i32) -> ()
  ttl.tile_regs_release
  return
}

// TTKERNEL-LABEL: func.func @topk_fused_and_steps
// TTKERNEL: ttkernel.topk_tile_init() {fused = true}
// TTKERNEL: ttkernel.topk_local_sort(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) {{[{].*}}fused = true
// TTKERNEL: ttkernel.topk_merge({{[^)]*}}) {{[{].*}}fused = true
// CPP-LABEL: void kernel_main()
// CPP: topk_tile_init<true>();
// CPP: topk_local_sort<false, DST_ACCUM_MODE, true>(
// CPP: topk_merge<false, false, DST_ACCUM_MODE, true>(
func.func @topk_fused_and_steps() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 4 : i32
  %end_step = arith.constant 6 : i32
  %start_step = arith.constant 4 : i32
  %iter = arith.constant 2 : i32
  %k = arith.constant 64 : i32
  ttl.tile_regs_acquire
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start
      end_step = %end_step start_step = %start_step {fused = true}
      : (index, i32, i32, i32, i32, i32) -> ()
  ttl.tile_topk_merge dst[%dst] iteration = %iter k = %k {fused = true}
      : (index, i32, i32) -> ()
  ttl.tile_regs_release
  return
}

// An explicit fp32_dest_acc_en is printed even when it is false. Later
// default template arguments stay omitted.
// TTKERNEL-LABEL: func.func @topk_fp32_disabled
// TTKERNEL: ttkernel.topk_local_sort({{[^)]*}}) {{[{].*}}fp32_dest_acc_en = false
// CPP-LABEL: void kernel_main()
// CPP: topk_local_sort<false, false>(
func.func @topk_fp32_disabled() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  ttl.tile_regs_acquire
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start {fp32_dest_acc_en = false}
      : (index, i32, i32, i32) -> ()
  ttl.tile_regs_release
  return
}

// TTKERNEL-LABEL: func.func @topk_rank_stamped
// TTKERNEL: ttkernel.topk_tile_init() {{[{].*}}rank_stamped = true{{.*}}tag_bits = 8 : i32
// TTKERNEL: ttkernel.topk_local_sort({{[^)]*}}) {{[{].*}}rank_stamped = true{{.*}}tag_bits = 8 : i32
// TTKERNEL: ttkernel.topk_rebuild({{[^)]*}}) {{[{].*}}rank_stamped = true{{.*}}tag_bits = 8 : i32
// TTKERNEL: ttkernel.topk_merge({{[^)]*}}) {{[{].*}}rank_stamped = true{{.*}}tag_bits = 8 : i32
// CPP-LABEL: void kernel_main()
// CPP: topk_tile_init<false, true, 8>();
// CPP: topk_local_sort<false, DST_ACCUM_MODE, false, true>(
// CPP: topk_rebuild<false, DST_ACCUM_MODE, false, true>(
// CPP: topk_merge<false, false, DST_ACCUM_MODE, false, true, TopkTieOrder::Unset, 8>(
func.func @topk_rank_stamped() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 16 : i32
  %logk = arith.constant 4 : i32
  %skip = arith.constant 1 : i32
  ttl.tile_regs_acquire
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start
      {rank_stamped = true, tag_bits = 8 : i32}
      : (index, i32, i32, i32) -> ()
  ttl.tile_topk_rebuild dst[%dst] direction = %dir
      iteration = %iter k = %k logk = %logk skip_second = %skip
      {rank_stamped = true, tag_bits = 8 : i32}
      : (index, i32, i32, i32, i32, i32) -> ()
  ttl.tile_topk_merge dst[%dst] iteration = %iter k = %k
      {rank_stamped = true, tag_bits = 8 : i32}
      : (index, i32, i32) -> ()
  ttl.tile_regs_release
  return
}

// A fused local sort and an unconfigured merge need different inits. The mode
// selects the slab helpers, so the two configurations occupy separate sync
// regions.
// TTKERNEL-LABEL: func.func @topk_reinit
// TTKERNEL: ttkernel.topk_tile_init() {fused = true}
// TTKERNEL-NEXT: ttkernel.topk_fuse_tile
// TTKERNEL-NEXT: ttkernel.topk_local_sort
// TTKERNEL: ttkernel.topk_defuse_tile
// TTKERNEL: ttkernel.topk_tile_init()
// TTKERNEL-NEXT: ttkernel.topk_merge
// CPP-LABEL: void kernel_main()
// CPP: topk_tile_init<true>();
// CPP-NEXT: topk_fuse_tile<true>(
// CPP-NEXT: topk_local_sort<false, DST_ACCUM_MODE, true>(
// CPP: topk_defuse_tile<true>(
// CPP: topk_tile_init();
// CPP-NEXT: topk_merge(
func.func @topk_reinit() attributes {ttkernel.thread = #ttkernel.thread<compute>} {
  %dst = arith.constant 0 : index
  %dir = arith.constant 0 : i32
  %end = arith.constant 4 : i32
  %start = arith.constant 0 : i32
  %iter = arith.constant 0 : i32
  %k = arith.constant 32 : i32
  ttl.tile_regs_acquire
  ttl.tile_topk_local_sort dst[%dst] direction = %dir
      end_phase = %end start_phase = %start {fused = true}
      : (index, i32, i32, i32) -> ()
  ttl.tile_regs_release
  ttl.tile_regs_acquire
  ttl.tile_topk_merge dst[%dst] iteration = %iter k = %k
      : (index, i32, i32) -> ()
  ttl.tile_regs_release
  return
}
