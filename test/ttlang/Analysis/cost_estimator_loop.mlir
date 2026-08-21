// RUN: ttlang-opt --split-input-file --ttkernel-cost-estimate=detail=1 %s -o /dev/null 2>&1 | FileCheck %s

// Loops, which the estimator unrolls in place rather than costing once.
//
// The compute function here is real lowered IR, not hand-written: it is
// test/ttlang/Conversion/TTLToTTKernel/fpu_binary_lowering.mlir's f32 add+tanh
// kernel run through ttl-subblock-compute-for-dst and convert-ttl-to-ttkernel,
// so the loop is a genuine subblock loop over a 2x3 block with the
// affine.linearize_index tile addressing that pass emits. Reader and writer
// threads are added to supply the credits, since a compute kernel waiting on
// buffers nothing fills would deadlock rather than be estimated.

module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @read() attributes {ttkernel.thread = #ttkernel.thread<noc>, ttl.noc_index = 0 : i32} {
    %c6_i32 = arith.constant 6 : i32
    %0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<12, !ttcore.tile<32x32, f32>>
    %2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<12, !ttcore.tile<32x32, f32>>
    ttkernel.cb_reserve_back(%0, %c6_i32) : (!ttkernel.cb<12, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.cb_reserve_back(%2, %c6_i32) : (!ttkernel.cb<12, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.noc_async_read_barrier() : () -> ()
    ttkernel.cb_push_back(%0, %c6_i32) : (!ttkernel.cb<12, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.cb_push_back(%2, %c6_i32) : (!ttkernel.cb<12, !ttcore.tile<32x32, f32>>, i32) -> ()
    return
  }

  func.func @compute() attributes {dst_full_sync_en = false, fp32_dest_acc_en = true, ttkernel.thread = #ttkernel.thread<compute>, ttl.unpack_to_dest_fp32 = array<i32>} {
    %c3_i32 = arith.constant 3 : i32
    %c6_i32 = arith.constant 6 : i32
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<12, !ttcore.tile<32x32, f32>>
    %1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<12, !ttcore.tile<32x32, f32>>
    %2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.cb<12, !ttcore.tile<32x32, f32>>
    ttkernel.cb_wait_front(%0, %c6_i32) : (!ttkernel.cb<12, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.cb_wait_front(%2, %c6_i32) : (!ttkernel.cb<12, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.binary_op_init_common(%0, %2, %1) : (!ttkernel.cb<12, !ttcore.tile<32x32, f32>>, !ttkernel.cb<12, !ttcore.tile<32x32, f32>>, !ttkernel.cb<12, !ttcore.tile<32x32, f32>>) -> ()
    scf.for %arg0 = %c0 to %c2 step %c1 {
      ttkernel.cb_reserve_back(%1, %c3_i32) : (!ttkernel.cb<12, !ttcore.tile<32x32, f32>>, i32) -> ()
      ttkernel.tile_regs_acquire() : () -> ()
      %3 = affine.linearize_index [%arg0, %c0] by (2, 3) : index
      ttkernel.add_tiles_init(%0, %2) : (!ttkernel.cb<12, !ttcore.tile<32x32, f32>>, !ttkernel.cb<12, !ttcore.tile<32x32, f32>>) -> ()
      ttkernel.add_tiles(%0, %2, %3, %3, %c0) : (!ttkernel.cb<12, !ttcore.tile<32x32, f32>>, !ttkernel.cb<12, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
      %4 = affine.linearize_index [%arg0, %c1] by (2, 3) : index
      ttkernel.add_tiles(%0, %2, %4, %4, %c1) : (!ttkernel.cb<12, !ttcore.tile<32x32, f32>>, !ttkernel.cb<12, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
      %5 = affine.linearize_index [%arg0, %c2] by (2, 3) : index
      ttkernel.add_tiles(%0, %2, %5, %5, %c2) : (!ttkernel.cb<12, !ttcore.tile<32x32, f32>>, !ttkernel.cb<12, !ttcore.tile<32x32, f32>>, index, index, index) -> ()
      ttkernel.tanh_tile_init() : () -> ()
      ttkernel.tanh_tile(%c0) : (index) -> ()
      ttkernel.tanh_tile(%c1) : (index) -> ()
      ttkernel.tanh_tile(%c2) : (index) -> ()
      ttkernel.tile_regs_commit() : () -> ()
      ttkernel.tile_regs_wait() : () -> ()
      ttkernel.pack_tile(%c0, %1, %c0, true) : (index, !ttkernel.cb<12, !ttcore.tile<32x32, f32>>, index) -> ()
      ttkernel.pack_tile(%c1, %1, %c1, true) : (index, !ttkernel.cb<12, !ttcore.tile<32x32, f32>>, index) -> ()
      ttkernel.pack_tile(%c2, %1, %c2, true) : (index, !ttkernel.cb<12, !ttcore.tile<32x32, f32>>, index) -> ()
      ttkernel.tile_regs_release() : () -> ()
      ttkernel.cb_push_back(%1, %c3_i32) : (!ttkernel.cb<12, !ttcore.tile<32x32, f32>>, i32) -> ()
    } {ttl.subblock_dim = 0 : index, ttl.subblock_loop_stride = 3 : index}
    ttkernel.cb_pop_front(%0, %c6_i32) : (!ttkernel.cb<12, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.cb_pop_front(%2, %c6_i32) : (!ttkernel.cb<12, !ttcore.tile<32x32, f32>>, i32) -> ()
    return
  }

  func.func @write() attributes {ttkernel.thread = #ttkernel.thread<noc>, ttl.noc_index = 1 : i32} {
    %c6_i32 = arith.constant 6 : i32
    %1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<12, !ttcore.tile<32x32, f32>>
    ttkernel.cb_wait_front(%1, %c6_i32) : (!ttkernel.cb<12, !ttcore.tile<32x32, f32>>, i32) -> ()
    ttkernel.noc_async_write_barrier() : () -> ()
    ttkernel.cb_pop_front(%1, %c6_i32) : (!ttkernel.cb<12, !ttcore.tile<32x32, f32>>, i32) -> ()
    return
  }
}

// Two iterations, unrolled in place: every body operation is placed once per
// iteration, in program order, so a lane holds A,B,C,A,B,C rather than A,A,B,B.
// The op counts are trip x body -- 21 on math for a body of nine, ten with the
// pre-loop init -- and the costs are the table's own numbers, which a
// regeneration can move.
// CHECK: cost estimate: 33 of 57 placements measured
// CHECK-NEXT: 0 unmatched {{.*}} untimed
// CHECK: kernels: read compute write
// CHECK: TRISC0 unpack: 13 ops
// CHECK: TRISC1 math: 21 ops
// CHECK: TRISC2 pack: 15 ops

// First iteration on math, then the second with no operation between the
// commit and the next acquire: the loop body repeated, not a body costed once
// and multiplied.
// CHECK: TRISC1 math{{$}}
// CHECK: ttkernel.binary_op_init_common {{.*}} meas
// CHECK-NEXT: ttkernel.tile_regs_acquire {{.*}} untimed
// CHECK-NEXT: ttkernel.add_tiles_init {{.*}} 90 {{.*}} meas
// CHECK-NEXT: ttkernel.add_tiles {{.*}} 18 {{.*}} meas
// CHECK-NEXT: ttkernel.add_tiles {{.*}} 18 {{.*}} meas
// CHECK-NEXT: ttkernel.add_tiles {{.*}} 18 {{.*}} meas
// CHECK-NEXT: ttkernel.tanh_tile_init {{.*}} meas
// CHECK-NEXT: ttkernel.tanh_tile {{.*}} 362 {{.*}} meas
// CHECK-NEXT: ttkernel.tanh_tile {{.*}} 362 {{.*}} meas
// CHECK-NEXT: ttkernel.tanh_tile {{.*}} 362 {{.*}} meas
// CHECK-NEXT: ttkernel.tile_regs_commit {{.*}} untimed
// CHECK-NEXT: ttkernel.tile_regs_acquire {{.*}} untimed
// CHECK-NEXT: ttkernel.add_tiles_init {{.*}} 90 {{.*}} meas
// CHECK-NEXT: ttkernel.add_tiles {{.*}} 18 {{.*}} meas

// -----

// The trip count has to be static. A loop bounded by a value only known at
// runtime cannot be unrolled, and steady-state extrapolation is not implemented,
// so the estimate is refused at the loop rather than produced from a body placed
// some arbitrary number of times. The trip count arrives as a function argument
// here to keep the refusal to one cause; a compute kernel reading it through
// `ttkernel.get_common_arg_val` would fail for a second, unrelated reason.
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @compute(%trip: index) attributes {dst_full_sync_en = false, fp32_dest_acc_en = false, ttkernel.thread = #ttkernel.thread<compute>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cb0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
    %cb1 = ttkernel.get_compile_time_arg_val(1) : () -> !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>
    ttkernel.binary_op_init_common(%cb0, %cb0, %cb1) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>) -> ()
    scf.for %i = %c0 to %trip step %c1 {
      ttkernel.tile_regs_acquire() : () -> ()
      ttkernel.add_tiles_init(%cb0, %cb0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>) -> ()
      ttkernel.add_tiles(%cb0, %cb0, %c0, %c0, %c0) : (!ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index, index, index) -> ()
      ttkernel.tile_regs_commit() : () -> ()
      ttkernel.tile_regs_wait() : () -> ()
      ttkernel.pack_tile(%c0, %cb1, %c0, true) : (index, !ttkernel.cb<4, !ttcore.tile<32x32, bf16>>, index) -> ()
      ttkernel.tile_regs_release() : () -> ()
    }
    return
  }
}

// The loop is named as the reason, at the loop, and no estimate is offered --
// rather than a latency computed from a program the hardware will not run.
// CHECK: warning: cost estimator cannot determine this loop's trip count
// CHECK-SAME: statically
// CHECK: cost estimate: unavailable, see the warnings above
