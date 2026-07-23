// Safety coverage for ttl-finalize-dfb-indices user/compiler DFB reuse.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s

// Different consumer threads must keep distinct physical CB state even when
// the common producer handles the DFBs sequentially.
// CHECK-LABEL: module {
// CHECK-NOT: ttl.dfb_index_map
// CHECK-LABEL: func.func @different_consumer_producer
// CHECK: ttl.bind_cb{cb_index = 0,
// CHECK: ttl.bind_cb{cb_index = 1,
// CHECK-LABEL: func.func @consumer_zero
// CHECK: ttl.bind_cb{cb_index = 0,
// CHECK-LABEL: func.func @consumer_one
// CHECK: ttl.bind_cb{cb_index = 1,
module {
  func.func @different_consumer_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.base_cta_index = 2 : i32} {
    %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %r0 = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %r1 = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @consumer_zero()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32} {
    %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @consumer_one()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32} {
    %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %w1 = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Operations in a loop body project to the loop operation. DFBs that appear
// sequential in one static iteration therefore overlap across the backedge.
// CHECK-LABEL: func.func @loop_backedge_is_live
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0,
// CHECK: ttl.bind_cb{cb_index = 1,
func.func @loop_backedge_is_live()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %r0 = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %r1 = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %w1 = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  }
  return
}

// -----

// A pipe may retain or address CB pages asynchronously, so a pipe-attached
// DFB is never an arena candidate.
// CHECK-LABEL: func.func @pipe_attached_is_dedicated
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0,
// CHECK: ttl.bind_cb{cb_index = 1,
func.func @pipe_attached_is_dedicated()
    attributes {ttl.kernel_thread = #ttkernel.thread<noc>, ttl.base_cta_index = 2 : i32} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %pipe = ttl.create_pipe src(0, 0) dst(1, 0) to(1, 0) net 0 : !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>
  %r0 = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  %send = ttl.copy %cb0, %pipe : (!ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>, !ttl.pipe<src(0, 0) dst(1, 0) to(1, 0) net 0>) -> !ttl.transfer_handle<write>
  ttl.cb_push %cb0 : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  %r1 = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_push %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  return
}

// -----

// Different element types imply different page sizes and cannot share a CB.
// CHECK-LABEL: func.func @different_page_types
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0,
// CHECK: ttl.bind_cb{cb_index = 1,
func.func @different_page_types()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32} {
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>
  %r0 = ttl.cb_reserve %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %cb0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %r1 = ttl.cb_reserve %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_push %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  %w1 = ttl.cb_wait %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
  ttl.cb_pop %cb1 : <[1, 1], !ttcore.tile<32x32, f32>, 2>
  return
}

// -----

// A larger compiler DFB may reuse a drained user slot. Runtime metadata must
// resize the physical slot to the compiler member's full capacity.
// CHECK: module attributes {ttl.compiler_allocated_dfbs = [{block_count = 4 : i32, dfb_index = 0 : i32, element_type = !ttcore.tile<32x32, bf16>, num_tiles = 8 : i32}]}
// CHECK-LABEL: func.func @user_compiler_capacity_merge
// CHECK-SAME: ttl.base_cta_index = 1 : i32
// CHECK-COUNT-2: ttl.bind_cb{cb_index = 0,
// CHECK-NOT: cb_index = 1
func.func @user_compiler_capacity_merge()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, ttl.base_cta_index = 2 : i32} {
  %user = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %compiler = ttl.bind_cb {cb_index = 1, block_count = 4} {ttl.compiler_allocated} : !ttl.cb<[2, 4], !ttcore.tile<32x32, bf16>, 4>
  %r0 = ttl.cb_reserve %user : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %user : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %w0 = ttl.cb_wait %user : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %user : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %r1 = ttl.cb_reserve %compiler : <[2, 4], !ttcore.tile<32x32, bf16>, 4> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %compiler : <[2, 4], !ttcore.tile<32x32, bf16>, 4>
  %w1 = ttl.cb_wait %compiler : <[2, 4], !ttcore.tile<32x32, bf16>, 4> -> tensor<2x4x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %compiler : <[2, 4], !ttcore.tile<32x32, bf16>, 4>
  return
}
