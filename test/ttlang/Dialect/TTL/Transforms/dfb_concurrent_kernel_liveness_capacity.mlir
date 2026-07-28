// Tests physical DFB allocation at the 32-index hardware boundary.
// RUN: ttlang-opt %s -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s

// Thirty-one unbounded DFBs interfere with every other DFB. The last three
// each have one reserve, push, wait, and pop. Their ordered lifetimes share
// physical index 31, reducing 34 logical DFBs to exactly 32 physical indices.

// CHECK: module attributes {ttl.dfb_allocations = [
// CHECK-COUNT-32: dfb_index = {{[0-9]+}} : i32
// CHECK-NOT: dfb_index = 32
// CHECK-LABEL: func.func @capacity_boundary
// CHECK-SAME: ttl.base_cta_index = 32 : i32
// CHECK: %[[REUSABLE0:.*]] = ttl.bind_cb{cb_index = 31, block_count = 2} {dfb_id = 31 : index}
// CHECK-NEXT: %[[REUSABLE1:.*]] = ttl.bind_cb{cb_index = 31, block_count = 2} {dfb_id = 32 : index}
// CHECK-NEXT: %[[REUSABLE2:.*]] = ttl.bind_cb{cb_index = 31, block_count = 2} {dfb_id = 33 : index}
// CHECK-NOT: ttl.bind_cb{cb_index = 32

module {
  func.func @capacity_boundary()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 34 : i32, ttl.crta_indices = []} {
    %persistent0 = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent1 = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent2 = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent3 = ttl.bind_cb {cb_index = 3, block_count = 2} {dfb_id = 3 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent4 = ttl.bind_cb {cb_index = 4, block_count = 2} {dfb_id = 4 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent5 = ttl.bind_cb {cb_index = 5, block_count = 2} {dfb_id = 5 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent6 = ttl.bind_cb {cb_index = 6, block_count = 2} {dfb_id = 6 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent7 = ttl.bind_cb {cb_index = 7, block_count = 2} {dfb_id = 7 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent8 = ttl.bind_cb {cb_index = 8, block_count = 2} {dfb_id = 8 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent9 = ttl.bind_cb {cb_index = 9, block_count = 2} {dfb_id = 9 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent10 = ttl.bind_cb {cb_index = 10, block_count = 2} {dfb_id = 10 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent11 = ttl.bind_cb {cb_index = 11, block_count = 2} {dfb_id = 11 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent12 = ttl.bind_cb {cb_index = 12, block_count = 2} {dfb_id = 12 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent13 = ttl.bind_cb {cb_index = 13, block_count = 2} {dfb_id = 13 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent14 = ttl.bind_cb {cb_index = 14, block_count = 2} {dfb_id = 14 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent15 = ttl.bind_cb {cb_index = 15, block_count = 2} {dfb_id = 15 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent16 = ttl.bind_cb {cb_index = 16, block_count = 2} {dfb_id = 16 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent17 = ttl.bind_cb {cb_index = 17, block_count = 2} {dfb_id = 17 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent18 = ttl.bind_cb {cb_index = 18, block_count = 2} {dfb_id = 18 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent19 = ttl.bind_cb {cb_index = 19, block_count = 2} {dfb_id = 19 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent20 = ttl.bind_cb {cb_index = 20, block_count = 2} {dfb_id = 20 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent21 = ttl.bind_cb {cb_index = 21, block_count = 2} {dfb_id = 21 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent22 = ttl.bind_cb {cb_index = 22, block_count = 2} {dfb_id = 22 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent23 = ttl.bind_cb {cb_index = 23, block_count = 2} {dfb_id = 23 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent24 = ttl.bind_cb {cb_index = 24, block_count = 2} {dfb_id = 24 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent25 = ttl.bind_cb {cb_index = 25, block_count = 2} {dfb_id = 25 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent26 = ttl.bind_cb {cb_index = 26, block_count = 2} {dfb_id = 26 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent27 = ttl.bind_cb {cb_index = 27, block_count = 2} {dfb_id = 27 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent28 = ttl.bind_cb {cb_index = 28, block_count = 2} {dfb_id = 28 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent29 = ttl.bind_cb {cb_index = 29, block_count = 2} {dfb_id = 29 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %persistent30 = ttl.bind_cb {cb_index = 30, block_count = 2} {dfb_id = 30 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reusable0 = ttl.bind_cb {cb_index = 31, block_count = 2} {dfb_id = 31 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reusable1 = ttl.bind_cb {cb_index = 32, block_count = 2} {dfb_id = 32 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reusable2 = ttl.bind_cb {cb_index = 33, block_count = 2} {dfb_id = 33 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %producer0 = ttl.cb_reserve %reusable0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %reusable0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %consumer0 = ttl.cb_wait %reusable0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %reusable0 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %producer1 = ttl.cb_reserve %reusable1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %reusable1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %consumer1 = ttl.cb_wait %reusable1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %reusable1 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %producer2 = ttl.cb_reserve %reusable2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %reusable2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %consumer2 = ttl.cb_wait %reusable2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %reusable2 : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
