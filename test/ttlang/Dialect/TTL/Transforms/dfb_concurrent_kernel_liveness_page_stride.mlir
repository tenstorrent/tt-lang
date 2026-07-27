// Tests transaction page-count compatibility for physical DFB reuse.
//
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s

// Equal transaction counts divide the four-page DFB capacity, so reuse is
// legal.

// CHECK-LABEL: func.func @uniform_page_count
// CHECK-SAME: ttl.base_cta_index = 1 : i32
// CHECK: ttl.bind_cb{cb_index = 0, {{.*}} {dfb_id = 0 : index}
// CHECK: ttl.bind_cb{cb_index = 0, {{.*}} {dfb_id = 1 : index}

func.func @uniform_page_count()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
  %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %first_reserved = ttl.cb_reserve %first : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %first : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %first_waited = ttl.cb_wait %first : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %first : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %second_reserved = ttl.cb_reserve %second : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %second : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %second_waited = ttl.cb_wait %second : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %second : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Different transaction page counts cannot share one physical DFB because
// their cumulative ring-pointer offsets do not preserve transaction alignment.

// CHECK-LABEL: func.func @mixed_page_count
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, {{.*}} {dfb_id = 0 : index}
// CHECK: ttl.bind_cb{cb_index = 1, {{.*}} {dfb_id = 1 : index}

func.func @mixed_page_count()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
  %partial = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %full = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %partial_reserved = ttl.cb_reserve %partial {num_tiles = 1 : i64} : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %partial {num_tiles = 1 : i64} : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %partial_waited = ttl.cb_wait %partial {num_tiles = 1 : i64} : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %partial {num_tiles = 1 : i64} : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %full_reserved = ttl.cb_reserve %full : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %full : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %full_waited = ttl.cb_wait %full : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %full : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// The two-page transactions may reuse one DFB, but the one-page transaction
// remains separate.

// CHECK-LABEL: func.func @compatible_subset
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, {{.*}} {dfb_id = 0 : index}
// CHECK: ttl.bind_cb{cb_index = 1, {{.*}} {dfb_id = 1 : index}
// CHECK: ttl.bind_cb{cb_index = 1, {{.*}} {dfb_id = 2 : index}

func.func @compatible_subset()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 3 : i32, ttl.crta_indices = []} {
  %one = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %two = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %three = ttl.bind_cb {cb_index = 2, block_count = 2} {dfb_id = 2 : index} : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %one_reserved = ttl.cb_reserve %one {num_tiles = 1 : i64} : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %one {num_tiles = 1 : i64} : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %one_waited = ttl.cb_wait %one {num_tiles = 1 : i64} : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %one {num_tiles = 1 : i64} : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %two_reserved = ttl.cb_reserve %two : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %two : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %two_waited = ttl.cb_wait %two : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %two : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %three_reserved = ttl.cb_reserve %three : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %three : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  %three_waited = ttl.cb_wait %three : <[1, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %three : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
  return
}

// -----

// Equal three-page transactions cannot share a four-page physical DFB because
// the transaction count does not divide the capacity.

// CHECK-LABEL: func.func @page_count_does_not_divide_capacity
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, {{.*}} {dfb_id = 0 : index}
// CHECK: ttl.bind_cb{cb_index = 1, {{.*}} {dfb_id = 1 : index}

func.func @page_count_does_not_divide_capacity()
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
  %first = ttl.bind_cb {cb_index = 0, block_count = 4} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %second = ttl.bind_cb {cb_index = 1, block_count = 4} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %first_reserved = ttl.cb_reserve %first {num_tiles = 3 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x3x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %first {num_tiles = 3 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %first_waited = ttl.cb_wait %first {num_tiles = 3 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x3x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %first {num_tiles = 3 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %second_reserved = ttl.cb_reserve %second {num_tiles = 3 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x3x!ttcore.tile<32x32, bf16>>
  ttl.cb_push %second {num_tiles = 3 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %second_waited = ttl.cb_wait %second {num_tiles = 3 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 4> -> tensor<1x3x!ttcore.tile<32x32, bf16>>
  ttl.cb_pop %second {num_tiles = 3 : i64} : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
  return
}
