// Tests each conservative condition that leaves a DFB lifetime unbounded.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s

// The valid baseline proves that two ordered lifetimes with one reserve, push,
// wait, and pop share one physical index. Each following section changes one
// condition and requires separate indices.

// CHECK-LABEL: func.func @bounded_baseline
// CHECK-SAME: ttl.base_cta_index = 1 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @bounded_baseline()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_reserved = ttl.cb_reserve %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_waited = ttl.cb_wait %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_reserved = ttl.cb_reserve %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_waited = ttl.cb_wait %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// More than one reserve prevents matching one reserve/push and wait/pop pair.

// CHECK-LABEL: func.func @multiple_reserves
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @multiple_reserves()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_reserved_0 = ttl.cb_reserve %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %first_reserved_1 = ttl.cb_reserve %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_waited = ttl.cb_wait %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_reserved = ttl.cb_reserve %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_waited = ttl.cb_wait %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// An acquisition nested in a separate region from its release is not a valid
// automatic synchronization interval.

// CHECK-LABEL: func.func @nested_reserve
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @nested_reserve()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_reserved = scf.execute_region -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
      %nested_reserved = ttl.cb_reserve %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      scf.yield %nested_reserved : tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    ttl.cb_push %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_waited = ttl.cb_wait %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_reserved = ttl.cb_reserve %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_waited = ttl.cb_wait %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// A multi-block function has no modeled linear program order.

// CHECK-LABEL: func.func @multiple_blocks
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @multiple_blocks()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    cf.br ^lifecycle
  ^lifecycle:
    %first_reserved = ttl.cb_reserve %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_waited = ttl.cb_wait %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_reserved = ttl.cb_reserve %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_waited = ttl.cb_wait %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// A reserve does not acquire a writable slot when its push executes first.

// CHECK-LABEL: func.func @push_before_reserve
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @push_before_reserve()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_push %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_reserved = ttl.cb_reserve %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %first_waited = ttl.cb_wait %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_reserved = ttl.cb_reserve %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_waited = ttl.cb_wait %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// A pop does not release a readable slot when its wait executes later.

// CHECK-LABEL: func.func @pop_before_wait
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @pop_before_wait()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_reserved = ttl.cb_reserve %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_waited = ttl.cb_wait %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %second_reserved = ttl.cb_reserve %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_waited = ttl.cb_wait %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// A push before the last producer-owned use does not terminate that producer
// interval.

// CHECK-LABEL: func.func @producer_use_after_push
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @producer_use_after_push()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_reserved = ttl.cb_reserve %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %late_producer_use = tensor.extract_slice %first_reserved[0, 0] [1, 1] [1, 1] : tensor<1x1x!ttcore.tile<32x32, bf16>> to tensor<1x1x!ttcore.tile<32x32, bf16>>
    %first_waited = ttl.cb_wait %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_reserved = ttl.cb_reserve %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_waited = ttl.cb_wait %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// A pop before the last consumer-owned use does not terminate that consumer
// interval.

// CHECK-LABEL: func.func @consumer_use_after_pop
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @consumer_use_after_pop()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_reserved = ttl.cb_reserve %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %first_waited = ttl.cb_wait %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %late_consumer_use = tensor.extract_slice %first_waited[0, 0] [1, 1] [1, 1] : tensor<1x1x!ttcore.tile<32x32, bf16>> to tensor<1x1x!ttcore.tile<32x32, bf16>>
    %second_reserved = ttl.cb_reserve %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %second_waited = ttl.cb_wait %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second_dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Unequal transfer counts do not prove zero occupancy after the pop.

// CHECK-LABEL: func.func @mismatched_tile_count
// CHECK-SAME: ttl.base_cta_index = 2 : i32
// CHECK: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// CHECK-NEXT: ttl.bind_cb{cb_index = 1, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @mismatched_tile_count()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first_dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 2], !ttcore.tile<16x32, bf16>, 2>
    %second_dfb = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 2], !ttcore.tile<16x32, bf16>, 2>
    %first_reserved = ttl.cb_reserve %first_dfb : <[1, 2], !ttcore.tile<16x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<16x32, bf16>>
    ttl.cb_push %first_dfb {num_tiles = 1 : i64} : <[1, 2], !ttcore.tile<16x32, bf16>, 2>
    %first_waited = ttl.cb_wait %first_dfb : <[1, 2], !ttcore.tile<16x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<16x32, bf16>>
    ttl.cb_pop %first_dfb : <[1, 2], !ttcore.tile<16x32, bf16>, 2>
    %second_reserved = ttl.cb_reserve %second_dfb : <[1, 2], !ttcore.tile<16x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<16x32, bf16>>
    ttl.cb_push %second_dfb : <[1, 2], !ttcore.tile<16x32, bf16>, 2>
    %second_waited = ttl.cb_wait %second_dfb : <[1, 2], !ttcore.tile<16x32, bf16>, 2> -> tensor<1x2x!ttcore.tile<16x32, bf16>>
    ttl.cb_pop %second_dfb : <[1, 2], !ttcore.tile<16x32, bf16>, 2>
    return
  }
}
