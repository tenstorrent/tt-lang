// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline | FileCheck %s --check-prefix=TTKERNEL
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline --convert-ttkernel-to-emitc | FileCheck %s --check-prefix=EMITC

// TTKERNEL-LABEL: func.func @mutate
// TTKERNEL-DAG: %[[ONE:.*]] = arith.constant 1 : index
// TTKERNEL-DAG: %[[ZERO:.*]] = arith.constant 0 : index
// TTKERNEL: %[[OUTPUT:.*]] = ttkernel.get_compile_time_arg_val({{[0-9]+}})
// TTKERNEL: %[[STATE:.*]] = ttkernel.get_compile_time_arg_val({{[0-9]+}})
// TTKERNEL: ttkernel.cb_wait_front(%[[STATE]],
// TTKERNEL-NOT: ttkernel.cb_reserve_back(%[[STATE]],
// TTKERNEL: ttkernel.pack_waited_tile({{.*}}, %[[STATE]], %[[ZERO]], true) {acquired_tiles = 2 : i64}
// TTKERNEL-NEXT: ttkernel.pack_waited_tile({{.*}}, %[[STATE]], %[[ONE]], true) {acquired_tiles = 2 : i64}
// TTKERNEL-NOT: ttkernel.pack_waited_tile
// TTKERNEL-NOT: ttkernel.cb_push_back(%[[STATE]],
// TTKERNEL: ttkernel.cb_reserve_back(%[[OUTPUT]],
// TTKERNEL: ttkernel.pack_tile_block({{.*}}, %[[OUTPUT]],
// TTKERNEL: ttkernel.cb_push_back(%[[OUTPUT]],
// TTKERNEL: ttkernel.cb_pop_front(%[[STATE]],

// The proof attribute is compiler-only. Runtime lowering uses the existing
// absolute-index pack API and emits no reserve or push for the mutated DFB.
// EMITC-LABEL: func.func @mutate
// EMITC-COUNT-2: emitc.call_opaque "pack_tile"
// EMITC-NOT: emitc.call_opaque "pack_tile"
// EMITC-NOT: pack_waited_tile

module attributes {
  ttl.launch_grid = [1, 1],
  ttl.target_arch = #ttcore.arch<blackhole>
} {
  // The real pipeline publishes input state from a data-movement thread.
  func.func @produce_state() attributes {
    ttl.base_cta_index = 2 : i32,
    ttl.crta_indices = [],
    ttl.kernel_thread = #ttkernel.thread<noc>,
    ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement>,
    ttl.noc_index = 0 : i32
  } {
    %state_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
    %block = ttl.cb_reserve %state_dfb
        : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %state_dfb
        : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
    return
  }

  func.func @mutate() attributes {
    ttl.base_cta_index = 2 : i32,
    ttl.crta_indices = [],
    ttl.kernel_thread = #ttkernel.thread<compute>,
    ttl.logical_kernel = #ttl.logical_kernel<kind = compute>
  } {
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        {dfb_id = 1 : index}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 2>
    %state_dfb = ttl.bind_cb {cb_index = 0, block_count = 1}
        {dfb_id = 0 : index}
        : !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>
    %wait = ttl.cb_wait %state_dfb
        : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    %state = ttl.attach_cb %wait, %state_dfb
        : (tensor<1x2x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 2], !ttcore.tile<32x32, bf16>, 1>)
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    %one = ttl.fill 1.000000e+00
        : tensor<1x2x!ttcore.tile<32x32, bf16>>
    %updated = ttl.add %state, %one
        : tensor<1x2x!ttcore.tile<32x32, bf16>>,
          tensor<1x2x!ttcore.tile<32x32, bf16>>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.store %updated, %wait
        : tensor<1x2x!ttcore.tile<32x32, bf16>>,
          tensor<1x2x!ttcore.tile<32x32, bf16>>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.store %state, %output
        : tensor<1x2x!ttcore.tile<32x32, bf16>>,
          tensor<1x2x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %output_dfb
        : <[1, 2], !ttcore.tile<32x32, bf16>, 2>
    ttl.cb_pop %state_dfb
        : <[1, 2], !ttcore.tile<32x32, bf16>, 1>
    return
  }
}
