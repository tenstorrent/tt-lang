// Tests normalization of statically repeated DFB protocol transactions.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s --check-prefix=REUSE
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPORT

// A loop producer and an explicit external consumer describe the same four
// transactions. The non-protocol producer access executes in the same loop.

// REUSE-LABEL: func.func @loop_to_explicit
// REUSE-SAME: ttl.base_cta_index = 1 : i32
// REUSE: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @loop_to_explicit()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %transaction = %lower to %upper step %step {
      %reserved = ttl.cb_reserve %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
      ttl.opaque_call "producer_use" dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) () {header = "producer_use.hpp"} : () -> ()
      ttl.cb_push %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    }
    ttl.opaque_call "explicit_consumer" dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>] () {header = "consumer.hpp"} : () -> ()
    scf.for %transaction = %lower to %upper step %step {
      %reserved = ttl.cb_reserve %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
      ttl.opaque_call "producer_use" dfb_dependencies(%second : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) () {header = "producer_use.hpp"} : () -> ()
      ttl.cb_push %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    }
    ttl.opaque_call "explicit_consumer" dfb_dependencies(%second : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>] () {header = "consumer.hpp"} : () -> ()
    return
  }
}

// -----

// Large exact runs remain compact while retaining their exact count and tile
// size in the allocation report.

// REPORT: operation=ttl.cb_reserve kernel=@large_static_run
// REPORT: node (0,0) quiescence=none
// REPORT-SAME: transactions=[run(count=1000,tiles=1)]

module {
  func.func @large_static_run()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 1000 : index
    %step = arith.constant 1 : index
    scf.for %transaction = %lower to %upper step %step {
      %reserved = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      %available = ttl.cb_wait %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    }
    return
  }
}

// -----

// A launch-node condition may enclose the producer loop and the explicit
// consumer independently. Effect sequence indices still order the four
// wait/pop pairs within the consumer invocation.

// REUSE-LABEL: func.func @nested_explicit_consumer
// REUSE-SAME: ttl.base_cta_index = 1 : i32
// REUSE: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 1 : index}
// REPORT: operation=ttl.cb_reserve kernel=@nested_explicit_consumer
// REPORT: node (0,0) quiescence=none
// REPORT-SAME: occurrences=[0:4, 1:4, 2:1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1]
// REPORT-SAME: transactions=[4, 4, 4, 4]

module attributes {ttl.launch_grid = array<i64: 2, 1>} {
  func.func @nested_explicit_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %core_x = ttl.core_x : index
    %zero = arith.constant 0 : index
    %two = arith.constant 2 : index
    %four = arith.constant 4 : index
    %one = arith.constant 1 : index
    %inside_grid = arith.cmpi ne, %core_x, %two : index
    scf.if %inside_grid {
      scf.for %transaction = %zero to %four step %one {
        %reserved = ttl.cb_reserve %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
        ttl.cb_push %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
      }
    }
    scf.if %inside_grid {
      ttl.opaque_call "consumer" dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>] () {header = "consumer.hpp"} : () -> ()
    }
    scf.for %transaction = %zero to %four step %one {
      %reserved = ttl.cb_reserve %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    }
    ttl.opaque_call "consumer" dfb_dependencies(%second : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>] () {header = "consumer.hpp"} : () -> ()
    return
  }
}

// -----

// An explicit producer summary and a loop consumer normalize identically.

// REUSE-LABEL: func.func @explicit_to_loop
// REUSE-SAME: ttl.base_cta_index = 1 : i32
// REUSE: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 1 : index}

module {
  func.func @explicit_to_loop()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    ttl.opaque_call "explicit_producer" dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 4>, #ttl.dfb_protocol_effect<push, 0, 4>, #ttl.dfb_protocol_effect<reserve, 0, 4>, #ttl.dfb_protocol_effect<push, 0, 4>, #ttl.dfb_protocol_effect<reserve, 0, 4>, #ttl.dfb_protocol_effect<push, 0, 4>, #ttl.dfb_protocol_effect<reserve, 0, 4>, #ttl.dfb_protocol_effect<push, 0, 4>] () {header = "producer.hpp"} : () -> ()
    scf.for %transaction = %lower to %upper step %step {
      %available = ttl.cb_wait %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
      ttl.opaque_call "consumer_use" dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) () {header = "consumer_use.hpp"} : () -> ()
      ttl.cb_pop %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    }
    ttl.opaque_call "explicit_producer" dfb_dependencies(%second : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 4>, #ttl.dfb_protocol_effect<push, 0, 4>, #ttl.dfb_protocol_effect<reserve, 0, 4>, #ttl.dfb_protocol_effect<push, 0, 4>, #ttl.dfb_protocol_effect<reserve, 0, 4>, #ttl.dfb_protocol_effect<push, 0, 4>, #ttl.dfb_protocol_effect<reserve, 0, 4>, #ttl.dfb_protocol_effect<push, 0, 4>] () {header = "producer.hpp"} : () -> ()
    scf.for %transaction = %lower to %upper step %step {
      %available = ttl.cb_wait %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
      ttl.opaque_call "consumer_use" dfb_dependencies(%second : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) () {header = "consumer_use.hpp"} : () -> ()
      ttl.cb_pop %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    }
    return
  }
}

// -----

// Independent producer and consumer loops retain their distinct iteration
// domains while matching by normalized transaction position.

// REUSE-LABEL: func.func @loop_to_loop
// REUSE-SAME: ttl.base_cta_index = 1 : i32
// REUSE: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// REUSE-NEXT: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 1 : index}
// REPORT-COUNT-6: transactions=[4, 4, 4, 4]

module {
  func.func @loop_to_loop()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %transaction = %lower to %upper step %step {
      %reserved = ttl.cb_reserve %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    }
    scf.for %transaction = %lower to %upper step %step {
      %available = ttl.cb_wait %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    }
    scf.for %transaction = %lower to %upper step %step {
      %reserved = ttl.cb_reserve %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    }
    scf.for %transaction = %lower to %upper step %step {
      %available = ttl.cb_wait %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    }
    return
  }
}

// -----

// Each straight-line acquisition owns only the direct DFB accesses before the
// next same-kind acquisition. Both producer and consumer runs remain bounded.

// REUSE-LABEL: func.func @straight_line_producer
// REUSE-SAME: ttl.base_cta_index = 1 : i32
// REUSE: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// REUSE-LABEL: func.func @straight_line_consumer
// REUSE-SAME: ttl.base_cta_index = 1 : i32
// REUSE: ttl.bind_cb{cb_index = 0, block_count = 2} {dfb_id = 0 : index}
// REPORT: DFB logical_id=0 bounded=1
// REPORT: node (0,0) quiescence=none
// REPORT-SAME: transactions=[1, 1]

module {
  func.func @straight_line_producer()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserved_0 = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.opaque_call "producer_use" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) () {header = "producer_use.hpp"} : () -> ()
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %reserved_1 = ttl.cb_reserve %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.opaque_call "producer_use" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) () {header = "producer_use.hpp"} : () -> ()
    ttl.cb_push %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }

  func.func @straight_line_consumer()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 1 : i32, ttl.crta_indices = []} {
    %dfb = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %waited_0 = ttl.cb_wait %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.opaque_call "consumer_use" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) () {header = "consumer_use.hpp"} : () -> ()
    ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %waited_1 = ttl.cb_wait %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.opaque_call "consumer_use" dfb_dependencies(%dfb : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) () {header = "consumer_use.hpp"} : () -> ()
    ttl.cb_pop %dfb : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
