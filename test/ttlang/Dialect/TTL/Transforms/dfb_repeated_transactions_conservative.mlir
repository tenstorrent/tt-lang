// Tests conservative rejection of unproved repeated DFB transactions.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' | FileCheck %s --check-prefix=ALLOC
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices{reuse-user-dfbs=true})' -debug-only=ttl-finalize-dfb-indices -o /dev/null 2>&1 | FileCheck %s --check-prefix=REPORT

// ALLOC-COUNT-9: ttl.base_cta_index = 2 : i32
// REPORT: quiescence=mismatched-transaction {{.*}} kernel=@mismatched_count
// REPORT: quiescence=mismatched-transaction {{.*}} kernel=@overlapping_consumer_acquires
// REPORT: quiescence=mismatched-transaction {{.*}} kernel=@mismatched_tiles
// REPORT: quiescence=unsupported-control-flow {{.*}} kernel=@dynamic_trip_count
// REPORT: quiescence=incomplete-use-order {{.*}} kernel=@differing_iteration_domains
// REPORT: quiescence=unsupported-control-flow {{.*}} kernel=@conditional_iteration
// REPORT: quiescence=incomplete-use-order {{.*}} kernel=@access_outside_interval
// REPORT: DFB conflict lhs=0 rhs=1 reason=pointer-owner-mismatch
// REPORT: quiescence=incomplete-use-order {{.*}} kernel=@unrelated_opaque_access

// Producer and consumer transaction counts must match.

module {
  func.func @mismatched_count()
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
    ttl.opaque_call "consumer" dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>] () {header = "consumer.hpp"} : () -> ()
    %second_reserved = ttl.cb_reserve %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second_available = ttl.cb_wait %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// A second wait before the first pop leaves consumer acquisitions overlapping.

module {
  func.func @overlapping_consumer_acquires()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "producer" dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<reserve, 0, 4>, #ttl.dfb_protocol_effect<push, 0, 4>, #ttl.dfb_protocol_effect<reserve, 0, 4>, #ttl.dfb_protocol_effect<push, 0, 4>] () {header = "producer.hpp"} : () -> ()
    ttl.opaque_call "consumer" dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>] () {header = "consumer.hpp"} : () -> ()
    %second_reserved = ttl.cb_reserve %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second_available = ttl.cb_wait %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Producer and consumer tile counts must match at every transaction position.

module {
  func.func @mismatched_tiles()
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
    ttl.opaque_call "consumer" dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 2>, #ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 2>, #ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 2>, #ttl.dfb_protocol_effect<wait, 0, 2>, #ttl.dfb_protocol_effect<pop, 0, 2>] () {header = "consumer.hpp"} : () -> ()
    %second_reserved = ttl.cb_reserve %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second_available = ttl.cb_wait %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// A dynamic loop trip count cannot define an exact transaction run.

module {
  func.func @dynamic_trip_count(%upper: index)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %step = arith.constant 1 : index
    scf.for %transaction = %lower to %upper step %step {
      %reserved = ttl.cb_reserve %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
      ttl.cb_push %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    }
    ttl.opaque_call "consumer" dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>] () {header = "consumer.hpp"} : () -> ()
    %second_reserved = ttl.cb_reserve %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second_available = ttl.cb_wait %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Separate loops do not prove aligned reserve and push executions.

module {
  func.func @differing_iteration_domains()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %transaction = %lower to %upper step %step {
      %reserved = ttl.cb_reserve %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    }
    scf.for %transaction = %lower to %upper step %step {
      ttl.cb_push %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    }
    ttl.opaque_call "consumer" dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>] () {header = "consumer.hpp"} : () -> ()
    %second_reserved = ttl.cb_reserve %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second_available = ttl.cb_wait %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// A runtime condition inside the loop prevents uniform per-iteration proof.

module {
  func.func @conditional_iteration(%condition: i1)
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    scf.for %transaction = %lower to %upper step %step {
      scf.if %condition {
        %reserved = ttl.cb_reserve %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
        ttl.cb_push %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
      }
    }
    ttl.opaque_call "consumer" dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>] () {header = "consumer.hpp"} : () -> ()
    %second_reserved = ttl.cb_reserve %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second_available = ttl.cb_wait %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// A payload access after the consumer release is outside the owned interval.

module {
  func.func @access_outside_interval()
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
    ttl.opaque_call "consumer" dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) dfb_effects [#ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>, #ttl.dfb_protocol_effect<wait, 0, 4>, #ttl.dfb_protocol_effect<pop, 0, 4>] () {header = "consumer.hpp"} : () -> ()
    scf.for %transaction = %lower to %upper step %step {
      ttl.opaque_call "late_use" dfb_dependencies(%first : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>) () {header = "late_use.hpp"} : () -> ()
    }
    %second_reserved = ttl.cb_reserve %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second_available = ttl.cb_wait %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}

// -----

// Lifecycles controlled by different NoC processors cannot share one index.

module {
  func.func @pointer_owner_noc0()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 0 : i32,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
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
    return
  }
  func.func @pointer_owner_noc1()
      attributes {ttl.kernel_thread = #ttkernel.thread<noc>,
                  ttl.noc_index = 1 : i32,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
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

// An unknown external DFB access prevents a complete lifetime proof.

module {
  func.func @unrelated_opaque_access()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>,
                  ttl.base_cta_index = 2 : i32, ttl.crta_indices = []} {
    %first = ttl.bind_cb {cb_index = 0, block_count = 2} {dfb_id = 0 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second = ttl.bind_cb {cb_index = 1, block_count = 2} {dfb_id = 1 : index} : !ttl.cb<[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %first_reserved = ttl.cb_reserve %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %first_available = ttl.cb_wait %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %first : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    ttl.opaque_call "unrelated" () {header = "unrelated.hpp", unknown_dfb_access} : () -> ()
    %second_reserved = ttl.cb_reserve %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_push %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    %second_available = ttl.cb_wait %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x4x!ttcore.tile<32x32, bf16>>
    ttl.cb_pop %second : <[1, 4], !ttcore.tile<32x32, bf16>, 2>
    return
  }
}
