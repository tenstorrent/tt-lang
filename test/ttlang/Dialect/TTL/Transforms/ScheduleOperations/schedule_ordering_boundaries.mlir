// Summary: The tile scheduler preserves operations with unmodeled ordering
// semantics while reordering classified tile operations.
// RUN: ttlang-opt %s \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-schedule-operations))' \
// RUN:   | FileCheck %s

// A broadcast normally sorts before an independent exponential. The signpost
// is an ordering boundary, so neither tile operation may move across it.

// CHECK-LABEL: func.func @preserve_unmodeled_operation_order
// CHECK:       ttl.dst_section {
// CHECK-NEXT:    %[[DST:.*]] = ttl.dst_index
// CHECK-NEXT:    %[[EXP:.*]] = ttl.tile_exp %[[DST]]
// CHECK-NEXT:    ttl.signpost "ttl_scope"
// CHECK-NEXT:    %[[BCAST:.*]] = ttl.tile_bcast
// CHECK-NEXT:    ttl.signpost "ttl_scope" {is_end}
// CHECK-NEXT:  }
func.func @preserve_unmodeled_operation_order(
    %input: !ttcore.tile<32x32, bf16>,
    %output: !ttcore.tile<32x32, bf16>) {
  %dst0 = arith.constant 0 : index
  %dst1 = arith.constant 1 : index
  ttl.dst_section {
    %dst = ttl.dst_index %input[%dst0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    %exp = ttl.tile_exp %dst into dst[%dst0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    ttl.signpost "ttl_scope"
    %bcast = ttl.tile_bcast %input, %output 1 : i32 into dst[%dst1] : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
    ttl.signpost "ttl_scope" {is_end}
    ttl.yield
  }
  return
}

// Pure bookkeeping does not observe execution and remains transparent to
// scheduling. The broadcast therefore sorts before the exponential.

// CHECK-LABEL: func.func @schedule_across_pure_bookkeeping
// CHECK:       ttl.dst_section {
// CHECK-NEXT:    %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[BCAST:.*]] = ttl.tile_bcast
// CHECK-NEXT:    %[[DST:.*]] = ttl.dst_index
// CHECK-NEXT:    %[[EXP:.*]] = ttl.tile_exp %[[DST]]
// CHECK-NEXT:  }
func.func @schedule_across_pure_bookkeeping(
    %input: !ttcore.tile<32x32, bf16>,
    %output: !ttcore.tile<32x32, bf16>) {
  %dst0 = arith.constant 0 : index
  %dst1 = arith.constant 1 : index
  ttl.dst_section {
    %dst = ttl.dst_index %input[%dst0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    %exp = ttl.tile_exp %dst into dst[%dst0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    %zero = arith.constant 0 : i32
    %bcast = ttl.tile_bcast %input, %output 1 : i32 into dst[%dst1] : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
    ttl.yield
  }
  return
}

// A pure division with an unproven nonzero divisor is not speculatable. It is
// therefore an ordering boundary even though it has no memory effects.

// CHECK-LABEL: func.func @preserve_non_speculatable_operation_order
// CHECK:       ttl.dst_section {
// CHECK-NEXT:    %[[DST:.*]] = ttl.dst_index
// CHECK-NEXT:    %[[EXP:.*]] = ttl.tile_exp %[[DST]]
// CHECK-NEXT:    %[[QUOTIENT:.*]] = arith.divsi
// CHECK-NEXT:    %[[BCAST:.*]] = ttl.tile_bcast
// CHECK-NEXT:  }
func.func @preserve_non_speculatable_operation_order(
    %input: !ttcore.tile<32x32, bf16>,
    %output: !ttcore.tile<32x32, bf16>, %dividend: i32, %divisor: i32) {
  %dst0 = arith.constant 0 : index
  %dst1 = arith.constant 1 : index
  ttl.dst_section {
    %dst = ttl.dst_index %input[%dst0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    %exp = ttl.tile_exp %dst into dst[%dst0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    %quotient = arith.divsi %dividend, %divisor : i32
    %bcast = ttl.tile_bcast %input, %output 1 : i32 into dst[%dst1] : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
    ttl.yield
  }
  return
}

// A pure operation that consumes a tile result is not transparent because
// moving its producer after it would violate SSA dominance.

// CHECK-LABEL: func.func @preserve_pure_consumer_order
// CHECK:       ttl.dst_section {
// CHECK-NEXT:    %[[DST:.*]] = ttl.dst_index
// CHECK-NEXT:    %[[TENSOR:.*]] = tensor.from_elements %[[DST]]
// CHECK-NEXT:    %[[BCAST:.*]] = ttl.tile_bcast
// CHECK-NEXT:  }
func.func @preserve_pure_consumer_order(
    %input: !ttcore.tile<32x32, bf16>,
    %output: !ttcore.tile<32x32, bf16>) {
  %dst0 = arith.constant 0 : index
  %dst1 = arith.constant 1 : index
  ttl.dst_section {
    %dst = ttl.dst_index %input[%dst0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    %tensor = tensor.from_elements %dst : tensor<1x!ttcore.tile<32x32, bf16>>
    %bcast = ttl.tile_bcast %input, %output 1 : i32 into dst[%dst1] : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
    ttl.yield
  }
  return
}

// A recursively speculatable region may capture a tile result without listing
// it as an operation operand. The scheduler must preserve the producer before
// that nested use.

// CHECK-LABEL: func.func @preserve_nested_pure_consumer_order
// CHECK:       ttl.dst_section {
// CHECK-NEXT:    %[[DST:.*]] = ttl.dst_index
// CHECK-NEXT:    %[[EXP:.*]] = ttl.tile_exp %[[DST]]
// CHECK-NEXT:    scf.if
// CHECK:           tensor.from_elements %[[EXP]]
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[BCAST:.*]] = ttl.tile_bcast
// CHECK-NEXT:  }
func.func @preserve_nested_pure_consumer_order(
    %input: !ttcore.tile<32x32, bf16>,
    %output: !ttcore.tile<32x32, bf16>, %condition: i1) {
  %dst0 = arith.constant 0 : index
  %dst1 = arith.constant 1 : index
  ttl.dst_section {
    %dst = ttl.dst_index %input[%dst0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    %exp = ttl.tile_exp %dst into dst[%dst0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    scf.if %condition {
      %tensor = tensor.from_elements %exp : tensor<1x!ttcore.tile<32x32, bf16>>
    }
    %bcast = ttl.tile_bcast %input, %output 1 : i32 into dst[%dst1] : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
    ttl.yield
  }
  return
}
