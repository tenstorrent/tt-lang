// Summary: A classified tile_fill participates in scheduling instead of
// bounding it. Companion to schedule_ordering_boundaries.mlir, which covers the
// operations that do bound a schedule. Both cases below place an independent
// broadcast after the fill: the broadcast may only reach the front of the
// section if the fill is scheduled with everything else, because an
// unclassified fill is an ordering boundary that splits the section into two
// independently sorted groups.
// RUN: ttlang-opt %s --split-input-file \
// RUN:   -pass-pipeline='builtin.module(func.func(ttl-schedule-operations))' \
// RUN:   | FileCheck %s

// The fill writes a register no other operation touches, so it carries no
// dependency and sorts by category alone. It lands after the operations that
// configure MATH from a dataflow buffer and before the exponential that depends
// on the named register.

// CHECK-LABEL: func.func @schedule_across_fill
// CHECK:       ttl.dst_section {
// CHECK-NEXT:    %[[BCAST:.*]] = ttl.tile_bcast
// CHECK-NEXT:    %[[DST:.*]] = ttl.dst_index
// CHECK-NEXT:    %[[FILL:.*]] = ttl.tile_fill
// CHECK-NEXT:    %[[EXP:.*]] = ttl.tile_exp %[[DST]]
// CHECK-NEXT:  }
func.func @schedule_across_fill(
    %input: !ttcore.tile<32x32, bf16>,
    %output: !ttcore.tile<32x32, bf16>) {
  %dst0 = arith.constant 0 : index
  %dst1 = arith.constant 1 : index
  %dst2 = arith.constant 2 : index
  ttl.dst_section {
    %dst = ttl.dst_index %input[%dst0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    %exp = ttl.tile_exp %dst into dst[%dst0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    %fill = ttl.tile_fill 3.000000e+00 into dst[%dst1] : !ttcore.tile<32x32, bf16>
    %bcast = ttl.tile_bcast %input, %output 1 : i32 into dst[%dst2] : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
    ttl.yield
  }
  return
}

// -----

// Participation does not weaken the register hazards. Here the fill overwrites
// the register the exponential reads, so the write-after-read dependency gives
// it a greater depth and pins it behind that reader. The broadcast still
// reaches the front, which distinguishes a scheduled fill from a boundary: a
// section that merely failed to sort would leave every operation in place.

// CHECK-LABEL: func.func @fill_stays_after_last_reader
// CHECK:       ttl.dst_section {
// CHECK-NEXT:    %[[BCAST:.*]] = ttl.tile_bcast
// CHECK-NEXT:    %[[DST:.*]] = ttl.dst_index
// CHECK-NEXT:    %[[EXP:.*]] = ttl.tile_exp %[[DST]]
// CHECK-NEXT:    %[[FILL:.*]] = ttl.tile_fill
// CHECK-NEXT:  }
func.func @fill_stays_after_last_reader(
    %input: !ttcore.tile<32x32, bf16>,
    %output: !ttcore.tile<32x32, bf16>) {
  %dst0 = arith.constant 0 : index
  %dst1 = arith.constant 1 : index
  ttl.dst_section {
    %dst = ttl.dst_index %input[%dst0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    %exp = ttl.tile_exp %dst into dst[%dst0] : !ttcore.tile<32x32, bf16> -> !ttcore.tile<32x32, bf16>
    %fill = ttl.tile_fill 3.000000e+00 into dst[%dst0] : !ttcore.tile<32x32, bf16>
    %bcast = ttl.tile_bcast %input, %output 1 : i32 into dst[%dst1] : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
    ttl.yield
  }
  return
}
