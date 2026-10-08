// RUN: ttlang-opt %s --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --split-input-file --canonicalize | FileCheck %s
// RUN: ttlang-opt %s --split-input-file --symbol-dce | FileCheck %s
// RUN: ttlang-opt %s --split-input-file --mlir-print-op-generic | ttlang-opt --split-input-file | FileCheck %s

// Summary: Verifies that networks and their fork, split, and merge records
// round-trip through both the custom and the generic form, and that
// canonicalization and symbol DCE keep them (they are not Pure and are public
// by default).

// Verify an empty network round-trips.
// CHECK-LABEL: ttl.dfb.network @empty {
// CHECK-NEXT: }
ttl.dfb.network @empty {
}

// -----

// Verify that the fork op round-trips.
// CHECK-LABEL: ttl.dfb.network @fork_net {
// CHECK-NEXT: ttl.dfb.fork @f_auto 0 : index -> [1, 2] storage = <auto>
// CHECK-NEXT: ttl.dfb.fork @f_replicated 3 : index -> [4, 5] storage = <replicated>
// CHECK-NEXT: ttl.dfb.fork @f_shared 6 : index -> [7, 8] storage = <shared>
// CHECK-NEXT: }
ttl.dfb.network @fork_net {
  ttl.dfb.fork @f_auto 0 : index -> [1, 2] storage = <auto>
  ttl.dfb.fork @f_replicated 3 : index -> [4, 5] storage = <replicated>
  ttl.dfb.fork @f_shared 6 : index -> [7, 8] storage = <shared>
}

// -----

// Verify that the split op with DFB id source round-trips.
// CHECK-LABEL: ttl.dfb.network @split_net_dfb_id_source {
// CHECK-NEXT: ttl.dfb.split @split 0 : index -> [1, 2] policy = <contiguous>
// CHECK-NEXT: }
ttl.dfb.network @split_net_dfb_id_source {
  ttl.dfb.split @split 0 : index -> [1, 2] policy = <contiguous>
}

// -----

// Verify that the split op with symbol source round-trips.
// CHECK-LABEL: ttl.dfb.network @split_net_symbol_source {
// CHECK-NEXT: ttl.dfb.merge @m [1 : index, 2 : index] policy = <first_ready>
// CHECK-NEXT: ttl.dfb.split @split @m -> [3, 4] policy = <round_robin>
// CHECK-NEXT: }
ttl.dfb.network @split_net_symbol_source {
  ttl.dfb.merge @m [1 : index, 2 : index] policy = <first_ready>
  ttl.dfb.split @split @m -> [3, 4] policy = <round_robin>
}

// -----

// Verify that the merge op round-trips with DFB id inputs.
// CHECK-LABEL: ttl.dfb.network @merge_net_dfb_ids {
// CHECK-NEXT: ttl.dfb.merge @merge [1 : index, 2 : index] policy = <round_robin>
// CHECK-NEXT: }
ttl.dfb.network @merge_net_dfb_ids {
  ttl.dfb.merge @merge [1 : index, 2 : index] policy = <round_robin>
}

// -----

// Verify that the merge op round-trips with symbol inputs.
// CHECK-LABEL: ttl.dfb.network @merge_net_symbols {
// CHECK-NEXT: ttl.dfb.merge @m1 [1 : index, 2 : index] policy = <round_robin>
// CHECK-NEXT: ttl.dfb.merge @m2 [3 : index, 4 : index] policy = <round_robin>
// CHECK-NEXT: ttl.dfb.merge @merge_final [@m1, @m2] policy = <round_robin>
// CHECK-NEXT: }
ttl.dfb.network @merge_net_symbols {
  ttl.dfb.merge @m1 [1 : index, 2 : index] policy = <round_robin>
  ttl.dfb.merge @m2 [3 : index, 4 : index] policy = <round_robin>
  ttl.dfb.merge @merge_final [@m1, @m2] policy = <round_robin>
}

// -----

// Verify that the merge op round-trips with mixed DFB id and symbol inputs.
// CHECK-LABEL: ttl.dfb.network @merge_net_mixed_inputs {
// CHECK-NEXT: ttl.dfb.merge @m1 [1 : index, 2 : index] policy = <round_robin>
// CHECK-NEXT: ttl.dfb.merge @m2 [3 : index, 4 : index] policy = <round_robin>
// CHECK-NEXT: ttl.dfb.merge @merge_final [@m1, @m2, 5 : index] policy = <round_robin>
// CHECK-NEXT: }
ttl.dfb.network @merge_net_mixed_inputs {
  ttl.dfb.merge @m1 [1 : index, 2 : index] policy = <round_robin>
  ttl.dfb.merge @m2 [3 : index, 4 : index] policy = <round_robin>
  ttl.dfb.merge @merge_final [@m1, @m2, 5 : index] policy = <round_robin>
}
