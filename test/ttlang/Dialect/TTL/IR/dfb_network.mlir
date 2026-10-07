// RUN: ttlang-opt %s --split-input-file | FileCheck %s
// RUN: ttlang-opt %s --split-input-file --canonicalize | FileCheck %s
// RUN: ttlang-opt %s --split-input-file --symbol-dce | FileCheck %s

// Summary: Verifies that a network round-trips and that canonicalization and
// symbol DCE keep it (it is not Pure and is public by default). Also verifies
// that the network enum attributes round-trip.

// Verify an empty network round-trips.
// CHECK-LABEL: ttl.dfb.network @empty {
// CHECK-NEXT: }
ttl.dfb.network @empty {
}

// -----

// Verify that the fork storage options round-trip.
// CHECK-LABEL: ttl.dfb.network @fork_storage attributes
// CHECK-SAME: {a = #ttl.dfb_network_fork_storage<replicated>,
// CHECK-SAME: b = #ttl.dfb_network_fork_storage<shared>,
// CHECK-SAME: c = #ttl.dfb_network_fork_storage<auto>}
// CHECK-NEXT: }
ttl.dfb.network @fork_storage attributes
 {a = #ttl.dfb_network_fork_storage<replicated>,
  b = #ttl.dfb_network_fork_storage<shared>,
  c = #ttl.dfb_network_fork_storage<auto>} {
}

// -----

// Verify that the split policy options round-trip.
// CHECK-LABEL: ttl.dfb.network @split_policy attributes
// CHECK-SAME: {a = #ttl.dfb_network_split_policy<round_robin>,
// CHECK-SAME: b = #ttl.dfb_network_split_policy<contiguous>}
// CHECK-NEXT: }
ttl.dfb.network @split_policy attributes
 {a = #ttl.dfb_network_split_policy<round_robin>,
  b = #ttl.dfb_network_split_policy<contiguous>} {
}

// -----

// Verify that the merge policy options round-trip.
// CHECK-LABEL: ttl.dfb.network @merge_policy attributes
// CHECK-SAME: {a = #ttl.dfb_network_merge_policy<round_robin>,
// CHECK-SAME: b = #ttl.dfb_network_merge_policy<first_ready>}
// CHECK-NEXT: }
ttl.dfb.network @merge_policy attributes
 {a = #ttl.dfb_network_merge_policy<round_robin>,
  b = #ttl.dfb_network_merge_policy<first_ready>} {
}
