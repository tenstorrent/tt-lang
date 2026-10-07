// Round-trip of ttl.dfb.network: the op parses and prints back unchanged.
// Canonicalization must not remove it (it has no results but is not Pure),
// and symbol DCE must not remove it (a network is public by default).

// RUN: ttlang-opt %s | FileCheck %s
// RUN: ttlang-opt %s --canonicalize | FileCheck %s
// RUN: ttlang-opt %s --symbol-dce | FileCheck %s

// Verify an empty network round-trips.
// CHECK-LABEL: ttl.dfb.network @net0 {
// CHECK-NEXT: }
module {
  ttl.dfb.network @net0 {
  }
}
