// Tests a successful fixed-limit exact check for every logical-ID permutation.
// RUN: %python %S/Inputs/generate_dfb_exact_coloring_permutations.py | ttlang-opt --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s

// Thirty mutually conflicting DFBs join a four-DFB conflict chain declared in
// first-fit order A,D,B,C. The first group requires 30 distinct indices; the
// chain makes first-fit use three more although two suffice. Exhaustive
// fixed-limit search finds a 32-index assignment for all 24 permutations of the
// chain's logical IDs.

// CHECK-COUNT-24: ttl.base_cta_index = 32 : i32
// CHECK-NOT: ttl.base_cta_index = 33
