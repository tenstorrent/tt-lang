// Tests a successful fixed-limit exact check for every logical-ID permutation.
// RUN: %python %S/Inputs/generate_dfb_exact_coloring_permutations.py | ttlang-opt --split-input-file -pass-pipeline='builtin.module(ttl-finalize-dfb-indices)' | FileCheck %s

// Thirty mutually conflicting DFBs join a four-vertex path declared in the
// first-fit order A,D,B,C. First-fit requires 33 indices. Exact coloring finds
// a 32-index assignment for all 24 permutations of the path's logical IDs.

// CHECK-COUNT-24: ttl.base_cta_index = 32 : i32
// CHECK-NOT: ttl.base_cta_index = 33
