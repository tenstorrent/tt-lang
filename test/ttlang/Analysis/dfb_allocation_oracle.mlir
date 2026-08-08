// Tests minimum and fixed-limit DFB index assignment against an independent
// exhaustive oracle and compares uniform, per-node, and hybrid contracts.
// RUN: ttlang-dfb-allocation-oracle-test | FileCheck %s

// CHECK: solver_graphs=33868
// CHECK-NEXT: capacity_reproducer=32
// CHECK-NEXT: capacity_search_states={{[1-9][0-9]*}}
// CHECK-NEXT: bounded_search_states=1
// CHECK-NEXT: fixed_limit_states={{[1-9][0-9]*}}
// CHECK-NEXT: minimum_proof_states=100
// CHECK-NEXT: contract_cases=262144
// CHECK-NEXT: per_node_improvements=149268
// CHECK-NEXT: two_group_improvements=142536
// CHECK-NEXT: maximum_uniform_penalty=2
