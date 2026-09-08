// Summary: Tests DFB index assignment and compiler L1 byte placement against
// independent exhaustive oracles and compares allocation-domain contracts.
// RUN: ttlang-dfb-allocation-oracle-test | FileCheck %s

// CHECK: solver_graphs=33868
// CHECK-NEXT: capacity_reproducer=32
// CHECK-NEXT: capacity_search_states={{[1-9][0-9]*}}
// CHECK-NEXT: bounded_search_states=1
// CHECK-NEXT: fixed_limit_states={{[1-9][0-9]*}}
// CHECK-NEXT: minimum_proof_states=100
// CHECK-NEXT: weighted_solver_cases=13689
// CHECK-NEXT: l1_multi_order_suboptimal_cases=68
// CHECK-NEXT: l1_multi_order_excess_units=80
// CHECK-NEXT: l1_multi_order_aggregate_efficiency_basis_points=9969
// CHECK-NEXT: l1_multi_order_worst_efficiency_basis_points=7142
// CHECK-NEXT: l1_placement_cases=5184
// CHECK-NEXT: l1_first_fit_suboptimal_cases=149
// CHECK-NEXT: l1_best_fit_suboptimal_cases=149
// CHECK-NEXT: l1_first_fit_excess_units=196
// CHECK-NEXT: l1_best_fit_excess_units=196
// CHECK-NEXT: l1_first_fit_max_excess_units=3
// CHECK-NEXT: l1_best_fit_max_excess_units=3
// CHECK-NEXT: l1_first_fit_aggregate_efficiency_basis_points=9926
// CHECK-NEXT: l1_best_fit_aggregate_efficiency_basis_points=9926
// CHECK-NEXT: l1_first_fit_suboptimal_efficiency_basis_points=7874
// CHECK-NEXT: l1_best_fit_suboptimal_efficiency_basis_points=7874
// CHECK-NEXT: l1_first_fit_worst_efficiency_basis_points=6666
// CHECK-NEXT: l1_best_fit_worst_efficiency_basis_points=6666
// CHECK-NEXT: l1_multi_order_large_cases=160
// CHECK-NEXT: target_capacities=32, 64, 32, 32
// CHECK-NEXT: system_desc_num_cbs=64, 32
// CHECK-NEXT: contract_cases=262144
// CHECK-NEXT: per_node_improvements=149268
// CHECK-NEXT: two_group_improvements=142536
// CHECK-NEXT: maximum_uniform_penalty=2
