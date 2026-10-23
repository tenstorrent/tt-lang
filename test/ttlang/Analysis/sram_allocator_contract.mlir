// The replaceable SRAM allocator rejects invalid input and faulty strategies.
// RUN: ttlang-sram-allocator-contract-test | FileCheck %s

// CHECK: allocator_contract_cases=7
