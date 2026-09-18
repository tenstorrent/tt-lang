// RUN: ttlang-execution-count-test %s | FileCheck %s

// This file verifies API queries for operations that are not nested in the
// analysis root region.

// A parentless module and an operation in another function are both outside
// the selected function-body root and therefore have unknown counts.
module attributes {
    test.expected_count = "unknown",
    test.label = "parentless_operation"} {
  func.func @analysis_root() attributes {test.analysis_root} {
    return
  }

  func.func @outside_root() {
    %zero = arith.constant 0 : index
    %target = arith.addi %zero, %zero {
      test.expected_count = "unknown",
      test.label = "outside_root"
    } : index
    return
  }
}

// CHECK-LABEL: outside_root = unknown
// CHECK-LABEL: parentless_operation = unknown
