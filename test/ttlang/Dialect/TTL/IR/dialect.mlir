// RUN: ttlang-opt %s | FileCheck %s

// CHECK-LABEL: func.func @test_ttl_dialect
func.func @test_ttl_dialect() {
  // Verify that the TTL dialect is loaded
  return
}
