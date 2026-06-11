// Tests for ttl-lower-scalar-cmpf pass: lowers arith.cmpf on float scalars
// (from raw_element_read bit patterns) to TTKernel soft-float comparison ops.
// RUN: ttlang-opt --ttl-lower-scalar-cmpf --canonicalize -cse --split-input-file %s | FileCheck %s

// -----

// f32 ogt -> ttkernel.float32_greater(lhs, rhs) on signless i32 bit patterns
// CHECK-LABEL: func.func @cmpf_ogt_f32
// CHECK-NOT: arith.cmpf
// CHECK: ttkernel.float32_greater(
// CHECK-SAME: ) : (i32, i32) -> i1
module {
  func.func @cmpf_ogt_f32(%a_int: i32, %b_int: i32) -> i1 {
    %a = builtin.unrealized_conversion_cast %a_int : i32 to f32
    %b = builtin.unrealized_conversion_cast %b_int : i32 to f32
    %cmp = arith.cmpf ogt, %a, %b : f32
    return %cmp : i1
  }
}

// -----

// bf16 ogt -> ttkernel.bfloat16_greater(lhs, rhs) on signless i16 bit patterns
// CHECK-LABEL: func.func @cmpf_ogt_bf16
// CHECK-NOT: arith.cmpf
// CHECK: ttkernel.bfloat16_greater(
// CHECK-SAME: ) : (i16, i16) -> i1
module {
  func.func @cmpf_ogt_bf16(%a_int: i16, %b_int: i16) -> i1 {
    %a = builtin.unrealized_conversion_cast %a_int : i16 to bf16
    %b = builtin.unrealized_conversion_cast %b_int : i16 to bf16
    %cmp = arith.cmpf ogt, %a, %b : bf16
    return %cmp : i1
  }
}

// -----

// f32 olt -> ttkernel.float32_greater with swapped operands
// CHECK-LABEL: func.func @cmpf_olt_f32
// CHECK-NOT: arith.cmpf
// CHECK: ttkernel.float32_greater(
module {
  func.func @cmpf_olt_f32(%a_int: i32, %b_int: i32) -> i1 {
    %a = builtin.unrealized_conversion_cast %a_int : i32 to f32
    %b = builtin.unrealized_conversion_cast %b_int : i32 to f32
    %cmp = arith.cmpf olt, %a, %b : f32
    return %cmp : i1
  }
}

// -----

// Constant float operand: 1.0f (0x3F800000) materialized as integer constant.
// CHECK-LABEL: func.func @cmpf_ogt_f32_constant
// CHECK-DAG: %[[BITS:.*]] = arith.constant 1065353216 : i32
// CHECK: ttkernel.float32_greater(
module {
  func.func @cmpf_ogt_f32_constant(%a_int: i32) -> i1 {
    %a = builtin.unrealized_conversion_cast %a_int : i32 to f32
    %one = arith.constant 1.0 : f32
    %cmp = arith.cmpf ogt, %a, %one : f32
    return %cmp : i1
  }
}

// -----

// bf16 olt -> ttkernel.bfloat16_greater with swapped operands
// CHECK-LABEL: func.func @cmpf_olt_bf16
// CHECK-NOT: arith.cmpf
// CHECK: ttkernel.bfloat16_greater(
// CHECK-SAME: ) : (i16, i16) -> i1
module {
  func.func @cmpf_olt_bf16(%a_int: i16, %b_int: i16) -> i1 {
    %a = builtin.unrealized_conversion_cast %a_int : i16 to bf16
    %b = builtin.unrealized_conversion_cast %b_int : i16 to bf16
    %cmp = arith.cmpf olt, %a, %b : bf16
    return %cmp : i1
  }
}
