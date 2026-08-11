// Tests scalar float arithmetic conversion to integer-backed TTKernel ops.
// RUN: ttlang-opt --ttkernel-lower-scalar-fp-types --split-input-file %s | FileCheck %s

// -----

// CHECK-LABEL: func.func @f32_arithmetic
// CHECK-SAME: (%[[LHS:.*]]: i32, %[[RHS:.*]]: i32) -> i32
// CHECK-NOT: arith.{{addf|subf|mulf}}
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[ADD:.*]] = ttkernel.float32_add(%[[LHS]], %[[RHS]]) : (i32, i32) -> i32
// CHECK: %[[SUB:.*]] = ttkernel.float32_sub(%[[ADD]], %[[RHS]]) : (i32, i32) -> i32
// CHECK: %[[MUL:.*]] = ttkernel.float32_mul(%[[SUB]], %[[LHS]]) : (i32, i32) -> i32
// CHECK: return %[[MUL]] : i32
module {
  func.func @f32_arithmetic(%lhs_bits: i32, %rhs_bits: i32) -> f32 {
    %lhs = builtin.unrealized_conversion_cast %lhs_bits : i32 to f32
    %rhs = builtin.unrealized_conversion_cast %rhs_bits : i32 to f32
    %sum = arith.addf %lhs, %rhs : f32
    %difference = arith.subf %sum, %rhs : f32
    %product = arith.mulf %difference, %lhs : f32
    return %product : f32
  }
}

// -----

// CHECK-LABEL: func.func @bf16_promoted_arithmetic
// CHECK-SAME: (%[[LHS:.*]]: i16, %[[RHS:.*]]: i16) -> i32
// CHECK-NOT: arith.extf
// CHECK-NOT: arith.mulf
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[LHS_EXT:.*]] = arith.extui %[[LHS]] : i16 to i32
// CHECK: %[[SHIFT:.*]] = arith.constant 16 : i32
// CHECK: %[[LHS_F32:.*]] = arith.shli %[[LHS_EXT]], %[[SHIFT]] : i32
// CHECK: %[[RHS_EXT:.*]] = arith.extui %[[RHS]] : i16 to i32
// CHECK: %[[RHS_SHIFT:.*]] = arith.constant 16 : i32
// CHECK: %[[RHS_F32:.*]] = arith.shli %[[RHS_EXT]], %[[RHS_SHIFT]] : i32
// CHECK: %[[MUL:.*]] = ttkernel.float32_mul(%[[LHS_F32]], %[[RHS_F32]]) : (i32, i32) -> i32
// CHECK: return %[[MUL]] : i32
module {
  func.func @bf16_promoted_arithmetic(%lhs_bits: i16, %rhs_bits: i16) -> f32 {
    %lhs_bf16 = builtin.unrealized_conversion_cast %lhs_bits : i16 to bf16
    %rhs_bf16 = builtin.unrealized_conversion_cast %rhs_bits : i16 to bf16
    %lhs = arith.extf %lhs_bf16 : bf16 to f32
    %rhs = arith.extf %rhs_bf16 : bf16 to f32
    %product = arith.mulf %lhs, %rhs : f32
    return %product : f32
  }
}

// -----

// CHECK-LABEL: func.func @loop_carried_multiply_add
// CHECK-SAME: (%[[LHS:.*]]: i32, %[[RHS:.*]]: i32, %[[ADDEND:.*]]: i32) -> i32
// CHECK-NOT: arith.{{addf|mulf}}
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[RESULT:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[ADDEND]]) -> (i32) {
// CHECK:   %[[PRODUCT:.*]] = ttkernel.float32_mul(%[[LHS]], %[[RHS]]) : (i32, i32) -> i32
// CHECK:   %[[NEXT:.*]] = ttkernel.float32_add(%[[ACC]], %[[PRODUCT]]) : (i32, i32) -> i32
// CHECK:   scf.yield %[[NEXT]] : i32
// CHECK: return %[[RESULT]] : i32
module {
  func.func @loop_carried_multiply_add(
      %lhs_bits: i32, %rhs_bits: i32, %addend_bits: i32) -> f32 {
    %lhs = builtin.unrealized_conversion_cast %lhs_bits : i32 to f32
    %rhs = builtin.unrealized_conversion_cast %rhs_bits : i32 to f32
    %addend = builtin.unrealized_conversion_cast %addend_bits : i32 to f32
    %lower = arith.constant 0 : index
    %upper = arith.constant 4 : index
    %step = arith.constant 1 : index
    %result = scf.for %iteration = %lower to %upper step %step
        iter_args(%accumulator = %addend) -> (f32) {
      %product = arith.mulf %lhs, %rhs : f32
      %next = arith.addf %accumulator, %product : f32
      scf.yield %next : f32
    }
    return %result : f32
  }
}
