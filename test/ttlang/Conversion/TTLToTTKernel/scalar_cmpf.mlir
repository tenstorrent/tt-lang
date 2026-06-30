// Tests for ttl-lower-scalar-fp-types pass: uses MLIR dialect conversion to
// lower scalar float types to integer bit patterns. Covers arith.cmpf -> soft-
// float ops, arith.truncf -> bit extraction, arith.constant float -> integer
// bit patterns, and SCF control-flow type propagation.
// RUN: ttlang-opt --ttl-lower-scalar-fp-types --canonicalize -cse --split-input-file %s | FileCheck %s

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

// -----

// arith.truncf f32 -> bf16 lowered to bit extraction (shrui + trunci).
// CHECK-LABEL: func.func @truncf_f32_to_bf16
// CHECK-SAME: (%[[ARG:.*]]: i32) -> i16
// CHECK-NOT: arith.truncf
// CHECK: %[[SHIFT:.*]] = arith.constant 16 : i32
// CHECK: %[[SHIFTED:.*]] = arith.shrui %[[ARG]], %[[SHIFT]] : i32
// CHECK: %[[RESULT:.*]] = arith.trunci %[[SHIFTED]] : i32 to i16
// CHECK: return %[[RESULT]] : i16
module {
  func.func @truncf_f32_to_bf16(%a_int: i32) -> bf16 {
    %a = builtin.unrealized_conversion_cast %a_int : i32 to f32
    %b = arith.truncf %a : f32 to bf16
    return %b : bf16
  }
}

// -----

// arith.constant float -> integer bit pattern.
// 1.0f = 0x3F800000 = 1065353216
// CHECK-LABEL: func.func @constant_f32
// CHECK-NOT: arith.constant{{.*}}: f32
// CHECK: %[[BITS:.*]] = arith.constant 1065353216 : i32
// CHECK: return %[[BITS]] : i32
module {
  func.func @constant_f32() -> f32 {
    %c = arith.constant 1.000000e+00 : f32
    return %c : f32
  }
}

// -----

// bf16 constant -> integer bit pattern.
// 2.5 bf16 = 0x4020 = 16416
// CHECK-LABEL: func.func @constant_bf16
// CHECK-NOT: arith.constant{{.*}}: bf16
// CHECK: %[[BITS:.*]] = arith.constant 16416 : i16
// CHECK: return %[[BITS]] : i16
module {
  func.func @constant_bf16() -> bf16 {
    %c = arith.constant 2.500000e+00 : bf16
    return %c : bf16
  }
}

// -----

// scf.if with float result: type propagation converts yield/result to integer.
// Canonicalize simplifies a trivial if/else to arith.select.
// CHECK-LABEL: func.func @scf_if_float_result
// CHECK-SAME: (%[[COND:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32) -> i32
// CHECK-NOT: scf.if
// CHECK: %[[RES:.*]] = arith.select %[[COND]], %[[A]], %[[B]] : i32
// CHECK: return %[[RES]] : i32
module {
  func.func @scf_if_float_result(%cond: i1, %a_int: i32, %b_int: i32) -> f32 {
    %a = builtin.unrealized_conversion_cast %a_int : i32 to f32
    %b = builtin.unrealized_conversion_cast %b_int : i32 to f32
    %res = scf.if %cond -> (f32) {
      scf.yield %a : f32
    } else {
      scf.yield %b : f32
    }
    return %res : f32
  }
}

// -----

// scf.for with float iter_arg: type propagation converts iter_arg and yield.
// Canonicalize folds away the identity loop, leaving just the init value.
// CHECK-LABEL: func.func @scf_for_float_iter_arg
// CHECK-SAME: (%[[INIT:.*]]: i32) -> i32
// CHECK-NOT: scf.for
// CHECK: return %[[INIT]] : i32
module {
  func.func @scf_for_float_iter_arg(%init_int: i32) -> f32 {
    %init = builtin.unrealized_conversion_cast %init_int : i32 to f32
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %res = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %init) -> (f32) {
      scf.yield %acc : f32
    }
    return %res : f32
  }
}
