// Tests for ttkernel-lower-scalar-fp-types pass: uses MLIR dialect conversion to
// lower scalar float types to integer bit patterns. Covers arith.cmpf -> soft-
// float ops, arith.truncf -> bit extraction, arith.constant float -> integer
// bit patterns, and SCF control-flow type propagation.
// RUN: ttlang-opt --ttkernel-lower-scalar-fp-types --split-input-file %s | FileCheck %s

// -----

// f32 ogt -> ttkernel.float32_greater(lhs, rhs) on signless i32 bit patterns
// CHECK-LABEL: func.func @cmpf_ogt_f32
// CHECK-SAME: (%[[A:.*]]: i32, %[[B:.*]]: i32) -> i1
// CHECK-NOT: arith.cmpf
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[CMP:.*]] = ttkernel.float32_greater(%[[A]], %[[B]]) : (i32, i32) -> i1
// CHECK-NEXT: return %[[CMP]] : i1
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
// CHECK-SAME: (%[[A:.*]]: i16, %[[B:.*]]: i16) -> i1
// CHECK-NOT: arith.cmpf
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[CMP:.*]] = ttkernel.bfloat16_greater(%[[A]], %[[B]]) : (i16, i16) -> i1
// CHECK-NEXT: return %[[CMP]] : i1
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
// CHECK-SAME: (%[[A:.*]]: i32, %[[B:.*]]: i32) -> i1
// CHECK-NOT: arith.cmpf
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[CMP:.*]] = ttkernel.float32_greater(%[[B]], %[[A]]) : (i32, i32) -> i1
// CHECK-NEXT: return %[[CMP]] : i1
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
// CHECK-SAME: (%[[A:.*]]: i32) -> i1
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[BITS:.*]] = arith.constant 1065353216 : i32
// CHECK: %[[CMP:.*]] = ttkernel.float32_greater(%[[A]], %[[BITS]]) : (i32, i32) -> i1
// CHECK-NEXT: return %[[CMP]] : i1
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
// CHECK-SAME: (%[[A:.*]]: i16, %[[B:.*]]: i16) -> i1
// CHECK-NOT: arith.cmpf
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[CMP:.*]] = ttkernel.bfloat16_greater(%[[B]], %[[A]]) : (i16, i16) -> i1
// CHECK-NEXT: return %[[CMP]] : i1
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
// bf16 is the upper 16 bits of f32, so this is just a shift-right by 16.
// CHECK-LABEL: func.func @truncf_f32_to_bf16
// CHECK-SAME: (%[[ARG:.*]]: i32) -> i16
// CHECK-NOT: arith.truncf
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[SHIFT:.*]] = arith.constant 16 : i32
// CHECK: %[[SHIFTED:.*]] = arith.shrui %[[ARG]], %[[SHIFT]] : i32
// CHECK-NEXT: %[[RESULT:.*]] = arith.trunci %[[SHIFTED]] : i32 to i16
// CHECK-NEXT: return %[[RESULT]] : i16
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
// CHECK-SAME: () -> i32
// CHECK-NOT: arith.constant{{.*}}: f32
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[BITS:.*]] = arith.constant 1065353216 : i32
// CHECK-NEXT: return %[[BITS]] : i32
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
// CHECK-SAME: () -> i16
// CHECK-NOT: arith.constant{{.*}}: bf16
// CHECK-NOT: builtin.unrealized_conversion_cast
// CHECK: %[[BITS:.*]] = arith.constant 16416 : i16
// CHECK-NEXT: return %[[BITS]] : i16
module {
  func.func @constant_bf16() -> bf16 {
    %c = arith.constant 2.500000e+00 : bf16
    return %c : bf16
  }
}

// -----

// scf.if with float result: type propagation converts yield/result to integer.
// CHECK-LABEL: func.func @scf_if_float_result
// CHECK-SAME: (%[[COND:.*]]: i1, %[[A:.*]]: i32, %[[B:.*]]: i32) -> i32
// CHECK: %[[RES:.*]] = scf.if %[[COND]] -> (i32)
// CHECK-NEXT:   scf.yield %[[A]] : i32
// CHECK-NEXT: } else {
// CHECK-NEXT:   scf.yield %[[B]] : i32
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
// CHECK-LABEL: func.func @scf_for_float_iter_arg
// CHECK-SAME: (%[[INIT:.*]]: i32) -> i32
// CHECK: %[[RES:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (i32)
// CHECK-NEXT:   scf.yield %[[ACC]] : i32
// CHECK: return %[[RES]] : i32
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

// -----

// Nested control flow: scf.for with float iter_arg, scf.if in loop body
// compares accumulator against threshold and conditionally clamps it.
// Float types propagate through both the for iter_arg and the if result.
// CHECK-LABEL: func.func @nested_for_if_clamp
// CHECK-SAME: (%[[CUR:.*]]: i32, %[[THRESH:.*]]: i32) -> i32
// CHECK-NOT: arith.cmpf
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[CUR]]) -> (i32)
// CHECK:   %[[GT:.*]] = ttkernel.float32_greater(%[[ACC]], %[[THRESH]]) : (i32, i32) -> i1
// CHECK-NEXT:   %[[NEXT:.*]] = scf.if %[[GT]] -> (i32)
// CHECK-NEXT:     scf.yield %[[THRESH]] : i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     scf.yield %[[ACC]] : i32
// CHECK:   scf.yield %[[NEXT]] : i32
// CHECK: return
module {
  func.func @nested_for_if_clamp(%cur_int: i32, %thresh_int: i32) -> f32 {
    %cur = builtin.unrealized_conversion_cast %cur_int : i32 to f32
    %thresh = builtin.unrealized_conversion_cast %thresh_int : i32 to f32
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %i = %c0 to %c10 step %c1 iter_args(%acc = %cur) -> (f32) {
      %gt = arith.cmpf ogt, %acc, %thresh : f32
      %next = scf.if %gt -> (f32) {
        scf.yield %thresh : f32
      } else {
        scf.yield %acc : f32
      }
      scf.yield %next : f32
    }
    return %result : f32
  }
}

// -----

// Deeply nested control flow: scf.for with scf.if inside scf.if.
// Implements a clamp loop: each iteration clamps the accumulator to [lo, hi].
// The outer if checks >hi, the inner else-branch checks <lo.
// CHECK-LABEL: func.func @nested_for_if_if_clamp
// CHECK-SAME: (%[[VAL:.*]]: i32, %[[LO:.*]]: i32, %[[HI:.*]]: i32) -> i32
// CHECK-NOT: arith.cmpf
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[VAL]]) -> (i32)
// CHECK:   %[[GT_HI:.*]] = ttkernel.float32_greater(%[[ACC]], %[[HI]]) : (i32, i32) -> i1
// CHECK-NEXT:   %[[OUTER:.*]] = scf.if %[[GT_HI]] -> (i32)
// CHECK-NEXT:     scf.yield %[[HI]] : i32
// CHECK-NEXT:   } else {
// CHECK-NEXT:     %[[LT_LO:.*]] = ttkernel.float32_greater(%[[LO]], %[[ACC]]) : (i32, i32) -> i1
// CHECK-NEXT:     %[[INNER:.*]] = scf.if %[[LT_LO]] -> (i32)
// CHECK-NEXT:       scf.yield %[[LO]] : i32
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %[[ACC]] : i32
// CHECK:     scf.yield %[[INNER]] : i32
// CHECK:   scf.yield %[[OUTER]] : i32
// CHECK: return
module {
  func.func @nested_for_if_if_clamp(%val_int: i32, %lo_int: i32,
                                     %hi_int: i32) -> f32 {
    %val = builtin.unrealized_conversion_cast %val_int : i32 to f32
    %lo = builtin.unrealized_conversion_cast %lo_int : i32 to f32
    %hi = builtin.unrealized_conversion_cast %hi_int : i32 to f32
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %val) -> (f32) {
      %above_hi = arith.cmpf ogt, %acc, %hi : f32
      %clamped = scf.if %above_hi -> (f32) {
        scf.yield %hi : f32
      } else {
        %below_lo = arith.cmpf olt, %acc, %lo : f32
        %inner = scf.if %below_lo -> (f32) {
          scf.yield %lo : f32
        } else {
          scf.yield %acc : f32
        }
        scf.yield %inner : f32
      }
      scf.yield %clamped : f32
    }
    return %result : f32
  }
}
