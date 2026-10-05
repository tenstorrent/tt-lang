// RUN: ttlang-opt --convert-ttkernel-to-emitc -o %t.emitc.mlir %s
// RUN: FileCheck %s --input-file=%t.emitc.mlir --check-prefix=EMITC
// RUN: ttlang-translate --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// Verify unsigned division and remainder of index values lower to size_t C++
// operators.

// EMITC-LABEL: func.func @kernel_main
// EMITC: %[[QUOTIENT:.*]] = emitc.div
// EMITC: %[[REMAINDER:.*]] = emitc.rem
// EMITC: emitc.add %[[QUOTIENT]], %[[REMAINDER]]

// CPP-LABEL: void kernel_main()
// CPP: size_t [[INDEX:.*]] = get_absolute_logical_x();
// CPP: size_t [[DIVISOR:.*]] = 4;
// CPP: size_t [[QUOTIENT:.*]] = [[INDEX]] / [[DIVISOR]];
// CPP: size_t [[REMAINDER:.*]] = [[INDEX]] % [[DIVISOR]];
// CPP: size_t {{.*}} = [[QUOTIENT]] + [[REMAINDER]];

func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %index = "ttkernel.my_logical_x_"() : () -> index
  %divisor = arith.constant 4 : index
  %quotient = arith.divui %index, %divisor : index
  %remainder = arith.remui %index, %divisor : index
  %sum = arith.addi %quotient, %remainder : index
  return
}
