// RUN: ttlang-opt --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.emitc.mlir --check-prefix=EMITC
// RUN: ttlang-translate --ttkernel-to-cpp %t.emitc.mlir > %t.cpp
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=COUNT

// Scalar float arithmetic reinterprets the integer-backed operands and results
// without invoking an external operation.

// EMITC-LABEL: func.func @kernel_main
// EMITC-COUNT-3: emitc.verbatim
// EMITC-NOT: ttkernel.float32_

// CPP-LABEL: void kernel_main()
// CPP: float {{.*}} = {{.*}} + {{.*}};
// CPP: float {{.*}} = {{.*}} - {{.*}};
// CPP: float {{.*}} = {{.*}} * {{.*}};
// CPP-NOT: float32_add
// CPP-NOT: float32_sub
// CPP-NOT: float32_mul
// COUNT-COUNT-9: __builtin_memcpy
func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %lhs = arith.constant 1069547520 : i32
  %rhs = arith.constant -1073741824 : i32
  %sum = ttkernel.float32_add(%lhs, %rhs) : (i32, i32) -> i32
  %difference = ttkernel.float32_sub(%sum, %rhs) : (i32, i32) -> i32
  %product = ttkernel.float32_mul(%difference, %lhs) : (i32, i32) -> i32
  return
}
