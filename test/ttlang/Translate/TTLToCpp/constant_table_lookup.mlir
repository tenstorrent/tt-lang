// RUN: ttlang-opt --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.emitc.mlir --check-prefix=EMITC
// RUN: ttlang-translate --ttkernel-to-cpp %t.emitc.mlir > %t.cpp
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=NUM

// Verify that TTKernel tables fitting a native 32-bit word become packed
// immediates while larger tables use bit-packed static C++ storage outside
// kernel_main.

// EMITC-DAG: emitc.global static const @__ttlang_constant_table_{{[A-F0-9]+}} : !emitc.array<1xui64> = #emitc.opaque<"{0x876543210ULL}">
// EMITC-DAG: emitc.global static const @__ttlang_constant_table_{{[A-F0-9]+}} : !emitc.array<2xui64> = #emitc.opaque<"{0xC5A928398A418820ULL, 0x107B9AULL}">
// EMITC-LABEL: func.func @kernel_main
// EMITC: emitc.call_opaque "experimental::constant_table_lookup_word"
// EMITC: emitc.call_opaque "experimental::constant_table_lookup_word"
// EMITC: emitc.get_global @__ttlang_constant_table_{{[A-F0-9]+}}
// EMITC: emitc.call_opaque "experimental::constant_table_lookup"
// EMITC: emitc.get_global @__ttlang_constant_table_{{[A-F0-9]+}}
// EMITC: emitc.call_opaque "experimental::constant_table_lookup"
// EMITC: emitc.call_opaque "experimental::constant_table_lookup_word"

// NUM-COUNT-2: static const uint64_t __ttlang_constant_table_
// CPP: #include <cstdint>
// CPP-DAG: static const uint64_t __ttlang_constant_table_{{[A-F0-9]+}}[1] = {0x876543210ULL};
// CPP-DAG: static const uint64_t __ttlang_constant_table_{{[A-F0-9]+}}[2] = {0xC5A928398A418820ULL, 0x107B9AULL};
// CPP-LABEL: void kernel_main()
// CPP-NOT: size_t v{{[0-9]+}}[3];
// Values [3, 5, 8] require four bits each and pack into 0x853.
// CPP: experimental::constant_table_lookup_word<4>({{.*}}, 0x853U)
// A 32-bit value remains a native-word immediate.
// CPP: experimental::constant_table_lookup_word<32>({{.*}}, 0x80000000U)
// Nine four-bit values require static 64-bit storage.
// CPP: experimental::constant_table_lookup<4>({{.*}}, __ttlang_constant_table_{{[A-F0-9]+}})
// Seventeen five-bit values cross a 64-bit boundary.
// CPP: experimental::constant_table_lookup<5>({{.*}}, __ttlang_constant_table_{{[A-F0-9]+}})
// Repeated values reuse the same packed-word literal.
// CPP: experimental::constant_table_lookup_word<4>({{.*}}, 0x853U)
func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %zero = arith.constant 0 : index
  %index = arith.constant 1 : index
  %value = ttkernel.experimental.constant_table_lookup %index, [3, 5, 8] : index
  %word32 = ttkernel.experimental.constant_table_lookup %zero,
      [2147483648] : index
  %word64 = ttkernel.experimental.constant_table_lookup %index,
      [0, 1, 2, 3, 4, 5, 6, 7, 8] : index
  %wide = ttkernel.experimental.constant_table_lookup %index,
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] : index
  %duplicate = ttkernel.experimental.constant_table_lookup %index,
      [3, 5, 8] : index
  return
}
