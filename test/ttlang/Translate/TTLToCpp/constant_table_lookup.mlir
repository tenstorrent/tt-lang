// RUN: ttlang-opt --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.emitc.mlir --check-prefix=EMITC
// RUN: ttlang-translate --ttkernel-to-cpp %t.emitc.mlir > %t.cpp
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=NUM

// Verify that immutable TTKernel tables become bit-packed static C++ storage
// without variadic template arguments or arrays allocated in kernel_main.

// EMITC: emitc.global static const @__ttlang_constant_table_{{[A-F0-9]+}} : !emitc.array<2xui64> = #emitc.opaque<"{0xC5A928398A418820ULL, 0x107B9AULL}">
// EMITC-NEXT: emitc.global static const @__ttlang_constant_table_{{[A-F0-9]+}} : !emitc.array<1xui64> = #emitc.opaque<"{0x853ULL}">
// EMITC-LABEL: func.func @kernel_main
// EMITC: emitc.get_global @[[NARROW:__ttlang_constant_table_[A-F0-9]+]]
// EMITC: emitc.call_opaque "experimental::constant_table_lookup"
// EMITC: emitc.get_global @__ttlang_constant_table_{{[A-F0-9]+}}
// EMITC: emitc.call_opaque "experimental::constant_table_lookup"
// EMITC: emitc.get_global @[[NARROW]]
// EMITC: emitc.call_opaque "experimental::constant_table_lookup"

// NUM-COUNT-2: static const uint64_t __ttlang_constant_table_
// CPP: #include <cstdint>
// CPP: static const uint64_t __ttlang_constant_table_{{[A-F0-9]+}}[1] = {0x853ULL};
// CPP-NEXT: static const uint64_t __ttlang_constant_table_{{[A-F0-9]+}}[2] = {0xC5A928398A418820ULL, 0x107B9AULL};
// CPP-LABEL: void kernel_main()
// CPP-NOT: size_t v{{[0-9]+}}[3];
// Values [3, 5, 8] require four bits each and pack into 0x853.
// CPP: experimental::constant_table_lookup<4>({{.*}}, [[NARROW_CPP:__ttlang_constant_table_[A-F0-9]+]])
// Seventeen five-bit values cross a 64-bit boundary.
// CPP: experimental::constant_table_lookup<5>({{.*}}, __ttlang_constant_table_{{[A-F0-9]+}})
// Repeated values reuse the same static table.
// CPP: experimental::constant_table_lookup<4>({{.*}}, [[NARROW_CPP]])
func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %index = arith.constant 1 : index
  %value = ttkernel.experimental.constant_table_lookup %index, [3, 5, 8] : index
  %wide = ttkernel.experimental.constant_table_lookup %index,
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] : index
  %duplicate = ttkernel.experimental.constant_table_lookup %index,
      [3, 5, 8] : index
  return
}
