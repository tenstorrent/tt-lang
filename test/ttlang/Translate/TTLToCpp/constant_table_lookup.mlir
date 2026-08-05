// RUN: ttlang-opt --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.emitc.mlir --check-prefix=EMITC
// RUN: ttlang-translate --ttkernel-to-cpp %t.emitc.mlir 2>&1 | FileCheck %s --check-prefix=CPP

// Verify that immutable TTKernel tables become bit-packed compile-time C++
// storage rather than arrays allocated in kernel_main.

// EMITC-LABEL: func.func @kernel_main
// EMITC: emitc.call_opaque "experimental::constant_table_lookup"

// CPP: static constexpr std::uint64_t packed_table[] = {PackedWords...};
// CPP-LABEL: void kernel_main()
// CPP-NOT: size_t v{{[0-9]+}}[3];
// Values [3, 5, 8] require four bits each and pack into 0x853.
// CPP: experimental::constant_table_lookup<4, 0x853ULL>
// Seventeen five-bit values cross a 64-bit boundary.
// CPP: experimental::constant_table_lookup<5, 0xC5A928398A418820ULL, 0x107B9AULL>
func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %index = arith.constant 1 : index
  %value = ttkernel.experimental.constant_table_lookup %index, [3, 5, 8] : index
  %wide = ttkernel.experimental.constant_table_lookup %index,
      [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] : index
  return
}
