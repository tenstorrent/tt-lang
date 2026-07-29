// RUN: ttlang-opt --convert-ttkernel-to-emitc %s -o %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.emitc.mlir --check-prefix=EMITC
// RUN: ttlang-translate --ttkernel-to-cpp %t.emitc.mlir 2>&1 | FileCheck %s --check-prefix=CPP

// Verify that immutable TTKernel tables become compile-time C++ storage rather
// than arrays allocated in kernel_main.

// EMITC-LABEL: func.func @kernel_main
// EMITC: emitc.call_opaque "experimental::constant_table_lookup"

// CPP: static constexpr std::size_t table[] = {Values...};
// CPP-LABEL: void kernel_main()
// CPP-NOT: size_t v{{[0-9]+}}[3];
// CPP: experimental::constant_table_lookup<3, 5, 8>
func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %index = arith.constant 1 : index
  %value = ttkernel.experimental.constant_table_lookup %index, [3, 5, 8] : index
  return
}
