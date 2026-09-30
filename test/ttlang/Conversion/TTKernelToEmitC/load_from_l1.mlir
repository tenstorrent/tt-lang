// RUN: ttlang-opt --convert-ttkernel-to-emitc -o %t.emitc.mlir %s
// RUN: FileCheck %s --input-file=%t.emitc.mlir --check-prefix=EMITC
// RUN: ttlang-translate --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// Verifies that a load from L1 invalidates the local L1 cache and reads
// through a volatile pointer, so that NoC-delivered data is observed.

// EMITC-LABEL: func.func @kernel_main
// EMITC: emitc.call_opaque "invalidate_l1_cache"()
// EMITC: emitc.call_opaque "reinterpret_cast<volatile tt_l1_ptr uint32_t*>"
// EMITC-SAME: -> !emitc.ptr<!emitc.opaque<"volatile tt_l1_ptr uint32_t">>
// EMITC: emitc.call_opaque "invalidate_l1_cache"()
// EMITC: emitc.call_opaque "reinterpret_cast<volatile tt_l1_ptr uint16_t*>"
// EMITC-SAME: -> !emitc.ptr<!emitc.opaque<"volatile tt_l1_ptr uint16_t">>

// CPP-LABEL: void kernel_main()
// CPP: invalidate_l1_cache();
// CPP-NEXT: volatile tt_l1_ptr uint32_t* [[PTR32:v[0-9]+]] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>
// CPP-NEXT: volatile tt_l1_ptr uint32_t {{v[0-9]+}} = [[PTR32]][
// CPP: invalidate_l1_cache();
// CPP-NEXT: volatile tt_l1_ptr uint16_t* [[PTR16:v[0-9]+]] = reinterpret_cast<volatile tt_l1_ptr uint16_t*>
// CPP-NEXT: volatile tt_l1_ptr uint16_t {{v[0-9]+}} = [[PTR16]][
func.func @kernel_main() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %address = arith.constant 4096 : i32
  %offset = arith.constant 3 : i32
  %word_pointer = ttkernel.reinterpret_cast(%address)
      : (i32) -> !ttkernel.l1_addr_ptr
  %word = ttkernel.load_from_l1(%word_pointer, %offset)
      : (!ttkernel.l1_addr_ptr, i32) -> i32
  %half_pointer = ttkernel.reinterpret_cast(%address)
      : (i32) -> !ttkernel.l1_addr_ptr<16>
  %half = ttkernel.load_from_l1(%half_pointer, %offset)
      : (!ttkernel.l1_addr_ptr<16>, i32) -> i16
  return
}
