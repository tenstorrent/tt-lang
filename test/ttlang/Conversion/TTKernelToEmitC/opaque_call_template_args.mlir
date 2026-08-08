// Verify static ttkernel.opaque_call arguments lower to typed C++ literals.
// RUN: ttlang-opt --convert-ttkernel-to-emitc --split-input-file -o %t.emitc.mlir %s
// RUN: FileCheck %s --input-file=%t.emitc.mlir --check-prefix=EMITC
// RUN: ttlang-translate --allow-unregistered-dialect --ttkernel-to-cpp -o %t.cpp %t.emitc.mlir
// RUN: FileCheck %s --input-file=%t.cpp --check-prefix=CPP

// Signed, boolean, and unsigned arguments retain their C++ meanings.
// EMITC-LABEL: func.func @typed_literals_to_emitc
// EMITC: emitc.call_opaque "typed"
// EMITC-SAME: template_args = [#emitc.opaque<"-5">, #emitc.opaque<"true">, #emitc.opaque<"4294967295U">]
func.func @typed_literals_to_emitc() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  ttkernel.opaque_call "typed" template_args [-5 : si32, true, 4294967295 : ui32] () {header = "typed.hpp"} : () -> ()
  return
}

// -----

// A descriptor becomes a C++ type argument in its original list position.
// EMITC-LABEL: func.func @dfb_descriptor_template_to_emitc
// EMITC-NOT: emitc.literal
// EMITC: emitc.call_opaque "describe"
// EMITC-SAME: template_args = [#emitc.opaque<"11">, #emitc.opaque<"ttlang::DFBDescriptor<3, 2, 4, 4096>">]
// EMITC-SAME: ttlang.opaque_header = "describe.hpp"
// EMITC-SAME: ttlang.requires_dfb_descriptor

// The emitted definition precedes the user header that names it.
// CPP-LABEL: #include <cstdint>
// CPP: namespace ttlang {
// CPP: struct DFBDescriptor {
// CPP: } // namespace ttlang
// CPP: #include "describe.hpp"
// CPP: describe<11, ttlang::DFBDescriptor<3, 2, 4, 4096>>();
func.func @dfb_descriptor_template_to_emitc() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  ttkernel.opaque_call "describe" template_args [11 : si32, #ttkernel.dfb_descriptor<3, 2, 4, 4096>] () {header = "describe.hpp"} : () -> ()
  return
}

// -----

// An unsigned boundary annotation creates an explicit uint32_t call operand.
// EMITC-LABEL: func.func @unsigned_func_arg_to_emitc
// EMITC: %[[SIGNED:.*]] = "emitc.constant"() <{value = -1 : i32}> : () -> i32
// EMITC-NEXT: %[[UNSIGNED:.*]] = emitc.cast %[[SIGNED]] : i32 to ui32
// EMITC-NEXT: emitc.call_opaque "use_address"(%[[UNSIGNED]])
// CPP-LABEL: #include "address.hpp"
// CPP: void kernel_main() {
// CPP: int32_t [[SIGNED:v[0-9]+]] = -1;
// CPP-NEXT: uint32_t [[UNSIGNED:v[0-9]+]] = (uint32_t) [[SIGNED]];
// CPP-NEXT: use_address([[UNSIGNED]]);
func.func @unsigned_func_arg_to_emitc() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %address = arith.constant -1 : i32
  ttkernel.opaque_call "use_address" (%address) {header = "address.hpp", unsigned_arg_indices = array<i32: 0>} : (i32) -> ()
  return
}
