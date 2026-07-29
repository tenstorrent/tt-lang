// Verify ttkernel.opaque_call template-arg lowering to EmitC literals.
// RUN: ttlang-opt --convert-ttkernel-to-emitc --split-input-file -o %t.emitc.mlir %s
// RUN: FileCheck %s --input-file=%t.emitc.mlir --check-prefix=EMITC

// EMITC-LABEL: func.func @semaphore_index_template_to_emitc
// EMITC: emitc.call_opaque "sem_tpl"
// EMITC-SAME: template_args = [#emitc.opaque<"5">]
func.func @semaphore_index_template_to_emitc() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %c5 = arith.constant 5 : i32
  ttkernel.opaque_call "sem_tpl" template_args(%c5) () {header = "sem_tpl.hpp"} : () -> ()
  return
}

// -----

// EMITC-LABEL: func.func @dfb_template_to_emitc
// EMITC: emitc.call_opaque "dfb_tpl"
// EMITC-SAME: template_args = [#emitc.opaque<"7">]
func.func @dfb_template_to_emitc() attributes {ttkernel.thread = #ttkernel.thread<noc>} {
  %cb = ttkernel.get_compile_time_arg_val(7) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, bf16>>
  %id = ttkernel.get_dfb_id %cb : <1, !ttcore.tile<32x32, bf16>>
  ttkernel.opaque_call "dfb_tpl" template_args(%id) () {header = "dfb_tpl.hpp"} : () -> ()
  return
}
