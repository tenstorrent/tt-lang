// RUN: ttlang-opt %s --ttl-lower-dprint-to-emitc | FileCheck %s

// CB-mode dprint keeps a typed get_compile_time_arg_val use as an
// emitc.verbatim operand instead of baking the index into C++ text.

func.func @cb_print() {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  "ttl.dprint"(%cb) {fmt = "input", mode = "cb"}
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
  return
}

// CHECK-LABEL: func.func @cb_print
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> i32
// CHECK: emitc.verbatim "ttmlir::dprint(ttmlir::CBPrinter({}));" args %[[CB]] : i32
