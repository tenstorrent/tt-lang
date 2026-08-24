// RUN: ttlang-opt %s --ttl-lower-dprint-to-emitc --split-input-file | FileCheck %s

// CB-mode dprint keeps a typed get_compile_time_arg_val use as an
// emitc.verbatim operand instead of baking the index into C++ text.

// CHECK-LABEL: func.func @cb_print
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> i32
// CHECK: emitc.verbatim "ttmlir::dprint(ttmlir::CBPrinter({}));" args %[[CB]] : i32
func.func @cb_print() {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  "ttl.dprint"(%cb) {fmt = "input", mode = "cb"}
      : (!ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> ()
  return
}

// -----

// Tile prints pass the CB id as an emitc operand. DEVICE_PRINT `{}` must be
// written `{{}` so emitc does not consume it, and C++ braces in SliceRange /
// loops must not use `}}` (that emits two closing braces).

// CHECK-LABEL: func.func @tile_print
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> i32
// CHECK: emitc.verbatim
// CHECK-SAME: TSLICE({}, 0, SliceRange{{[{]}}{{[{]}}.h0=
// CHECK-SAME: .ws=1}, true, false)
// CHECK-SAME: args %[[CB]] : i32
func.func @tile_print() {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %ready = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %tile = ttl.attach_cb %ready, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  "ttl.dprint"(%tile) {fmt = "tile", mode = "tile"}
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>) -> ()
  return
}

// -----

// CB-backed tensor/page prints use the same emitc operand rules as tile mode.

// CHECK-LABEL: func.func @tensor_print
// CHECK: %[[CB:.*]] = ttkernel.get_compile_time_arg_val(0) : () -> i32
// CHECK: emitc.verbatim
// CHECK-SAME: get_read_ptr({})
// CHECK-SAME: args %[[CB]] : i32
func.func @tensor_print() {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %ready = ttl.cb_wait %cb : <[1, 1], !ttcore.tile<32x32, bf16>, 2> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %tile = ttl.attach_cb %ready, %cb : (tensor<1x1x!ttcore.tile<32x32, bf16>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>) -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  "ttl.dprint"(%tile) {fmt = "pages", mode = "tensor", num_pages = 1}
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>) -> ()
  return
}
