// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// Verify init_sfpu parses and prints correctly with two CBs.
// CHECK-LABEL: func.func @init_sfpu_basic
// CHECK:         %[[CB0:.*]] = ttl.bind_cb
// CHECK:         %[[CB1:.*]] = ttl.bind_cb
// CHECK:         ttl.init_sfpu(%[[CB0]], %[[CB1]]) : <[1, 1], f32, 2>, <[1, 1], f32, 2>
func.func @init_sfpu_basic() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb0 = ttl.bind_cb {cb_index = 0, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, buffer_factor = 2} : !ttl.cb<[1, 1], f32, 2>
  ttl.init_sfpu(%cb0, %cb1) : !ttl.cb<[1, 1], f32, 2>, !ttl.cb<[1, 1], f32, 2>
  func.return
}

// -----

// Verify init_sfpu with same CB for both input and output (common case).
// CHECK-LABEL: func.func @init_sfpu_same_cb
// CHECK:         %[[CB:.*]] = ttl.bind_cb
// CHECK:         ttl.init_sfpu(%[[CB]], %[[CB]]) : <[2, 2], !ttcore.tile<32x32, bf16>, 1>, <[2, 2], !ttcore.tile<32x32, bf16>, 1>
func.func @init_sfpu_same_cb() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>
  ttl.init_sfpu(%cb, %cb) : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 1>
  func.return
}

// -----

// Verify init_sfpu with different CB element types (input bf16, output f32).
// CHECK-LABEL: func.func @init_sfpu_different_types
// CHECK:         %[[ICB:.*]] = ttl.bind_cb{{.*}}cb_index = 0
// CHECK:         %[[OCB:.*]] = ttl.bind_cb{{.*}}cb_index = 2
// CHECK:         ttl.init_sfpu(%[[ICB]], %[[OCB]]) : <[1, 1], !ttcore.tile<32x32, bf16>, 1>, <[1, 1], !ttcore.tile<32x32, f32>, 1>
func.func @init_sfpu_different_types() attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %icb = ttl.bind_cb {cb_index = 0, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>
  %ocb = ttl.bind_cb {cb_index = 2, buffer_factor = 1} : !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  ttl.init_sfpu(%icb, %ocb) : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 1>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 1>
  func.return
}
