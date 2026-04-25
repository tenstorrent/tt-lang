// Negative tests for ttl-subblock-compute-for-dst with --strict-f32-acc.
// The check fires when a user-written accumulation loop (+=) with non-f32
// output requires subblocking, because bf16 L1 intermediates truncate f32
// DST partial sums per K step.

// RUN: ttlang-opt %s \
// RUN:   --pass-pipeline='builtin.module(func.func( \
// RUN:     ttl-annotate-l1-acc-loops, convert-ttl-to-compute, \
// RUN:     ttl-assign-dst{enable-fpu-binary-ops=0}, \
// RUN:     ttl-subblock-compute-for-dst{strict-f32-acc=true}))' \
// RUN:   --verify-diagnostics --split-input-file

// bf16 output 3x3 = 9 tiles exceeds f32 DST capacity (4): should error.

func.func @strict_f32_subblock_bf16_error(
    %arg0: tensor<3x2x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<2x3x!ttcore.tile<32x32, bf16>>) -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, fp32_dest_acc_en} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[3, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[3, 3], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<3x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[3, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<3x2x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<2x3x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 3], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x3x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[3, 3], !ttcore.tile<32x32, bf16>, 2> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
  scf.for %k = %c0 to %c2 step %c1 {
    // expected-error @below {{subblocking accumulation loop reduces precision}}
    %mm = ttl.matmul %a, %b : tensor<3x2x!ttcore.tile<32x32, bf16>>, tensor<2x3x!ttcore.tile<32x32, bf16>> -> tensor<3x3x!ttcore.tile<32x32, bf16>>
    ttl.store %mm, %reserve {accumulate} : tensor<3x3x!ttcore.tile<32x32, bf16>>, tensor<3x3x!ttcore.tile<32x32, bf16>>
  }
  ttl.cb_push %cb2 : <[3, 3], !ttcore.tile<32x32, bf16>, 2>
  func.return %reserve : tensor<3x3x!ttcore.tile<32x32, bf16>>
}

// -----

// bf16 output 2x2 = 4 tiles fits in f32 DST (4): no subblocking, no error.

// expected-no-diagnostics
func.func @strict_f32_fits_in_dst_ok(
    %arg0: tensor<2x2x!ttcore.tile<32x32, bf16>>,
    %arg1: tensor<2x2x!ttcore.tile<32x32, bf16>>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>, fp32_dest_acc_en} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cb0 = ttl.bind_cb {cb_index = 0, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb1 = ttl.bind_cb {cb_index = 1, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %cb2 = ttl.bind_cb {cb_index = 2, block_count = 2} : !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>
  %a = ttl.attach_cb %arg0, %cb0 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %b = ttl.attach_cb %arg1, %cb1 : (tensor<2x2x!ttcore.tile<32x32, bf16>>, !ttl.cb<[2, 2], !ttcore.tile<32x32, bf16>, 2>) -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  %reserve = ttl.cb_reserve %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
  scf.for %k = %c0 to %c2 step %c1 {
    %mm = ttl.matmul %a, %b : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>> -> tensor<2x2x!ttcore.tile<32x32, bf16>>
    ttl.store %mm, %reserve {accumulate} : tensor<2x2x!ttcore.tile<32x32, bf16>>, tensor<2x2x!ttcore.tile<32x32, bf16>>
  }
  ttl.cb_push %cb2 : <[2, 2], !ttcore.tile<32x32, bf16>, 2>
  func.return %reserve : tensor<2x2x!ttcore.tile<32x32, bf16>>
}
