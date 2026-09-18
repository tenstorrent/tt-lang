// Verifies row-normalization recognition with commuted operands and
// conservative rejection when the target lacks the schedule or DST capacity.
// RUN: ttlang-opt %s --split-input-file -pass-pipeline='builtin.module(func.func(ttl-print-compute-op-creation-plans))' -o /dev/null 2>&1 | FileCheck %s

// Commuting the scalar add and final product does not change recognition.
// CHECK-LABEL: ComputeOp creation plan @row_normalization_commuted
// CHECK:       ttl.mul kind=fused recipe=row_normalization legal=true inputs=1 outputs=1 transactions=1
// CHECK:       order=[C0]
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @row_normalization_commuted()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %input_wait = ttl.cb_wait %input_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %input_wait, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %square = ttl.mul %input, %input
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %square, %scaler 0 : i32 [0, 1]
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %mean_square = ttl.mul_unary_const %reduced, 9.765625e-04
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %epsilon = ttl.fill 1.000000e-05
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %biased = ttl.add %epsilon, %mean_square
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %inverse_rms = ttl.rsqrt %biased
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scalar = ttl.block.broadcast %inverse_rms dims = [0, 1], shape = [1, 1]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %result = ttl.mul %scalar, %input
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// A target without the schedule capability retains ordinary materialization.
// CHECK-LABEL: ComputeOp creation plan @row_normalization_unsupported_schedule
// CHECK-NOT:   recipe=row_normalization
module attributes {ttl.target_arch = #ttcore.arch<wormhole_b0>} {
  func.func @row_normalization_unsupported_schedule()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %input_wait = ttl.cb_wait %input_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %input_wait, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %square = ttl.mul %input, %input
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %square, %scaler 0 : i32 [0, 1]
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %mean_square = ttl.mul_unary_const %reduced, 9.765625e-04
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %epsilon = ttl.fill 1.000000e-05
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %biased = ttl.add %mean_square, %epsilon
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %inverse_rms = ttl.rsqrt %biased
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scalar = ttl.block.broadcast %inverse_rms dims = [0, 1], shape = [1, 1]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %result = ttl.mul %input, %scalar
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// A nine-tile row exceeds the schedule and effective DST capacity limits.
// CHECK-LABEL: ComputeOp creation plan @row_normalization_exceeds_capacity
// CHECK-NOT:   recipe=row_normalization
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @row_normalization_exceeds_capacity()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 9], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 9], !ttcore.tile<32x32, bf16>, 2>
    %input_wait = ttl.cb_wait %input_dfb
        : <[1, 9], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x9x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %input_wait, %input_dfb
        : (tensor<1x9x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 9], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x9x!ttcore.tile<32x32, bf16>>
    %square = ttl.mul %input, %input
        : tensor<1x9x!ttcore.tile<32x32, bf16>>,
          tensor<1x9x!ttcore.tile<32x32, bf16>>
          -> tensor<1x9x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %square, %scaler 0 : i32 [0, 1]
        : (tensor<1x9x!ttcore.tile<32x32, bf16>>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %mean_square = ttl.mul_unary_const %reduced, 1.085069e-04
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %epsilon = ttl.fill 1.000000e-05
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %biased = ttl.add %mean_square, %epsilon
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %inverse_rms = ttl.rsqrt %biased
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scalar = ttl.block.broadcast %inverse_rms dims = [0, 1], shape = [1, 9]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x9x!ttcore.tile<32x32, bf16>>
    %result = ttl.mul %input, %scalar
        : tensor<1x9x!ttcore.tile<32x32, bf16>>,
          tensor<1x9x!ttcore.tile<32x32, bf16>>
          -> tensor<1x9x!ttcore.tile<32x32, bf16>>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 9], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x9x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %output
        : tensor<1x9x!ttcore.tile<32x32, bf16>>,
          tensor<1x9x!ttcore.tile<32x32, bf16>>
    return
  }
}

// -----

// The specialized schedule publishes exactly one output transaction.
// CHECK-LABEL: ComputeOp creation plan @row_normalization_multiple_outputs
// CHECK:       ttl.mul kind=fused recipe=row_normalization legal=false inputs=1 outputs=2 transactions=2
// CHECK:       rejected=row-normalization block requires exactly one output store transaction
module attributes {ttl.target_arch = #ttcore.arch<blackhole>} {
  func.func @row_normalization_multiple_outputs()
      attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    %input_dfb = ttl.bind_cb {cb_index = 0, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %output_dfb = ttl.bind_cb {cb_index = 1, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %other_output_dfb = ttl.bind_cb {cb_index = 2, block_count = 2}
        : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
    %input_wait = ttl.cb_wait %input_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %input = ttl.attach_cb %input_wait, %input_dfb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %square = ttl.mul %input, %input
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scaler = ttl.fill 1.000000e+00
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %reduced = ttl.reduce %square, %scaler 0 : i32 [0, 1]
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           tensor<1x1x!ttcore.tile<32x32, bf16>>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %mean_square = ttl.mul_unary_const %reduced, 9.765625e-04
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %epsilon = ttl.fill 1.000000e-05
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
    %biased = ttl.add %epsilon, %mean_square
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %inverse_rms = ttl.rsqrt %biased
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %scalar = ttl.block.broadcast %inverse_rms dims = [0, 1], shape = [1, 1]
        : tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %result = ttl.mul %scalar, %input
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %output = ttl.cb_reserve %output_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
    %other_output = ttl.cb_reserve %other_output_dfb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %result, %other_output
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
    return
  }
}
