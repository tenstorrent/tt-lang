// Verifies rejection of malformed selected compute-pipeline schedule metadata.
// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

// The compiler-owned schedule attribute must use its registered enum type.
func.func @invalid_selected_schedule_type(
    %lhs: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
  // expected-error @below {{attribute 'ttl.selected_compute_pipeline_schedule' must be a #ttl.compute_pipeline_schedule attribute}}
  %result = ttl.mul %lhs, %rhs
      {ttl.selected_compute_pipeline_schedule = "invalid"}
      : tensor<1x1x!ttcore.tile<32x32, bf16>>,
        tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  return %result : tensor<1x1x!ttcore.tile<32x32, bf16>>
}
