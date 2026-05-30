// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// Summary: Verifies `ttl.accumulation_scope` parses and prints its accumulation
// policy for overwrite and explicit initial-value modes.

// CHECK-LABEL: func.func @accumulation_scope_overwrite
func.func @accumulation_scope_overwrite() {
  // CHECK: ttl.accumulation_scope outs(%{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  // CHECK-NEXT: } {combiners = [0 : i32], initial_modes = [0 : i32]}
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ttl.yield
  } {combiners = [0 : i32], initial_modes = [0 : i32]}
  return
}

// -----

// CHECK-LABEL: func.func @accumulation_scope_explicit_init
func.func @accumulation_scope_explicit_init() {
  // CHECK: ttl.accumulation_scope outs(%{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>) inits(%{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  // CHECK-NEXT: } {combiners = [0 : i32], initial_modes = [2 : i32]}
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ttl.yield
  } {combiners = [0 : i32], initial_modes = [2 : i32]}
  return
}
