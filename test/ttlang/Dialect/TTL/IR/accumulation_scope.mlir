// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// Summary: Verifies `ttl.accumulation_scope` parses and prints accumulation
// policies, including multi-output stateful accumulation metadata.

// CHECK-LABEL: func.func @accumulation_scope_overwrite
func.func @accumulation_scope_overwrite() {
  // CHECK: ttl.accumulation_scope outs(%{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } combiners([add]) initial_modes([overwrite])
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ttl.yield
  } combiners([add]) initial_modes([overwrite])
  return
}

// -----

// CHECK-LABEL: func.func @accumulation_scope_explicit_init
func.func @accumulation_scope_explicit_init() {
  // CHECK: ttl.accumulation_scope outs(%{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>) inits(%{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } combiners([add]) initial_modes([explicit])
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ttl.yield
  } combiners([add]) initial_modes([explicit])
  return
}

// -----

// CHECK-LABEL: func.func @accumulation_scope_multi_output
func.func @accumulation_scope_multi_output() {
  // CHECK: ttl.accumulation_scope outs(%{{.*}}, %{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) inits(%{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  // CHECK-NEXT:   ttl.yield
  // CHECK-NEXT: } combiners([add, add]) initial_modes([explicit, overwrite])
  %out0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.accumulation_scope outs(%out0, %out1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                            tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init0 : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
    ttl.yield
  } combiners([add, add]) initial_modes([explicit, overwrite])
  return
}

// -----

// CHECK-LABEL: func.func @accumulation_scope_stateful_multi_output
func.func @accumulation_scope_stateful_multi_output() {
  // CHECK: ttl.accumulation_scope outs(%{{.*}}, %{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) inits(%{{.*}}, %{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  // CHECK-NEXT: ^bb0(%[[ARG0:.*]]: tensor<1x1x!ttcore.tile<32x32, bf16>>, %[[ARG1:.*]]: tensor<1x1x!ttcore.tile<32x32, bf16>>):
  // CHECK-NEXT:   %[[NEXT0:.*]] = ttl.add %[[ARG0]], %[[ARG1]]
  // CHECK-NEXT:   %[[NEXT1:.*]] = ttl.add %[[ARG1]], %[[NEXT0]]
  // CHECK-NEXT:   ttl.yield %[[NEXT0]], %[[NEXT1]] : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  // CHECK-NEXT: } combiners([yielded, yielded]) initial_modes([explicit, explicit])
  %out0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.accumulation_scope outs(%out0, %out1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                            tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init0, %init1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                              tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
       %acc1: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    %next0 = ttl.add %acc0, %acc1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %next1 = ttl.add %acc1, %next0 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>> -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.yield %next0, %next1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  } combiners([yielded, yielded]) initial_modes([explicit, explicit])
  return
}
