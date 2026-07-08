// RUN: ttlang-opt %s --split-input-file | FileCheck %s

// Summary: Verifies `ttl.accumulation_scope` parses and prints accumulation
// policies, including multi-output yielded state.

// CHECK-LABEL: func.func @accumulation_scope_overwrite
func.func @accumulation_scope_overwrite() {
  // CHECK: ttl.accumulation_scope outs(%{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  // CHECK-NEXT: ^bb0(%[[ACC:.*]]: tensor<1x1x!ttcore.tile<32x32, bf16>>):
  // CHECK-NEXT:   ttl.yield %[[ACC]] : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // CHECK-NEXT: } initial_modes([overwrite])
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    ttl.yield %acc : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } initial_modes([overwrite])
  return
}

// -----

// CHECK-LABEL: func.func @accumulation_scope_init
func.func @accumulation_scope_init() {
  // CHECK: ttl.accumulation_scope outs(%{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>) inits(%{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  // CHECK-NEXT: ^bb0(%[[ACC:.*]]: tensor<1x1x!ttcore.tile<32x32, bf16>>):
  // CHECK-NEXT:   ttl.yield %[[ACC]] : tensor<1x1x!ttcore.tile<32x32, bf16>>
  // CHECK-NEXT: } initial_modes([init])
  %out = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.accumulation_scope outs(%out : tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    ttl.yield %acc : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } initial_modes([init])
  return
}

// -----

// CHECK-LABEL: func.func @accumulation_scope_multi_output
func.func @accumulation_scope_multi_output() {
  // CHECK: ttl.accumulation_scope outs(%{{.*}}, %{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>) inits(%{{.*}} : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  // CHECK-NEXT: ^bb0(%[[ACC0:.*]]: tensor<1x1x!ttcore.tile<32x32, bf16>>, %[[ACC1:.*]]: tensor<1x1x!ttcore.tile<32x32, bf16>>):
  // CHECK-NEXT:   ttl.yield %[[ACC0]], %[[ACC1]] : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  // CHECK-NEXT: } initial_modes([init, overwrite])
  %out0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %out1 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  %init0 = tensor.empty() : tensor<1x1x!ttcore.tile<32x32, bf16>>
  ttl.accumulation_scope outs(%out0, %out1 : tensor<1x1x!ttcore.tile<32x32, bf16>>,
                                            tensor<1x1x!ttcore.tile<32x32, bf16>>)
      inits(%init0 : tensor<1x1x!ttcore.tile<32x32, bf16>>) {
  ^bb0(%acc0: tensor<1x1x!ttcore.tile<32x32, bf16>>,
       %acc1: tensor<1x1x!ttcore.tile<32x32, bf16>>):
    ttl.yield %acc0, %acc1 : tensor<1x1x!ttcore.tile<32x32, bf16>>, tensor<1x1x!ttcore.tile<32x32, bf16>>
  } initial_modes([init, overwrite])
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
  // CHECK-NEXT: } initial_modes([init, init])
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
  } initial_modes([init, init])
  return
}
