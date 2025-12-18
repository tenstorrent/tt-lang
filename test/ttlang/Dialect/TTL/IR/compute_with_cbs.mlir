// RUN: ttlang-opt %s --canonicalize --split-input-file | FileCheck %s
// Purpose: positive coverage for ttl.compute with tensor-only operands and CB
// associations via ttl.attach_cb, including CB reuse.

// Simple compute with distinct CBs.
// CHECK-LABEL: func.func @compute_with_cbs
// CHECK-SAME: (%[[A:.*arg0]]: tensor<2x2x!ttcore.tile<32x32, f32>>,
// CHECK-SAME:  %[[B:.*arg1]]: tensor<2x2x!ttcore.tile<32x32, f32>>,
// CHECK-SAME: %[[CBA:.*arg2]]: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
// CHECK-SAME: %[[CBB:.*arg3]]: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
// CHECK-SAME: %[[CBOUT:.*arg4]]: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
func.func @compute_with_cbs(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                            %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
                            %cbb: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
                            %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK:      %[[INIT:.*]] = tensor.empty
  // CHECK-NEXT: %[[A_CB:.*]] = ttl.attach_cb %[[A]], %[[CBA]]
  // CHECK-NEXT: %[[B_CB:.*]] = ttl.attach_cb %[[B]], %[[CBB]]
  // CHECK-NEXT: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]], %[[CBOUT]]
  // CHECK-NEXT: %[[RESULT:.*]] = ttl.compute ins(%[[A_CB]], %[[B_CB]] : {{.*}}) outs(%[[INIT_CB]] : {{.*}})
  // CHECK-NEXT: ^bb0(%[[AT:.*]]: !ttcore.tile<32x32, f32>, %[[BT:.*]]: !ttcore.tile<32x32, f32>, %[[CT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[SUM:.*]] = ttl.tile_add %[[AT]], %[[BT]]
  // CHECK-NEXT:   ttl.yield %[[SUM]]
  // CHECK-NEXT: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %b_att = ttl.attach_cb %b, %cbb
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute
      ins(%a_att, %b_att : tensor<2x2x!ttcore.tile<32x32, f32>>,
                           tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%at: !ttcore.tile<32x32, f32>,
       %bt: !ttcore.tile<32x32, f32>,
       %ct: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %at, %bt : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// CB reuse when the same tensor accessor is used twice.
// CHECK-LABEL: func.func @compute_with_cbs_reuse
// CHECK-SAME: (%[[A:.*arg0]]: tensor<2x2x!ttcore.tile<32x32, f32>>,
// CHECK-SAME:  %[[CBA:.*arg1]]: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
// CHECK-SAME:  %[[CBOUT:.*arg2]]: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>> {
func.func @compute_with_cbs_reuse(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                  %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
                                  %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK:      %[[INIT:.*]] = tensor.empty
  // CHECK-NEXT: %[[A_CB0:.*]] = ttl.attach_cb %[[A]], %[[CBA]]
  // CHECK-NEXT: %[[A_CB1:.*]] = ttl.attach_cb %[[A]], %[[CBA]]
  // CHECK-NEXT: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]], %[[CBOUT]]
  // CHECK-NEXT: %[[RESULT:.*]] = ttl.compute ins(%[[A_CB0]], %[[A_CB1]] : {{.*}}) outs(%[[INIT_CB]] : {{.*}})
  // CHECK-NEXT: ^bb0(%[[AT0:.*]]: !ttcore.tile<32x32, f32>, %[[AT1:.*]]: !ttcore.tile<32x32, f32>, %[[CT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK-NEXT:   %[[SUM:.*]] = ttl.tile_add %[[AT0]], %[[AT1]]
  // CHECK-NEXT:   ttl.yield %[[SUM]]
  // CHECK-NEXT: } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  // CHECK-NEXT: return %[[RESULT]]
  %init = tensor.empty() : tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att0 = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %a_att1 = ttl.attach_cb %a, %cba
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %init_att = ttl.attach_cb %init, %cbout
      : (tensor<2x2x!ttcore.tile<32x32, f32>>, !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
        -> tensor<2x2x!ttcore.tile<32x32, f32>>
  %0 = ttl.compute
      ins(%a_att0, %a_att1 : tensor<2x2x!ttcore.tile<32x32, f32>>,
                             tensor<2x2x!ttcore.tile<32x32, f32>>)
      outs(%init_att : tensor<2x2x!ttcore.tile<32x32, f32>>)
      {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>,
                        affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%at0: !ttcore.tile<32x32, f32>,
       %at1: !ttcore.tile<32x32, f32>,
       %ct: !ttcore.tile<32x32, f32>):
    %sum = ttl.tile_add %at0, %at1 : !ttcore.tile<32x32, f32>
    ttl.yield %sum : !ttcore.tile<32x32, f32>
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}
