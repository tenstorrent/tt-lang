// RUN: ttlang-opt %s --canonicalize --split-input-file | FileCheck %s
// Purpose: positive coverage for ttl.compute with tensor-only operands and CB
// associations via ttl.attach_cb, including CB reuse.

// Simple compute with distinct CBs.
// CHECK-LABEL: func.func @compute_with_cbs
// CHECK-SAME: (%[[A:.*arg0]]: tensor<2x2x!ttcore.tile<32x32, f32>>,
// CHECK-SAME:  %[[B:.*arg1]]: tensor<2x2x!ttcore.tile<32x32, f32>>,
// CHECK-SAME: %[[CBA:.*arg2]]: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
// CHECK-SAME: %[[CBB:.*arg3]]: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
// CHECK-SAME: %[[CBOUT:.*arg4]]: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
func.func @compute_with_cbs(%a: tensor<2x2x!ttcore.tile<32x32, f32>>, %b: tensor<2x2x!ttcore.tile<32x32, f32>>,
                            %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
                            %cbb: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
                            %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK:      %[[INIT:.*]] = tensor.empty
  // CHECK-NEXT: %[[A_CB:.*]] = ttl.attach_cb %[[A]], %[[CBA]]
  // CHECK-NEXT: %[[B_CB:.*]] = ttl.attach_cb %[[B]], %[[CBB]]
  // CHECK-NEXT: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]], %[[CBOUT]]
  // CHECK-NEXT: ttl.cb_reserve
  // CHECK-NEXT: %[[RESULT:.*]] = ttl.compute ins(%[[A_CB]], %[[B_CB]] : {{.*}}) outs(%[[INIT_CB]] : {{.*}})
  // CHECK-NEXT: ^bb0(%[[AT:.*]]: !ttcore.tile<32x32, f32>, %[[BT:.*]]: !ttcore.tile<32x32, f32>, %[[CT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK:        ttl.iter_index
  // CHECK:        ttl.iter_index
  // CHECK:        %[[SUM:.*]] = ttl.tile_add %[[AT]], %[[BT]]
  // CHECK:        ttl.tile_store
  // CHECK-NEXT:   ttl.yield
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
  %out_view = ttl.cb_reserve %cbout : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
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
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %sum = ttl.tile_add %at, %bt into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// CB reuse when the same tensor accessor is used twice.
// CHECK-LABEL: func.func @compute_with_cbs_reuse
// CHECK-SAME: (%[[A:.*arg0]]: tensor<2x2x!ttcore.tile<32x32, f32>>,
// CHECK-SAME:  %[[CBA:.*arg1]]: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
// CHECK-SAME:  %[[CBOUT:.*arg2]]: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>) -> tensor<2x2x!ttcore.tile<32x32, f32>>
func.func @compute_with_cbs_reuse(%a: tensor<2x2x!ttcore.tile<32x32, f32>>,
                                  %cba: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>,
                                  %cbout: !ttl.cb<[1, 1], !ttcore.tile<32x32, f32>, 2>)
    -> tensor<2x2x!ttcore.tile<32x32, f32>> {
  // CHECK:      %[[INIT:.*]] = tensor.empty
  // CHECK-NEXT: %[[A_CB0:.*]] = ttl.attach_cb %[[A]], %[[CBA]]
  // CHECK-NEXT: %[[A_CB1:.*]] = ttl.attach_cb %[[A]], %[[CBA]]
  // CHECK-NEXT: %[[INIT_CB:.*]] = ttl.attach_cb %[[INIT]], %[[CBOUT]]
  // CHECK-NEXT: ttl.cb_reserve
  // CHECK-NEXT: %[[RESULT:.*]] = ttl.compute ins(%[[A_CB0]], %[[A_CB1]] : {{.*}}) outs(%[[INIT_CB]] : {{.*}})
  // CHECK-NEXT: ^bb0(%[[AT0:.*]]: !ttcore.tile<32x32, f32>, %[[AT1:.*]]: !ttcore.tile<32x32, f32>, %[[CT:.*]]: !ttcore.tile<32x32, f32>):
  // CHECK:        ttl.iter_index
  // CHECK:        ttl.iter_index
  // CHECK:        %[[SUM:.*]] = ttl.tile_add %[[AT0]], %[[AT1]]
  // CHECK:        ttl.tile_store
  // CHECK-NEXT:   ttl.yield
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
  %out_view = ttl.cb_reserve %cbout : <[1, 1], !ttcore.tile<32x32, f32>, 2> -> tensor<1x1x!ttcore.tile<32x32, f32>>
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
    %i = ttl.iter_index 0 : index
    %j = ttl.iter_index 1 : index
    %c0 = arith.constant 0 : index
    %sum = ttl.tile_add %at0, %at1 into dst[%c0] : !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32> -> !ttcore.tile<32x32, f32>
    ttl.tile_store %sum, %out_view[%i, %j] from dst[%c0] : !ttcore.tile<32x32, f32>, tensor<1x1x!ttcore.tile<32x32, f32>>
    ttl.yield
  } -> tensor<2x2x!ttcore.tile<32x32, f32>>
  func.return %0 : tensor<2x2x!ttcore.tile<32x32, f32>>
}

// -----

// A nested matmul enables the additional matmul tile dimensions for the
// complete compute region.
// CHECK-LABEL: func.func @compute_with_nested_matmul_subtile
// CHECK:       ttl.compute
// CHECK:       scf.if
// CHECK:       ttl.tile_matmul_block
func.func @compute_with_nested_matmul_subtile(
    %condition: i1,
    %lhs: tensor<1x1x!ttcore.tile<1x32, bf16>>,
    %rhs: tensor<1x1x!ttcore.tile<32x32, bf16>>,
    %lhs_dfb: !ttl.cb<[1, 1], !ttcore.tile<1x32, bf16>, 2>,
    %rhs_dfb: !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>,
    %output_dfb: !ttl.cb<[1, 1], !ttcore.tile<1x32, bf16>, 2>)
    -> tensor<1x1x!ttcore.tile<1x32, bf16>> {
  %init = tensor.empty() : tensor<1x1x!ttcore.tile<1x32, bf16>>
  %attached_lhs = ttl.attach_cb %lhs, %lhs_dfb
      : (tensor<1x1x!ttcore.tile<1x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<1x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<1x32, bf16>>
  %attached_rhs = ttl.attach_cb %rhs, %rhs_dfb
      : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %attached_init = ttl.attach_cb %init, %output_dfb
      : (tensor<1x1x!ttcore.tile<1x32, bf16>>,
         !ttl.cb<[1, 1], !ttcore.tile<1x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<1x32, bf16>>
  %output = ttl.cb_reserve %output_dfb
      : <[1, 1], !ttcore.tile<1x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<1x32, bf16>>
  %result = ttl.compute
      ins(%attached_lhs, %attached_rhs
          : tensor<1x1x!ttcore.tile<1x32, bf16>>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>)
      outs(%attached_init : tensor<1x1x!ttcore.tile<1x32, bf16>>)
      {indexing_maps = [affine_map<(row, column) -> (row, column)>,
                        affine_map<(row, column) -> (row, column)>,
                        affine_map<(row, column) -> (row, column)>],
       iterator_types = ["parallel", "parallel"]} {
  ^bb0(%lhs_tile: !ttcore.tile<1x32, bf16>,
       %rhs_tile: !ttcore.tile<32x32, bf16>,
       %output_tile: !ttcore.tile<1x32, bf16>):
    %c0 = arith.constant 0 : index
    %product = scf.if %condition -> (!ttcore.tile<1x32, bf16>) {
      %product = ttl.tile_matmul_block %lhs_tile, %rhs_tile into dst[%c0]
          : !ttcore.tile<1x32, bf16>, !ttcore.tile<32x32, bf16>
            -> !ttcore.tile<1x32, bf16>
      scf.yield %product : !ttcore.tile<1x32, bf16>
    } else {
      scf.yield %output_tile : !ttcore.tile<1x32, bf16>
    }
    ttl.tile_store %product, %output[%c0, %c0] from dst[%c0]
        : !ttcore.tile<1x32, bf16>,
          tensor<1x1x!ttcore.tile<1x32, bf16>>
    ttl.yield
  } -> tensor<1x1x!ttcore.tile<1x32, bf16>>
  func.return %result : tensor<1x1x!ttcore.tile<1x32, bf16>>
}
