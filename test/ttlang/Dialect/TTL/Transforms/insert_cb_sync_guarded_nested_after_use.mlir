// A guarded producer block whose last storage use precedes a nested
// acquisition must be published before entering the nested region.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-insert-cb-sync))' | FileCheck %s
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-insert-cb-sync,ttl-insert-cb-sync))' | FileCheck %s

// CHECK-LABEL: func.func @guarded_reserve_before_nested_reserve
// CHECK: %[[CB:.+]] = ttl.bind_cb
// CHECK: scf.if
// CHECK:   ttl.cb_reserve %[[CB]]
// CHECK:   ttl.store
// CHECK-NEXT:   ttl.cb_push %[[CB]]
// CHECK:   ttl.cb_wait %[[CB]]
// CHECK:   ttl.add
// CHECK-NEXT:   ttl.cb_pop %[[CB]]
// CHECK:   scf.for
// CHECK:     ttl.cb_reserve %[[CB]]
// CHECK:     ttl.store
// CHECK-NEXT:     ttl.cb_push %[[CB]]
func.func @guarded_reserve_before_nested_reserve(
    %condition: i1, %source: tensor<1x1x!ttcore.tile<32x32, bf16>>)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %cb = ttl.bind_cb {cb_index = 0, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %two = arith.constant 2 : index
  %guarded = scf.if %condition -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
    %first = ttl.cb_reserve %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %source, %first
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
    %waited = ttl.cb_wait %cb
        : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %view = ttl.attach_cb %waited, %cb
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %unused = ttl.add %view, %source
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.for %iv = %zero to %two step %one {
      %second = ttl.cb_reserve %cb
          : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.store %source, %second
          : tensor<1x1x!ttcore.tile<32x32, bf16>>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>
    }
    scf.yield %first : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } else {
    %inactive = "builtin.unrealized_conversion_cast"()
        {ttl.inactive_guarded_dfb} : ()
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %inactive : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  func.return
}
