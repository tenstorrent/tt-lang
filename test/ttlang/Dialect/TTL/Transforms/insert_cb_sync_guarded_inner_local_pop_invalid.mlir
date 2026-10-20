// A guarded pop after an inner loop does not release the held block before
// that loop waits on the same DFB.
// RUN: ttlang-opt %s --pass-pipeline='builtin.module(func.func(ttl-insert-cb-sync))' --verify-diagnostics

func.func @guarded_wait_with_inner_local_pop(%condition: i1)
    attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
  %in = ttl.bind_cb {cb_index = 0, block_count = 4}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>
  %out = ttl.bind_cb {cb_index = 1, block_count = 2}
      : !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 2>
  %reserve = ttl.cb_reserve %out
      : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
      -> tensor<1x1x!ttcore.tile<32x32, bf16>>
  %held = scf.if %condition -> tensor<1x1x!ttcore.tile<32x32, bf16>> {
    // expected-error @below {{dataflow buffer block is still acquired when a nested region acquires the same buffer again}}
    %first = ttl.cb_wait %in
        : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    %first_view = ttl.attach_cb %first, %in
        : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
           !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>)
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    ttl.store %first_view, %reserve
        : tensor<1x1x!ttcore.tile<32x32, bf16>>,
          tensor<1x1x!ttcore.tile<32x32, bf16>>
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %two = arith.constant 2 : index
    scf.for %iv = %zero to %two step %one {
      // expected-note @below {{the buffer is acquired again here}}
      %second = ttl.cb_wait %in
          : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      %second_view = ttl.attach_cb %second, %in
          : (tensor<1x1x!ttcore.tile<32x32, bf16>>,
             !ttl.cb<[1, 1], !ttcore.tile<32x32, bf16>, 4>)
          -> tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.store %second_view, %reserve {accumulate}
          : tensor<1x1x!ttcore.tile<32x32, bf16>>,
            tensor<1x1x!ttcore.tile<32x32, bf16>>
      ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    }
    ttl.cb_pop %in : <[1, 1], !ttcore.tile<32x32, bf16>, 4>
    scf.yield %first_view : tensor<1x1x!ttcore.tile<32x32, bf16>>
  } else {
    %inactive = "builtin.unrealized_conversion_cast"()
        {ttl.inactive_guarded_dfb} : ()
        -> tensor<1x1x!ttcore.tile<32x32, bf16>>
    scf.yield %inactive : tensor<1x1x!ttcore.tile<32x32, bf16>>
  }
  ttl.cb_push %out : <[1, 1], !ttcore.tile<32x32, bf16>, 2>
  func.return
}
