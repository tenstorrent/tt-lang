// RUN: ttlang-opt --convert-ttl-to-ttkernel --verify-diagnostics --split-input-file %s

// -----

// ttl.store view must come from ttl.cb_reserve (not a function argument).
// The conversion fails because getCBFromView cannot trace from a block argument
// back to a CB - it expects a view from ttl.cb_reserve.
// expected-error @below {{failed to legalize operation 'ttl.store' that was explicitly marked illegal}}
module {
  func.func @store_view_not_from_reserve(%tile: !ttcore.tile<32x32, bf16>, %view: tensor<1x1x!ttcore.tile<32x32, bf16>>) attributes {ttl.kernel_thread = #ttkernel.thread<compute>} {
    ttl.store %tile, %view : !ttcore.tile<32x32, bf16>, tensor<1x1x!ttcore.tile<32x32, bf16>>
    func.return
  }
}
