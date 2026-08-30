// RUN: ttlang-opt %s --ttkernel-annotate-dfb-use --verify-diagnostics --split-input-file

module {
  func.func @out_of_range() attributes {
      ttl.base_cta_index = 2 : i32,
      ttkernel.thread = #ttkernel.thread<noc>} {
    // expected-error @below {{DFB index 2 is outside the enclosing function's DFB range [0, 2)}}
    ttkernel.dfb_resource_use {indices = array<i32: 2>}
    return
  }
}
