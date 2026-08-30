// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

module {
  // expected-error @below {{'ttkernel.dfb_resource_use' op expects ancestor op 'func.func'}}
  ttkernel.dfb_resource_use {indices = array<i32: 0>}
}

// -----

module {
  func.func @empty() {
    // expected-error @below {{requires at least one DFB index}}
    ttkernel.dfb_resource_use {indices = array<i32>}
    return
  }
}

// -----

module {
  func.func @negative() {
    // expected-error @below {{DFB index must be nonnegative, got -1}}
    ttkernel.dfb_resource_use {indices = array<i32: -1>}
    return
  }
}

// -----

module {
  func.func @duplicate() {
    // expected-error @below {{DFB indices must be strictly increasing without duplicates}}
    ttkernel.dfb_resource_use {indices = array<i32: 1, 1>}
    return
  }
}

// -----

module {
  func.func @descending() {
    // expected-error @below {{DFB indices must be strictly increasing without duplicates}}
    ttkernel.dfb_resource_use {indices = array<i32: 2, 1>}
    return
  }
}
