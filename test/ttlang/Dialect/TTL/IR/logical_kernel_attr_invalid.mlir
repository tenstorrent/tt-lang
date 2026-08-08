// RUN: ttlang-opt %s --verify-diagnostics --split-input-file

module {
  func.func @empty_identity() attributes {
    // expected-error @below {{logical kernel identity must be nonempty}}
    ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "", operation = "models.router:42">
  } {
    return
  }
}

// -----

module {
  func.func @operation_without_identity() attributes {
    // expected-error @below {{canonical logical kernel cannot have an operation or role}}
    ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, operation = "models.router:42">
  } {
    return
  }
}

// -----

module {
  func.func @identity_without_owner() attributes {
    // expected-error @below {{named logical kernel requires exactly one of an operation or compiler-owned role}}
    ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "sender">
  } {
    return
  }
}

// -----

module {
  func.func @operation_and_role() attributes {
    // expected-error @below {{named logical kernel requires exactly one of an operation or compiler-owned role}}
    ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "sender", operation = "models.router:42", role = "pipe_source">
  } {
    return
  }
}
