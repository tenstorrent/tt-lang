// RUN: ttlang-opt --verify-diagnostics --split-input-file %s

// Tests malformed logical-kernel metadata independently.

// A canonical selector cannot claim an operation.
// expected-error @below {{canonical logical kernel cannot have an operation or role}}
func.func @canonical_with_operation() attributes {ttl.logical_kernel = #ttl.logical_kernel<kind = compute, operation = "operation">} {
  return
}

// -----

// A named handle must belong to an operation or a compiler-owned role.
// expected-error @below {{named logical kernel requires exactly one of an operation or compiler-owned role}}
func.func @named_without_owner() attributes {ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader">} {
  return
}

// -----

// An operation handle cannot also claim a compiler-owned role.
// expected-error @below {{named logical kernel requires exactly one of an operation or compiler-owned role}}
func.func @named_with_two_owners() attributes {ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation", role = "source">} {
  return
}

// -----

// Present identities cannot use an empty string to denote absence.
// expected-error @below {{logical kernel identity must be nonempty}}
func.func @empty_identity() attributes {ttl.logical_kernel = #ttl.logical_kernel<kind = compute, identity = "">} {
  return
}

// -----

// Present operation identities must be nonempty.
// expected-error @below {{logical kernel operation must be nonempty}}
func.func @empty_operation() attributes {ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "">} {
  return
}

// -----

// Present compiler-owned roles must be nonempty.
// expected-error @below {{logical kernel role must be nonempty}}
func.func @empty_role() attributes {ttl.logical_kernel = #ttl.logical_kernel<kind = data_movement, identity = "<source>", role = "">} {
  return
}
