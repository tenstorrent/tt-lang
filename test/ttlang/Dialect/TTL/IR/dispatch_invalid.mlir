// RUN: ttlang-opt %s --split-input-file --verify-diagnostics

// Verify the symbol and argument ABI of dispatch targets and invocations.

module {
  // expected-error @below {{declares 1 argument names for 2 inputs}}
  ttl.dispatch.target @bad_names : (i32, i32) -> () {
    argument_contracts = [#ttl.dispatch_argument<read, ordinary>, #ttl.dispatch_argument<write, ordinary>],
    argument_names = ["only_one"],
    operation_identity = "bad-names"
  }
}

// -----

// Verify that invocation operands match the target's declared input types.
module {
  ttl.dispatch.target @target : (i32) -> () {
    argument_contracts = [#ttl.dispatch_argument<read, ordinary>],
    argument_names = ["value"],
    operation_identity = "target"
  }
  func.func @controller(%value: i64) attributes {ttl.dispatcher} {
    // expected-error @below {{argument 0 has type 'i64', expected 'i32' for target 'target'}}
    ttl.dispatch.invoke @target(%value : i64)
    func.return
  }
}

// -----

// Verify that every invocation resolves to a dispatch target declaration.
module {
  func.func @controller(%value: i32) attributes {ttl.dispatcher} {
    // expected-error @below {{'missing' does not reference a ttl.dispatch.target}}
    ttl.dispatch.invoke @missing(%value : i32)
    func.return
  }
}

// -----

// Verify that target outputs use caller-owned operands instead of SSA results.
module {
  // expected-error @below {{must not declare results; target outputs are tensor operands owned by the caller}}
  ttl.dispatch.target @bad_result : (i32) -> i32 {
    argument_contracts = [#ttl.dispatch_argument<read, ordinary>],
    argument_names = ["value"],
    operation_identity = "bad-result"
  }
}
