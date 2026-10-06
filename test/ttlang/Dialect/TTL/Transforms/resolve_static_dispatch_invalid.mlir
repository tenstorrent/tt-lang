// RUN: ttlang-opt %s --split-input-file --ttl-resolve-static-dispatch --verify-diagnostics

// Verify that unresolved runtime control flow is rejected instead of being
// converted into a runtime reload-table branch.

module {
  ttl.dispatch.target @target : (i32) -> () {
    argument_contracts = [#ttl.dispatch_argument<read, ordinary>],
    argument_names = ["value"],
    operation_identity = "target"
  }
  func.func @controller(%condition: i1, %value: i32)
      attributes {ttl.dispatcher} {
    // expected-error @below {{remains in a static dispatch controller}}
    scf.if %condition {
      ttl.dispatch.invoke @target(%value : i32)
    }
    func.return
  }
}

// -----

// Verify that an ordinary value cannot carry a producer result into a later
// reload image.
module {
  ttl.dispatch.target @producer : (i32) -> () {
    argument_contracts = [#ttl.dispatch_argument<write, ordinary>],
    argument_names = ["value"],
    operation_identity = "producer"
  }
  ttl.dispatch.target @consumer : (i32) -> () {
    argument_contracts = [#ttl.dispatch_argument<read, ordinary>],
    argument_names = ["value"],
    operation_identity = "consumer"
  }
  func.func @controller(%value: i32) attributes {ttl.dispatcher} {
    ttl.dispatch.invoke @producer(%value : i32)
    // expected-error @below {{declare handoff or persistent_state storage}}
    ttl.dispatch.invoke @consumer(%value : i32)
    func.return
  }
}

// -----

// Verify that an empty dispatcher cannot resolve to a reload sequence.
module {
  // expected-error @below {{static dispatch controller must invoke at least one target}}
  func.func @controller() attributes {ttl.dispatcher} {
    func.return
  }
}

// -----

// Verify that the dispatcher marker uses the frontend's unit-attribute form.
module {
  ttl.dispatch.target @target : (i32) -> () {
    argument_contracts = [#ttl.dispatch_argument<read, ordinary>],
    argument_names = ["value"],
    operation_identity = "target"
  }
  // expected-error @below {{requires 'ttl.dispatcher' to be a unit attribute}}
  func.func @controller(%value: i32) attributes {ttl.dispatcher = "yes"} {
    ttl.dispatch.invoke @target(%value : i32)
    func.return
  }
}

// -----

// Verify that dispatchers use caller-owned output operands rather than returns.
module {
  ttl.dispatch.target @target : (i32) -> () {
    argument_contracts = [#ttl.dispatch_argument<read, ordinary>],
    argument_names = ["value"],
    operation_identity = "target"
  }
  // expected-error @below {{must not return values; target outputs are tensor operands}}
  func.func @controller(%value: i32) -> i32 attributes {ttl.dispatcher} {
    ttl.dispatch.invoke @target(%value : i32)
    func.return %value : i32
  }
}
