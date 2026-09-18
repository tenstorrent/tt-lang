// Summary: Verifies DFB reconfiguration attribute and operation diagnostics.
// RUN: ttlang-opt --verify-diagnostics --split-input-file %s

func.func @negative_ordinal() {
  // expected-error @below {{DFB reconfiguration ordinal must be nonnegative}}
  ttl.dfb_reconfiguration #ttl.dfb_reconfiguration<-1, participants[#ttl.logical_kernel<kind = compute>, #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">, #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">]>
  return
}

// -----

func.func @wrong_participant_kinds() {
  // expected-error @below {{DFB reconfiguration requires one compute and two data movement participants}}
  ttl.dfb_reconfiguration #ttl.dfb_reconfiguration<0, participants[#ttl.logical_kernel<kind = compute>, #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">]>
  return
}

// -----

func.func @duplicate_participant() {
  // expected-error @below {{DFB reconfiguration participants must be distinct}}
  ttl.dfb_reconfiguration #ttl.dfb_reconfiguration<0, participants[#ttl.logical_kernel<kind = compute>, #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">, #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">]>
  return
}

// -----

func.func @noncanonical_participants() {
  // expected-error @below {{DFB reconfiguration participants must use canonical order}}
  ttl.dfb_reconfiguration #ttl.dfb_reconfiguration<0, participants[#ttl.logical_kernel<kind = compute>, #ttl.logical_kernel<kind = data_movement, identity = "writer", operation = "operation">, #ttl.logical_kernel<kind = data_movement, identity = "reader", operation = "operation">]>
  return
}
