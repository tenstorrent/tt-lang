// Verifies the TTL to TTKernel pipeline rejects invalid accumulation strategy
// option values before lowering.
//
// RUN: ttlang-opt %s --ttl-to-ttkernel-pipeline='accumulation-strategy=invalid' --verify-diagnostics --split-input-file

// expected-error @below {{op invalid accumulation strategy `invalid`; expected auto, dst, or l1-pack}}
func.func @invalid_accumulation_strategy() {
  func.return
}
