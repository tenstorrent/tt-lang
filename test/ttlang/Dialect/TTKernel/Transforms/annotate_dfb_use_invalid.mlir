// RUN: ttlang-opt %s -ttkernel-annotate-dfb-use --verify-diagnostics --split-input-file

// expected-error @below {{`ttkernel-annotate-dfb-use` requires finalized DFB allocation metadata; run `ttl-finalize-dfb-indices` first}}
module {
}

// -----

// expected-error @below {{`ttkernel-annotate-dfb-use` requires finalized DFB allocation metadata; run `ttl-finalize-dfb-indices` first}}
module attributes {ttl.dfb_allocations = 0 : i64} {
}
