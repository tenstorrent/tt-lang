// Verify that opaque-call lowering rejects unrepresentable DFB descriptors.
// RUN: ttlang-opt %s --split-input-file --verify-diagnostics --convert-ttl-to-ttkernel

// A descriptor requires a byte-addressable DFB page size.
func.func @sub_byte_descriptor() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 2, block_count = 4} : !ttl.cb<[2, 3], i4, 4>
  // expected-error @below {{'ttl.opaque_call' op DFB descriptor element type must occupy a positive whole number of bytes, got 'i4'}}
  ttl.opaque_call "describe" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%dfb : !ttl.cb<[2, 3], i4, 4>) () {header = "describe.hpp"} : () -> ()
  return
}

// -----

// Descriptor page counts must fit the generated uint32_t template parameter.
func.func @descriptor_page_count_overflow() attributes {ttl.kernel_thread = #ttkernel.thread<noc>} {
  %dfb = ttl.bind_cb {cb_index = 2, block_count = 1} : !ttl.cb<[4294967296], i8, 1>
  // expected-error @below {{'ttl.opaque_call' op DFB descriptor dimensions or page size exceed uint32_t}}
  ttl.opaque_call "describe" template_args [#ttl.external_template_arg<dfb_descriptor, 0>] template_dfbs(%dfb : !ttl.cb<[4294967296], i8, 1>) () {header = "describe.hpp"} : () -> ()
  return
}
