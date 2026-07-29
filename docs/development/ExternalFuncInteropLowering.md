# External Function Interop Lowering

## Overview

External function interop in TT-Lang is expressed through `ttl.call_extern_func`
and lowered through the normal compiler pipeline:

1. Python AST emits `ttl.opaque_call`.
2. TTL to TTKernel conversion lowers argument categories.
3. TTKernel to EmitC conversion resolves template arguments to integer literals.
4. TTKernel C++ emission inserts the requested header include and emits the call.

The call remains opaque to the compiler. No semantic validation of the external
function signature is performed beyond argument-kind lowering rules.

## Template argument resolution

`template_args` are delayed until TTKernel to EmitC lowering and must be
compile-time evaluable.

- Scalar literals (`int`, `bool`, `float`) are lowered to i32 SSA values.
  - `float` values use IEEE-754 f32 bit encoding.
- Direct DFB values in `template_args` are auto-detected by the Python frontend
  and lowered to integer DFB indices.

`ttl.call_extern_func(..., template_args=[ttl.get_dfb_id(dfb)])` is rejected in
frontend syntax checks; direct DFB values are the required user-facing form.

## Function argument mapping

`func_args` are lowered by category in `ConvertTTLToTTKernel`:

- DFB -> CB index via `ttkernel.get_compile_time_arg_val`.
- Tensor -> `ttkernel.TensorAccessor` materialization by default.
- `ttl.raw_addr(tensor)` -> raw i32 base address from runtime common args.
- Scalar values -> forwarded, with scalar float bit-pattern rewriting performed
  later by `ttkernel-lower-scalar-fp-types`.

## `ttl.raw_addr` constraints

`ttl.raw_addr` only accepts base tensor function arguments with TTL layout
encoding. Slice/view-like values are intentionally rejected. This preserves
deterministic runtime-argument mapping and avoids implicit offset semantics.

Follow-up work should define explicit semantics for slice/view-like operands,
including base-address-plus-offset legality and layout interactions.
