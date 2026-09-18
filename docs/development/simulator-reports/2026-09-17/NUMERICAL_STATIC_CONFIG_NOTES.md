# Mixed exponential/broadcast numerical failure

This compact export retains compiled metadata and per-run summaries. Tensor
captures and other per-case artifacts referenced below remain in the local raw
archive, described in the [archive policy](README.md).

## Scope and result

This is read-only analysis of the existing diagnostic captures, not a new
emulator execution or a fix-validation run. Compiler source is
`70c5d093df06d211aa78332447bbd8d602d2ba79`; emulator source is
`7292395ce55a208a8ede0a4635a9f2167c8c4939`, with Metal
`b6c508c4790fac0a11597e43a5421624cb461553`.

The first-core-only descriptor hypothesis has strong, specific confirmation
for both variants of
`test_user_dfb_reuse.py::test_disjoint_incompatible_static_configurations_do_not_reuse_dfb`.
All four captures (two runs, DRAM and L1) reproduce the exact wrong-format
decode predicted before tensor capture. This is not evidence of generally
unsupported FP32 arithmetic or insufficient exponential accuracy.

## Captured evidence

Artifacts are under `numerical-probe/run-001/` and `numerical-probe/run-002/`.
Each contains these case directories:

- `python_test_user_dfb_reuse.py_test_disjoint_incompatible_static_configurations_do_not_reuse_dfb_dram_-8dee17d963-call`
- `python_test_user_dfb_reuse.py_test_disjoint_incompatible_static_configurations_do_not_reuse_dfb_l1_-d5d6e7ef91-call`

Inspect `tensors.pt`, `compiled-metadata.json`, `final.mlir`, `summary.json`,
and `traceback.txt` in each directory. The saved tensors were loaded with
`torch.load(..., map_location="cpu", weights_only=True)` using the host
`<host-python>`; no device runtime was used.

The metadata agrees in all four captures:

| Physical DFB | Role | Allocation nodes | Format | Page bytes | Storage index |
| --- | --- | --- | --- | --- | --- |
| 0 | Exponential input | `(0,0)` | float32 | 4096 | 0 |
| 1 | Output | `(0,0)`, `(1,0)` | float32 | 4096 | 1 |
| 2 | Broadcast input | `(1,0)` | float32 | 4096 | 0 |

The generic kernel covers both cores (`{[0-0 - 1-0]}`); per-kernel specialized
core ranges are null. The compiler correctly keeps three physical indices.
The two inputs share a storage index only across disjoint cores, not within
one core. In particular, the missing-on-first-core broadcast slot is **DFB 2**,
not DFB 1.

The captured `final.mlir` line 1 confirms these allocations. Its compute
configuration at line 2 enables FP32 DST and unpack-to-destination for input 0.
Lines 19-27 copy input 0, compute exponential, and pack output 1 on core 0.
Lines 38-44 initialize and execute `unary_bcast<BroadcastType::ROW>` on input 2,
then pack output 1 on core 1. The failure reaches the numerical assertion;
the three-allocation assertion passes.

| Capture | Left-half mismatches | Right-half mismatches | Right-half exact wrong-decode match |
| --- | --- | --- | --- |
| run-001 DRAM | 0 | 992 | yes |
| run-001 L1 | 0 | 992 | yes |
| run-002 DRAM | 0 | 992 | yes |
| run-002 L1 | 0 | 992 | yes |

Mismatch counts use the test's `rtol=0.002`, `atol=0.0005`. The exponential
half's maximum absolute error is `1.1920928955078125e-07` in every capture.
Every row of the right half is exactly:

```text
[0, 0, 0, 0.03125, 0, 0.0625, 0, 0.09375,
 0, 0.125, 0, 0.15625, 0, 0.1875, 0, 0.21875,
 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

As an independent arithmetic check, face-layout-encoding the captured FP32
`second_input`, reinterpreting its bytes as BF16, decoding the first 1024 BF16
elements as a 32x32 four-face tile, and broadcasting its first row matches
the entire captured right half bit-for-bit in every case.

## Mechanism and ownership

The pinned runtime's
`tt_metal/impl/emulation/emulated_program_runner.cpp:1573-1679` builds shared
JIT format/geometry descriptors from only `first_core`, using
`circular_buffers_on_core(first_core)` and `dataflow_buffers_on_core(first_core)`.
Unconfigured entries remain format 255 (`Invalid`). DFB 2 is absent on core 0,
although the same compute kernel uses it on core 1.

The relevant emulator sources are relative to
`<pinned-emulator-checkout>`:

- `include/jit_hw/api/cb_api.h:60-82`: constexpr format tables come from the
  JIT defines, not the currently executing core's live buffer metadata.
- `include/jit_hw/api/compute/common.h:430`: `cb_data_format` reads that table.
- `include/jit_hw/api/compute/common.h:1025-1029`: tile loading selects the
  decoder from that format and tile geometry.
- `include/jit_hw/api/compute/common.h:1016-1021`: an invalid format reaches
  the BF16 decoder. It is not inferred as FP32 from the 4096-byte page.
- `include/jit_hw/api/compute/bcast.h:161-169`: unary row broadcast first loads
  the input tile, then replicates its decoded first row.
- `include/jit_hw/api/compute/nfaces.h:48-53`: the face-layout indexing used by
  the independent byte-reinterpretation calculation.
- `include/jit_hw/api/compute/common.h:632-646`: output DFB 1 is correctly
  known as FP32, so packing preserves the already-misdecoded values.

The hardware descriptor collector differs: pinned Metal
`tt_metal/impl/program/program.cpp:2749` calls `set_cb_data_fmt_and_tile` with
all logical kernel core ranges; lines 2321-2333 visit every intersecting CB.
The intersection predicate is in `impl/buffers/circular_buffer.cpp:100-102`.
Thus a consistently typed physical slot used only on another core is still
represented in the hardware compilation descriptors.

This localizes the probable fix to emulation descriptor collection, not the
compiler's allocation policy or FP32 arithmetic. Match the hardware's
kernel-range-wide descriptor collection and validate consistency for slots
present on multiple cores. Do not add a blanket page-size-to-format fallback:
page size does not uniquely identify data format and would hide other errors.

## Confidence and remaining validation

Confidence is high: the captured sparse allocation metadata and all four
full output tensors agree with the source-derived prediction. The descriptor
selection code was inspected directly in the active runtime source earlier;
the exact emitted `EMULE_CB_DATA_FORMATS` string is not included in these
captures. No patched-runtime A/B validation has been performed, so this note
does not claim the defect is fixed or all other numerical failures share it.

After a narrowly scoped runtime change, rerun both captured variants and
compare full tensors. Add emulator regressions for a generic multi-core
kernel whose input or output slot exists only away from its first core;
exercise FP32 and at least one non-default tile shape, plus a uniform-core
control. Existing tree-reduction failures are related candidates and need
their own captured evidence and post-fix validation.
