# Focused numerical probe: tree reduction and multicast loopback

## Result and confidence

Both independent probe processes reproduced the same failures and passing
controls. The artifacts strongly support two distinct emulator defects:

1. **Generic FP32 tree reduction: missing non-root-only CB metadata.** A host-side
   model of the resulting BF16 byte packing into an FP32 buffer reproduces
   **all 448 output float32 bit patterns exactly**, for both DRAM and L1 in both
   repeats. This is byte-interpretation corruption, not ordinary BF16 precision
   loss or a missing subtree sum.
2. **Cross-DFB multicast loopback: sender completion counted twice.** Both repeats
   have exactly the predicted 1024-element bad tile: rows `32:64`, columns `0:32`
   are zero, while all other 7168 elements are 7. This exactly matches the
   premature-publication region predicted from the source-confirmed multicast
   atomic sender-inclusion mismatch.

No emulator source was changed and no patch-based counterfactual test was run.
The tree conclusion combines actual compiler metadata, runtime source, and an
exact independent byte-level numerical reconstruction. The loopback conclusion
combines a confirmed hardware/emulator semantic mismatch and its exact predicted
output region; the semaphore counter itself was not instrumented at runtime.

## Provenance and scope

- Compiler: `70c5d093df06d211aa78332447bbd8d602d2ba79`, clean when the sweep started.
- Runtime image: `tt-lang-emule:compiler-suite-7292395`.
- Emulator: `7292395ce55a208a8ede0a4635a9f2167c8c4939`.
- Metal: `b6c508c4790fac0a11597e43a5421624cb461553`.
- Probe: `numerical-probe/run-001` and `numerical-probe/run-002`, each a separate
  Python/pytest process, `TT_METAL_EMULE_MODE=1`, no fiber-worker override.
- Each repeat: five expected-to-investigate failures and five passing controls.
- Passing controls relevant here: `test_gather_bcast_multi_iter`, specialized FP32
  tree reduction with DRAM and L1, and generic BF16 tree reduction with DRAM and L1.
- Postprocessing used host Python
  `<host-python>`, Torch `2.12.0`, loading only
  saved synthetic tensors with `weights_only=True`. It did not launch new device
  or emulator tests.

The mixed exponential/broadcast cases are analyzed separately in
`NUMERICAL_STATIC_CONFIG_NOTES.md`.

## Artifact locations

Within each `numerical-probe/run-00N/` directory:

| Case | Case directory |
| --- | --- |
| Loopback | `python_pipe_test_pipenet_multi_iter.py_test_cross_dfb_multicast_loopback-d7e77ee8c8-call` |
| Tree DRAM FP32 | `python_pipe_test_pipenet_tree_reduce.py_test_tree_reduce_default-dram-fp32_-948d021766-call` |
| Tree L1 FP32 | `python_pipe_test_pipenet_tree_reduce.py_test_tree_reduce_default-l1-fp32_-f4e2be84ef-call` |

Only `compiled-metadata.json` is versioned in each case directory here. The raw
archive also contains `tensors.pt`, the per-case `summary.json`, generated source,
and the captured failure traceback. These are the primary dynamic evidence.
The tracked per-run `summary.json` records control outcomes; `junit.xml` remains
in the raw archive. See the [archive policy](README.md).

The generated-source paths in `compiled-metadata.json` match those inspected
read-only during the full sweep. Each case's `generated_kernels` list identifies
the C++ files copied into its `generated/` directory; `captured.stdout.txt` also
contains the generated source and kernel-path announcements. For example, the
first generic DRAM FP32 case preserves `ttlang_kernel_compute_3a6c1e3e.cpp`,
`ttlang_kernel_exchange_3249908d.cpp`, and
`ttlang_kernel_store_output_bf146192.cpp`. Use explicit paths or
`rg --files --no-ignore` when inventorying these files: ordinary `rg --files`
hides generated files that match repository ignore patterns.

## FP32 tree: metadata confirms the missing slot

Both captured generic FP32 variants compile one compute kernel across
`{[0-0 - 7-0]}` with no specialized per-kernel range. The metadata contains:

| Physical CB | Purpose | Format / tile | Allocation nodes |
| --- | --- | --- | --- |
| 0 | Non-root accumulator | FP32 / 32x32 / 4096-byte page | `(1,0)` through `(7,0)` only |
| 1 | Received child partials | FP32 / 32x32 / 4096-byte page | `(0,0)`, `(2,0)`, `(4,0)`, `(6,0)` |
| 2 | Staged source input | FP32 / 32x32 / 4096-byte page | All eight cores |
| 3 | Root output | FP32 / 16x32 / 2048-byte page | `(0,0)` only |

The root genuinely has **no CB0 descriptor**. This is not inferred merely from
the Python kernel's branch structure: it appears in the actual captured
`cb_configs[].allocation_nodes`.

Relevant source chain:

1. Compiler metadata is emitted by
   `lib/Dialect/TTL/Transforms/TTLFinalizeDFBIndices.cpp:153-157`.
2. Host allocation respects those nodes even for generic kernels in
   `python/ttl/kernel_runner.py:2983-3061`.
3. Pinned emulator `tt_metal/impl/emulation/emulated_program_runner.cpp:1573-1679`
   collects all JIT CB format/geometry descriptors using only the kernel's first
   core, here `(0,0)`. An absent CB slot is assigned `DataFormat::Invalid` (255).
4. `include/jit_hw/api/compute/common.h:430-439` uses that format enum alone;
   Invalid is not recognized as a 32-bit format. The format guard permits Invalid
   around line 543. The pack implementation then falls through to its default
   BF16 path at lines 703-720.
5. Generic non-root compute correctly calls `init_sfpu(..., CB0)` before packing,
   but this cannot repair missing compile-time format metadata. It writes BF16
   bytes into a page that the receiver's valid CB1 descriptor interprets as FP32.
6. Specialized non-root kernels compile against their own core, where CB0's FP32
   descriptor exists. Generic BF16 does not expose this format substitution.

### Hardware ownership

Pinned Metal's hardware path does not have the same first-core restriction:

- `tt_metal/impl/program/program.cpp:2749` passes all kernel core ranges to
  `set_cb_data_fmt_and_tile`.
- The collector at lines 2321-2333 adds every intersecting CB descriptor.
- `tt_metal/impl/buffers/circular_buffer.cpp:100-102` defines the predicate using
  range intersection, not containment of one first core.

Therefore a slot present only on another executing core is included in hardware's
descriptor table. This is an emulator/hardware inconsistency for sparse per-core
allocation with consistent descriptors per physical slot, not evidence that the
compiler must allocate unused buffers on root. No fresh hardware execution was
performed in this diagnostic pass.

## FP32 tree: exact numerical reconstruction

Observed in all four captures (two repeats times DRAM/L1):

- 448 of 448 values fail the test's tolerance.
- Maximum absolute error against the intended result: `10.828098773956299`.
- RMSE against the intended result: approximately `3.3639893531799316`.
- DRAM and L1 captured outputs match, and both repeats match.
- The wrong result is not explained by simply omitting or duplicating complete
  source tiles: an eight-input linear fit still has RMSE about `2.16810207145`.
- Modeling the specific wrong packing/interpretation produces **448 of 448 exact
  float32 bit matches**, with maximum reconstruction error `0.0`, in every capture.

The model follows the emitted tree: leaves 1,3,5,7; core 2 consumes 3; core 6
consumes 7; core 4 consumes 5 then 6; root consumes 1,2,4. Every non-root pack uses
BF16 round-to-nearest-even, including L1 pack accumulation. The receiver then
interprets those bytes as an FP32 face-ordered tile. Root's own output packing
remains FP32.

The following is the exact host-only reconstruction used. Run from the worktree
root after restoring the raw archive under `.ttlang-sim/`; it only reads the
captured synthetic tensors and cannot run from the compact Git export alone:

```python
from pathlib import Path
import torch

base = Path(".ttlang-sim/reports/20260917T233554Z-pyfnmtqs/numerical-probe")

def bf16_pack_read_as_fp32(tile):
    faces = (
        tile.to(torch.bfloat16)
        .reshape(2, 16, 2, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(-1)
    )
    raw = torch.zeros(2048, dtype=torch.bfloat16)  # 4096-byte FP32 page
    raw[:1024] = faces                          # erroneous 2048-byte BF16 pack
    return (
        raw.view(torch.float32)
        .reshape(2, 2, 16, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(32, 32)
    )

for path in sorted(base.glob("run-*/python_pipe_*tree*fp32*/tensors.pt")):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    source = saved["source_host"]
    accumulators = {}
    for core in [7, 6, 5, 4, 3, 2, 1]:
        value = source[:, core * 32 : (core + 1) * 32].to(torch.bfloat16)
        for child in {2: [3], 4: [5, 6], 6: [7]}.get(core, []):
            value = (
                value.float() + bf16_pack_read_as_fp32(accumulators[child])
            ).to(torch.bfloat16)
        accumulators[core] = value
    predicted = source[:16, :32].clone()
    for child in [1, 2, 4]:
        predicted += bf16_pack_read_as_fp32(accumulators[child])[:16]
    predicted = predicted[:14]
    actual = saved["actual"]
    assert torch.equal(predicted.view(torch.int32), actual.view(torch.int32))
    print(path, "448/448 bitwise matches")
```

The emulator BF16 rounding operation is defined in
`include/jit_hw/api/bfloat16.h:21-27`; it matches the host Torch BF16 conversion
for these finite inputs. The reconstruction's zero-filled unused page half is
consistent with these captures, not a general promise about all buffer reuse.

## Loopback: exact predicted region reproduced twice

Both `run-001` and `run-002` contain:

- Output shape 64x128, BF16.
- Exactly 1024 mismatches, all zero.
- Mismatch mask equals rows `32:64`, columns `0:32` exactly.
- Exactly 7168 other values, all 7.
- No output values retain the -42 initialization sentinel.

This is the source/root's second stripe, exactly as predicted before inspecting
the tensor. The writer executes but publishes its second destination page too
early; this is not a missing output-store path or a generally unsupported
multicast/loopback operation.

Full source and hardware references are preserved in `LOOPBACK_SOURCE_NOTES.md`.
In brief, the sender emits remote multicast completion plus an explicit local
completion increment, matching hardware's source-excluding multicast atomics.
Emule's multicast-atomic implementation instead increments every worker in the
rectangle, including the source. Root therefore sees completion counts 2 and 4,
and its second wait-for-at-least-2 can pass after the first payload.

Two independent repeats already establish the exact predicted signature. A
one-worker rerun could characterize scheduling sensitivity, but is not needed to
identify this defect and would not replace a focused semantic regression.

## Recommended follow-up, not performed here

- Emulator JIT metadata: collect consistent descriptors over the complete kernel
  core range, matching Metal's hardware descriptor collection, and reject
  incompatible same-index descriptors rather than silently picking one.
- Emulator format handling: consider making unexpected Invalid-format compute
  use fail loudly instead of silently falling through to BF16.
- Multicast atomics: implement source exclusion for `noc_semaphore_inc_multicast`,
  preserving explicit source-inclusive payload behavior and existing NoC1 rules.
- Add targeted regressions for sparse non-root FP32 CB descriptors and two-stripe
  source-in-range multicast completion, then rerun these controls and the suite.

Those are proposed fixes for a later authorized implementation pass; this pass
only diagnosed and recorded evidence.
