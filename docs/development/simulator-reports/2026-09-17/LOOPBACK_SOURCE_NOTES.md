# Cross-DFB multicast-loopback source diagnosis

Follow-up: both focused repeats reproduced the predicted corrupt tile. See
[captured numerical evidence](NUMERICAL_TREE_LOOPBACK_NOTES.md) and
[final triage](TRIAGE.md). The remainder preserves the pre-probe source analysis.

Status: the emulator/hardware semaphore semantic mismatch below is confirmed by
source inspection. Its connection to this run's numerical failure is a specific,
falsifiable candidate pending the focused tensor probe. No runtime fix or extra
test was executed as part of this inspection.

## Provenance

- Compiler source under test: `70c5d093` in this worktree.
- Runtime image: `tt-lang-emule:compiler-suite-7292395`.
- Emulator commit: `7292395ce55a208a8ede0a4635a9f2167c8c4939`.
- Metal commit: `b6c508c4790fac0a11597e43a5421624cb461553`.
- Running-container emulator header:
  `/opt/tt-emule/include/jit_hw/api/dataflow/dataflow_api.h`.
- Read-only host copy:
  `<pinned-emulator-checkout>/include/jit_hw/api/dataflow/dataflow_api.h`.
- Both header copies have SHA-256
  `6c7f3f56c9ad6e7dec5b936d83ab3bb03acd69fc533f706ac2f92f35124dd0cc`.

Container source paths in this note were inspected in the active full-suite
container `d7ff0cdc6ee1`. They are not host filesystem paths. The upcoming probe
preserves its own generated kernels in the case report directories.

## Test and generated protocol

`test/python/pipe/test_pipenet_multi_iter.py:247-300` defines
`cross_dfb_multicast_loopback` and its test:

- Four cores `(0,0)` through `(3,0)` receive two stripes.
- Core `(0,0)` produces BF16 tiles filled with 7 into source CB0.
- All four cores receive into a distinct destination CB1, including the source.
- Destination storage has two 2048-byte slots.
- Output starts filled with -42; the final expected shape is 64 by 128, all 7.

Generated sender `/tmp/default/ttlang_kernel_dm_read_ac088fca.cpp`:

- Lines 126-129 wait for source CB0 and all four receiver-ready signals.
- Lines 130-147 compute `destination_base + slot * 2048`, advancing slot modulo 2.
- Line 148 sends CB0's read pointer to that address with `MCAST_INCL_SRC`.
- Line 149 completes payload writes before signaling completion.
- Line 150 calls `noc_semaphore_inc_multicast(..., increment=1, num_dests=3)`.
- Line 151 separately increments the source/root's completion semaphore by 1.
- Lines 152-154 complete atomics before consuming the source CB0 page.

This matches compiler lowering in
`lib/Dialect/TTL/Transforms/PipeLowering.cpp:2205-2236`: multicast atomics target
remote receivers, so the source is excluded from the destination count and
receives a separate local increment.

Generated receiver `/tmp/default/ttlang_kernel_dm_write_1375c1a8.cpp:119-139`:

1. Reserve destination CB1's next page and signal receiver-ready.
2. Increase a local expected-completion counter: 1 for stripe 1, 2 for stripe 2.
3. Wait for the completion semaphore to be at least that counter (line 129).
4. Publish CB1, immediately consume it, and write it to output.

Thus completion counts must advance exactly once per payload on every receiver.

## Confirmed emulator mismatch

The pinned emulator's `noc_semaphore_inc_multicast` at
`include/jit_hw/api/dataflow/dataflow_api.h:931-982`:

- Ignores `num_dests` (line 932).
- Iterates every worker in the multicast rectangle (lines 959-971).
- Increments every resolved worker's semaphore and wakes its waiter
  (lines 972-975).
- Does **not** exclude the caller/source coordinates.

The source is inside this test's rectangle, so line 150 of the generated sender
already increments its completion semaphore once. Line 151 increments it a second
time. The source/root therefore observes completion counts 2, 4 for two payloads,
while remote receivers observe 1, 2.

This is distinct from payload loopback support. Payload loopback exists:
`include/jit_hw/api/dataflow/noc.h:250-276` forwards `MCAST_INCL_SRC` to the runtime
`__emule_multicast_write` helper. The latter explicitly implements source
inclusion/exclusion in
`tt_metal/impl/emulation/emulated_program_runner.cpp:421-514`.

## Hardware ownership check

Pinned Metal's hardware source under `/opt/tt-emule-runtime/tt-metal` agrees with
the compiler's remote-plus-local completion split:

- `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2281-2302` documents that the
  multicast semaphore sender is not one of its destinations; `num_dests` is
  bounded by the number of other cores.
- The implementation at lines 2305-2324 invokes
  `noc_fast_multicast_atomic_increment`.
- Blackhole implementation
  `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h:1198-1250`
  issues a multicast atomic command. Its control flags at lines 1235-1238 contain
  `NOC_CMD_BRCST_PACKET`, but **not** `NOC_CMD_BRCST_SRC_INCLUDE`.
- In contrast, the explicit payload-loopback operation in the same file at
  lines 579-606 sets `NOC_CMD_BRCST_SRC_INCLUDE` at line 596.

The emulator multicast-atomic helper therefore lacks the hardware source
exclusion. This inspection does not claim a fresh hardware test result.

## Falsifiable numerical prediction

After the first payload, root completion may already be 2. Its second-stripe
wait-for-2 can then succeed before the second payload arrives, allowing the root
writer to publish and copy an unwritten destination slot.

The strongest expected signature is:

- Wrong output in **rows 32:64, columns 0:32**: root's second stripe.
- Those 1024 elements may be zero or stale destination-buffer contents rather
  than 7; the remaining seven tiles should contain 7.
- Race timing can mask the failure if the second payload arrives before the root
  reads it. Partial-tile corruption is also possible if the memcpy operations
  overlap in time, so the precise values are not guaranteed.
- The output's -42 sentinel should normally be overwritten even in the bad tile,
  because the writer still executes; preserved -42 would point to an additional
  or different delivery issue.

Inspect the probe's captured tensor, per-row/per-column mismatch totals, zeros,
sevens, and sentinel counts. Repeating with one fiber worker is diagnostic for
scheduling dependence, not a fix or a proof that the original protocol is sound.

If the tensor signature disagrees, retain the source-confirmed semaphore defect
but do not attribute this numerical failure to it without additional evidence.
