# Resident intermediate DFB: PACK→UNPACK synchronization findings (WIP)

## Context

When a tile-level op writes a value to the dest registers (DST) and that value is
then consumed by a *DFB-input op* (one whose tile lowering requires its operand
to come from a circular buffer — e.g. `matmul`, `reduce`, `transpose`), the
compiler must materialize the intermediate to L1 through a dataflow buffer (DFB /
circular buffer, "CB"). `TTLInsertIntermediateDFBs` does this by emitting
`bind_cb` → `cb_reserve` → `store` (a `pack_tile`) → `cb_wait` → `attach_cb`.

We were exploring whether such compiler-inserted intermediates could use a
**"resident scratchpad" pattern** — reserve once, then pack into and read from a
single fixed slot every iteration, with **no `cb_push_back` / `cb_wait_front` /
`cb_pop_front` handshake** — to remove per-use CB sync traffic. (The user-facing
`.read()` / `.store()` resident primitives are a separate, already-landed change;
this doc is about the compiler auto-applying the no-handshake form to *inserted*
intermediates.)

**Conclusion up front:** the bare no-handshake scratchpad is **not safe in
general**. The consumer's unpack of the intermediate races the producer's pack —
the unpack can read the CB's L1 region before the pack has flushed to it. The CB
handshake is the only mechanism that orders a PACK-thread L1 write before an
UNPACK-thread L1 read, and no cheaper *single-instruction* stall can replace it.
Where it appeared to "work" (matmul consumer), it was a **timing artifact**, not a
guarantee.

## Hardware background (tt-metal LLK, Wormhole/Blackhole)

A compute kernel runs on three RISCs: **UNPACK (TRISC0) → MATH (TRISC1) →
PACK (TRISC2)**, each executing only its own instructions, in program order.

- `tile_regs_acquire/commit/wait/release` synchronize **MATH↔PACK on the DST
  register only** (via the `MATH_PACK` semaphore). The UNPACK thread does not
  participate in `tile_regs_*` at all. There is **no PACK↔UNPACK semaphore** in
  hardware.
- Therefore `tile_regs_*` provides **zero ordering** between the PACK thread
  writing a tile to L1 and a later UNPACK thread reading that L1.
- The **CB handshake is the only PACK→UNPACK sync**: `cb_push_back` (PACK thread)
  does `TTI_STALLWAIT(STALL_THCON, PACK)` — "wait for the pack to finish" — and
  *then* bumps the `tiles_received` counter in L1; `cb_wait_front` (UNPACK thread)
  blocks polling `tiles_received`. The signal is a **monotonic per-data counter**:
  "*this specific tile* is now in L1."
- tt-metal compute kernels universally use the full handshake for any CB
  round-trip; even in-place helpers (e.g. SDPA's `max_block_inplace`) pack into a
  CB and then do `pop_front`/`reserve_back`/`push_back` before re-reading. No
  shipped tt-metal kernel uses a bare (handshake-free) scratchpad.

## What we observed

Minimal tt-lang op: `out = transpose(2 * x)`. The `2*x` is plain SSA fed into
`transpose` (a DFB-input op), so the compiler materializes it into one resident
intermediate CB, then `transpose_wh` reads it. We compared the compiler emitting
the intermediate **with** the handshake vs **resident (no handshake)**.

### Consumer-op dependence (single iteration)

Run on hardware, no-handshake intermediate, consumer varied:

| consumer of the intermediate CB | no-handshake PCC | with-handshake PCC |
| --- | --- | --- |
| `transpose_wh` | 0.12 ❌ | 1.0 |
| `reduce` (`reduce_max`) | 0.83 ❌ | 1.0 |
| `reduce` of a matmul result | 0.25 ❌ | 1.0 |
| `matmul` (`y @ w`) | **1.0 ✅** | 1.0 |

The matmul consumer was the only one that passed without the handshake.

### Why matmul "passed" — a timing artifact, not a contract

The matmul consumer's codegen has a `cb_wait_front` on its *other* operand (the
DM-fed `w`) sitting between the intermediate's pack and the matmul's read. That
unrelated wait stalls the UNPACK thread long enough that the intermediate's pack
retires before it is read. The tt-metal LLK confirms there is **no** init/unpack
barrier that the matmul unpack has and `reduce`/`transpose` lack — all three omit
PACK→UNPACK sync identically. So matmul's pass is incidental scheduling delay
(its heavier two-operand, address-reprogramming unpack MOP plus the `w` wait), not
a synchronization guarantee. Remove the delay and it would race like the others.

### Single iteration masks the race; a loop exposes it

A single iteration on a freshly opened device can pass even for the racing
consumers, because the intermediate's L1 slot starts clean — if the unpack races,
it reads zero/garbage on a fresh slot, which on a single short run can still land
correct or close. **Reusing the resident intermediate across a loop, with distinct
data per iteration, reliably exposes the race**: the consumer reads the *prior*
iteration's stale slot.

Standalone tt-metal (`ttnn.generic_op`) reproducer, 64 iterations, distinct data,
`transpose_wh` consumer:

| mode | overall PCC | bad iterations |
| --- | --- | --- |
| full handshake | 1.000000 | 0 / 64 |
| bare scratchpad (no handshake) | 0.577 | 64 / 64 |
| `TTI_STALLWAIT(STALL_UNPACK, PACK)` barrier | 0.577 | 64 / 64 |
| 64 unpacker `TTI_NOP`s | 0.577 | 64 / 64 |

### Why a single-instruction stall does NOT fix it

`TTI_STALLWAIT(STALL_UNPACK, PACK)` stalls the unpacker until the packers are
**idle**. But the PACK thread *lags* the UNPACK thread in the pipeline. When
UNPACK reaches the stall (right after unpacking this iteration's producer inputs),
MATH may still be computing and **PACK has not started this iteration's pack yet**
— so "packers idle" is momentarily true, the stall passes immediately, and UNPACK
goes on to read the stale slot. An idle check answers "is a pack in flight *right
now*"; it cannot answer "has *this* data been produced yet." That is exactly the
property the `tiles_received` counter provides and a bare stall (or NOPs) does not.

## Deterministic options (to drop the FIFO bookkeeping while staying correct)

1. **Full CB handshake** — `cb_push_back` / `cb_wait_front` / `cb_pop_front`
   (+`cb_reserve_back`). Correct, deterministic, what every tt-metal kernel uses.
   Highest sync cost (two L1 counter stores + a poll loop + pointer bookkeeping).
2. **A `t6_semaphore` post/wait** — the PACK thread posts a spare semaphore *after*
   its pack (`t6_semaphore_post<PACK>`), the UNPACK thread waits-on-nonzero *before*
   the read. This is a counter/semaphore signal (so it handles the pipeline lag),
   but with no FIFO pointer bookkeeping — cheaper than the full handshake. This is
   the same class of mechanism used by some DST-resident tt-metal compute kernels.
   **Not yet tested here** — this is the remaining candidate for "remove the FIFO
   traffic but stay correct."

A bare `STALL_UNPACK/PACK` stall and NOP padding are **ruled out** (above).

## Status

- The compiler-inserted-intermediate no-handshake change (`TTLInsertIntermediateDFBs`
  emitting `block_count = 1` + `resident`) is **WIP and currently unsafe** — it
  corrupts looping multi-stage ops. The accompanying `.mlir` test updates and the
  `TTLVerifyPipeNetGuards` resident-skip are part of the same WIP.
- A diagnostic build with `block_count = 1` **but the handshake kept** passes — so
  the L1 saving from a single block is fine; only the no-handshake elision is unsafe.
- The user-facing `.read()` / `.store()` resident primitives are a separate, landed
  change and are unaffected (the caller takes responsibility for the pattern).

## Reproducer

`cbsync/` is a standalone `ttnn.generic_op` harness (no tt-lang compiler in the
loop) that reproduces and isolates all of the above:

- `kernels/compute.cpp` — the loop kernel with `-D` toggles:
  `HANDSHAKE`, `BARRIER` (`TTI_STALLWAIT(STALL_UNPACK, PACK)`), `NOPS=N`, or none
  (bare scratchpad).
- `kernels/compute_emitted.cpp` — the exact tt-lang-emitted resident kernel for
  `transpose(2*x)`, for byte-level comparison against the hand-written one.
- `kernels/reader.cpp` / `kernels/writer.cpp` — stream N input/output chunks.
- `run_cbsync.py` — `CBSYNC_MODE={baseline,handshake,barrier,nops}`,
  `CBSYNC_ITERS` (default 64), `CBSYNC_SEED`. Prints overall and per-iteration PCC.

Run: `CBSYNC_MODE=baseline CBSYNC_ITERS=64 python3 cbsync/run_cbsync.py`
