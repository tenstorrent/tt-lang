# Gemma 4 decode bringup — implementation notes

Running log of compiler gaps, deferred optimizations, gotchas, and perf
deltas found while building the Gemma 4 26B-A4B decode atoms. Extend as
work progresses; entries should say what was hit, the workaround, and what
to circle back to.

## Deferred optimizations

- **GEMV K-blocked weight stream**: per-core W blocks must span the whole
  K-band because block subviews (`x_blk[0:1, k0:k1]`) are not supported
  yet (#671). With subviews, GEMV could stream `(bk, bn)` blocks and start
  compute on the first 8 tiles instead of the full band, shrinking L1
  pressure and warmup latency. Circle back when #671 lands.
- **GEMV Kp>2 reduce**: column reduce reuses matmul's Kp=2 single step.
  130-core layouts want Kp in {4, 5} with a tree.
- **Multi-card Pipes**: collectives cut atoms at ttnn CCL today. Pipes with
  a device dimension + fabric lowering remove those cuts (fabric details in
  /tmp/fabric_spike.md).

## Fixed bugs

- **attach_cb broke read/write-ptr selection**: raw element reads on a
  `wait()`ed block addressed via `get_write_ptr` because the wait was hidden
  behind `ttl.attach_cb` (fixed: trace attach_cb in isBlockFromCBWait;
  needs upstream comment on the elementRW PR). Symptom: same-thread
  reserve→copy→wait→read silently reads zeros; cross-thread looks fine.

## Conventions

- Gemma's (1+w) RMSNorm uses the stock rmsnorm op; the host loader stores
  1+w. Scale-less norms (router, v_norm) pass an all-ones weight; replace
  with a no-weight variant only if it shows up in profiles.

## Gotchas

- Python-bool `if` statements and ternaries do NOT trace inside atom
  bodies (fails as "Binary operands not found" pointing at a bogus
  docstring line). Variants must be separate function bodies; closure
  scalars (ints/floats) are the only compile-time parametrization.
  Reproducer: a pristine flash core with `if False:` around v_blk.

- Conditional `ttl.copy` under data-dependent `if`/`else` shared one CB
  block before the ptr fix; retest now (was: both branches read garbage,
  appeared as "always takes one branch").
- `raw_element_read` value math lowers bit-level; only cmpf, fptosi(f32/bf16
  to i32) and extf(bf16 to f32) are handled. Anything else dies in
  unrealized_conversion_cast legalization.
- One hw test at a time on card1 (4 cards); reset only via reset script.
- DM thread limit: 1 compute + 2 datamovement kernels per core.

## Perf deltas

- gemv e2e wall-clock is dispatch-bound: ~0.23 ms floor across all shapes
  (PCC 1.0). At 16 MB streamed that floor reads 127 GB/s; the weight stream
  itself is much faster. Use a cycles/ Tracy variant before drawing GB/s
  conclusions, and expect the fused multi-layer atom to amortize dispatch.
- indexed_gemv 8-expert gate_up stream (32E resident, K=2816, N=1408):
  0.28 ms e2e = 226 GB/s including dispatch; the dominant MoE DRAM term is
  already near the 260-300 GB/s/card budget before any device-side tuning.

## Fused-atom findings (attn stage A)

- mcast self-delivery on the source core corrupts the received block
  (heads on the loopback core were noise, all others exact). Workaround:
  the norm/mcast source core does no compute (9-col grid, col 0 src
  only). File a reproducer later; fix would also let the source core
  consume a CB block directly without pipe loopback.
- Added mcast_block (DFB-source mcast); copying via a DRAM scratch row
  between same-thread write and read races: writes are not flushed
  before the next copy reads.
- L1 budget: ~1.46 MB cap is the binding constraint; chunk K bands at
  22t for streamed weights and stream the norm in 11t chunks. RoPE
  rotate-half as h@R (R = permutation tensor) costs 128 KB CB.
- Early `return` is unsupported in atom bodies; guard whole-core roles
  with `if`.
- Future: gather QKV per-head fanout via fp8 weights stream cuts W CB
  in half; mlp pad lives at 576 (Nt=18) for band divisibility.
- Stage B pipework: one mcast_block call serves a whole multi-pipe net (one
  block per src pipe lands on every dst); per-head O accumulation is gated
  with index conditionals (`if row_c == qh % 2`); unused recv blocks must
  be consumed (zero-weight matmul) — same rule as unused waits.
- Composition direction (user): every op grows an inlinable DFB-boundary
  core; wrappers carry tests/benches; layer atoms are PipeNets + cores
  only. Monolithic atoms are scaffolding to retire.
- Core-first refactor landed: rmsnorm_core (caller feeds x chunks TWICE:
  sum-sq pass then scale pass), gemv_band_core, rope_core, kv_patch_core;
  flash cores already existed. Stage-A attn atom = DFB decls + PipeNets +
  core calls, PCC unchanged.
- Double-produce gotcha: a core that reserves its out DFB means the
  caller must NOT reserve it too; branch-local reserves only.
- Cores callable under runtime `if` must take every DFB as a param; flash
  is now make_flash_window_core (caller stages K/V/mask chunks; chunk
  copies may sit under conditionals).
- DFB index cap: stage-B attn atom needs 50 indices vs 32; finalize reuse
  merges nothing because role-guarded bodies collapse to one top-level
  interval. Finer path-based coloring is UNSOUND: three RISCs drift across
  inlined-atom phases with shared CB counters/pointers (issue 679; reset op
  is the durable fix). Direction: declare scratch in the lowest-level core,
  prefer compiler-inserted temps, manually share same-shape DFBs across
  role-disjoint phases; cut the atom only if forced.
- Remote pitfall: never git-checkout compiler files on the remote; its HEAD
  is stale vs synced working files (cost: stage A head-0 pcc -0.07
  deterministic until local-HEAD files were resent; reset/cache innocent).
- ready net wait-for cycle: q recv(ready) -> obc mcast vs kv send(ready) -> orecv wait; decompose next
- Cycle fix next: scheduler hint = post receive before dependent send. Hoist orecv dst publication ahead of the ready recv on q cores (mcast_block currently pubs after the role if), or carry the kv-ready token in qkv_red instead of a separate net.
- Root cause of SPSC errors under the cap: the DM scheduler splits copies on
  one DFB across DM0/DM1 (fkv k/v copies, line refs in /tmp/diag.log). Fix:
  pin all DM copies of a cb_index to one DM thread; that re-legalizes fkv
  and the node-disjoint shares, putting the atom under 32 directly.
- SPSC ordering: ttl-verify-dfb-spsc REQUIRES finalized cb indices (assert in
  record); it runs post-merge by design. The cb7/12/23 multi-producer errors
  at <=32 indicate fine single-thread merges crossing producer funcs - check
  class keys vs trisc/ncrisc/brisc bodies (atom_split deepcopies bodies, so
  matching reserve ops exist in all three threads; finalize sees 3 funcs!).
  Likely fix: classes use bind func sets and reserves coalesce per thread;
  pop dead-code copies of reserves per body cause cross-thread keys. Look at
  finalize recordRole: reserves dropped on other threads after split? Verify
  inlined sq reserve survives in DM bodies; if so, prune dead reserves
  before finalize.
- TTL_DFB_TRACE=1 dumps slot/pinned layout. Attn floor today: 12 pinned
  (pipes + multi-thread out/band/pos) + 19 slots + ~2 compiler temps = 33.
  Cut path: out is pinned (consumers compute + DM); copy band back via a
  compute store frees one index; further squeeze: merge cb46/cb29 sized
  alike across class boundary needs reset op (#679).
- Cap decision: attn floor is 33 (11 pins + 22 slots, rest cross-thread).
  Per guidance, CB-count-bound cuts are sanctioned: split O projection into
  its own atom (drain o/m/l, drop wo/op/orecv/ostage) until #679 reset op.
- MILESTONE: full pre-AR attn atom (O cut to separate GEMV) passes ALL
  compiler stages at worker_l1_size=1_362_000 (32 indices, budget, program
  size, verifiers; no relax flag). Runtime hangs in the flash phase: 60s
  timeout, kill + reset recovers. Debug next via ttlang hang debug flow:
  signposts per phase on q cores; suspect fl/fmask bc1 recurrence or the
  q tok gate. Known-good single-buffer hang patterns in
  ~/Downloads/ttl_blaze_examples.md (private; do not cite here).
- Hang diagnosis (TT_METAL_WATCHER): worker cores all idle/done (smsg DDDD),
  only dispatch cores wait (UAPW/NWBD) - host-side completion hang, not a CB
  deadlock. Suspect 9x2 grid done-signal counting or read-back. Next: drain
  q rows from o_heads on host, compare against grid signaling; check
  generated runner waits per-core done count.
- Hang isolation: standalone (no pytest) also hangs; watcher shows ALL cores
  idle, no kernels launched (k_ids 0) => host stalls pre-dispatch with
  workers never started. Stage A under worker_l1_size=1_362_000 hits a
  TRACE error (hg reserve "expression does not produce a value") - hg copy
  hoist leak into stage A; fix that first, then test stage A at reduced L1
  to split worker_l1 vs atom-content as hang cause.
- OPEN: stage A trace fails 'cannot call .reserve() on hg_cb: expression
  does not produce a value' at attn_atom.py:88 since rmsnorm/rope local
  scratch (inliner hoist). Cap math says 32 ok; suspect tracer name
  resolution for hg after hoisted insertion (alias or _lift_setup order).
  Repro: pytest test_gemma_attn_atom. Hoist insertion now after last setup
  (atom_inline). Full-atom hang debug queued: leave process alive, run
  third-party/tt-metal/tools/triage triage.py --run=dump_callstacks
  --run=dump_fast_dispatch --llm-output; kill+reset ONLY after.
- Stage A trace error root cause: hg_cb decl was simply missing in stage A
  (lost during gamma-fold edits). With decl restored, stage A ran but PCC
  fell to ~0.003: interleaving x+gamma into one nx stream feeds two copy
  call sites on a single DFB whose copies the splitter may place on
  different DM threads. Reverted gamma fold (dedicated g_cb); stage A back
  to PCC 1.0; hx_cb dropped (head staged through head_cb bc3) keeps 32.
- Full atom now LAUNCHES (was pre-dispatch); kernels run on cols 1-8 row 0
  while norm + row1 finish. Triage signature: all trisc1 at one wait PC.
  Bisects (each ~3 min, reset between): ready sends moved BEFORE the kv
  patch chain => completes; ANY DM copy before pipe_send on kv cores
  (even a 1-tile pos read) => hangs. Patch loop, copy-back target,
  read_index, q recv pre-grant all irrelevant. Suspect pipe scheduler:
  per-net send batches block on credits/waits that never arrive when an
  intervening DM stream op delays the send leg. /tmp/repro_copy_before_send
  (2-core) and /tmp/repro_ready_mcast (5-core, dual mcast nets + double
  recv + read_index) both PASS, so trigger needs stage A's band/qkv nets.
- tools/cb_counts.py: per-CB wait/pop/reserve/push counts per thread from
  emitted C++ (literal loops folded). cb5 = pipe ready sends (per-dst
  unicast + sem inc + wait_min on acks before push); unbalanced rows are
  role-branch artifacts; branch-aware version is the next tool step.
- Two-atom cut (sanctioned, dispatch count is a tunable): attn split into
  make_attn_patch_atom (stage A + KV ring patch, (9,2)) and make_flash_atom
  ((4,1)); dispatch order replaces the kv->q ready handshake whose pipes
  deadlock today. OPEN: even with no handshake the patch atom hangs in
  HOST land - workers idle, no kernels, host blocks in q_heads to_torch
  readback; variants flip between hang / abort / segfault at close.
  Sanity-checked card with kv_append + stage A (pass) - device fine.
  Smells like generic_op launch-side memory corruption; repro is
  /tmp/diag_split.py + bisect logs /tmp/ds*.log on card1.
- Core utilization (planned, post-bringup per plan P5/P2 audits): today's
  attn step uses 18 cores patch + 4 cores flash of 130. Path up: 32k flash
  seq-splits across 8 cores per head (64 cores), flash combine over pipes;
  RoPE/QK-norm on both grid rows; norm sharded over 4 cols (#671 block
  subviews remove the xn DRAM stage); kv patch overlaps q rows on grid
  row 1; full layer atom adds MLP + experts on remaining cores.
- Cache read primitives needed: ttl.read_index (landed) covers runtime
  positions; missing for utilization: block subview slices (#671) for
  norm/QKV sharding; cross-chunk flash combine pipes (live merge instead
  of one-core windows); a CB reset op (#679) before any cross-thread
  slot sharing tightens the index budget further.
- Launch-hang fault isolation (diag_args.py, padded stage A): 13 io tensors
  PASS; +kv pos copy any position PASS; +read_index+band fill+copy-back
  PASS; +patch core loop PASS. Patch atom STILL hangs with kv chain
  hoisted to body top level (same structure as the passing diag). Deltas
  left: kv chain reads all four caches + q_heads readback after write.
  Card sanity (kv_append, stage A) passes between runs.
- tools/cb_blocks.py: per-top-level-branch CB credit counts (loops folded).
  Patch atom imbalances: cb0 trisc waits 4 vs ncrisc pushes 3; cb4 (head
  staging) waits 6 vs reserves 5. The head_cb bc3 staging (hd, h=wait,
  h2.store(h), h3.store(h), core waits x2) emits 1 extra wait without a
  push; cb0 producer short one push. Validates the unbalanced-CB theory
  for the launch hang. Fix: restage head feed (e.g. hd + h2 only with
  norm core consuming hd, h2; or producer pushes 4).
- ROOT CAUSE of flash hang: merged K/V fkv buffer (cap squeeze). Window
  core wait order per chunk is k, mask, v; merged stream queues v before
  mask, vd reserve waits k pop, mask never lands, core waits mask = credit
  cycle deadlock. Split fk/fv fixes it; merged variant needs kd, md, vd
  order. NOT a compiler bug. Patch atom host-stall remains open (separate).
- MILESTONE: full pre-AR sliding attention E2E ON HW via dispatch chain
  heads atom (PCC 1.0) -> kv_append x4 (k_row tile param) -> flash atom
  (4,1) split fk/fv -> host O matmul; PCC > 0.98 in 5.5s. Dispatch count
  is the accepted bring-up cut; refusing only untested compositions.
- O projection on device: flash atom writes one wide (1, 4Dt) row, gemv
  (8,2) bn=11 projects 1024 -> 2816 as the next dispatch. Attention now
  has ZERO host compute: heads + 4x kv_append + flash + gemv, PCC > 0.98.
  Fused retry with kd, md, vd staging fix moved the fused atom from device
  deadlock to the open host-launch stall (workers never start), so the
  bc1 cycle was the fused device bug; host stall is separate, shelved.
