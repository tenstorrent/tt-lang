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
