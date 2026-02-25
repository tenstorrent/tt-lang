---
description: Profile and optimize a TT-Lang kernel for better performance
argument-hint: <kernel-file>
---

## Prerequisites

All tools are installed at `~/.claude/commands/tools/`. Use `--help` on any tool to understand usage. Read `run-test.sh` if you need to understand flags in detail.

Before doing anything else, run the smoke test:
```bash
~/.claude/commands/tools/smoke-test.sh
```
If it fails, STOP and ask the user to fix their remote setup.

**You MUST read `/ttl-import` for full context on the TT-Lang programming model before making any changes.**

## Tools Available

```bash
~/.claude/commands/tools/run-test.sh /path/to/kernel.py                # Functional simulator
~/.claude/commands/tools/run-test.sh --hw /path/to/kernel.py           # Real hardware
~/.claude/commands/tools/run-test.sh --perf --hw /path/to/kernel.py    # HW + NOC/DFB profiling
~/.claude/commands/tools/run-test.sh --auto-profile --hw /path/to/k.py # Per-line cycle counts
~/.claude/commands/tools/remote-run.sh <command>                       # Run command on remote
```

Use `--perf` as your primary profiling tool. Use `--auto-profile` only when you need per-line cycle breakdown to diagnose a specific bottleneck.

Log is at `/tmp/ttlang_test_output.log` on the remote. Grep for `THREAD SUMMARY`, `PERF SUMMARY`, `DFB FLOW GRAPH`, `PIPE GRAPH`.

## Task

Optimize a TT-Lang kernel for better performance. Establish a baseline, identify bottlenecks, propose a plan, and iterate.

## Input

$ARGUMENTS

## Before You Start: Understand the Real Workload

Before optimizing, ask the user:
- **What data sizes will this kernel run on in production?** The test data may be much smaller than the real workload. Do not over-optimize for test data.
- **Where will this run?** Single chip? Multi-chip? This affects core budget.

Keep real-world constraints in mind throughout. For example, do not move everything to L1 just because the test data fits -- if the real data is larger, you need streaming. Optimizations must hold for the production workload, not just the test case.

## Optimization Targets

Three goals, in priority order:

### 1. Maximize Core Utilization (target: 100%)

The kernel MUST use all available cores. If the kernel runs on `grid=(1, 1)`, it is leaving performance on the table. Partition work across cores using the multicore patterns from `/ttl-import`. Check `PERF SUMMARY` for grid size.

### 2. Reduce DRAM Traffic

Minimize unnecessary DRAM reads and writes. Key strategies:
- Fuse multiple kernels into one (eliminate intermediate DRAM round-trips)
- Stream large tensors through small CBs instead of loading everything at once
- Reuse data already in L1 instead of re-reading from DRAM

Note: if tensors are small enough, moving them to L1 memory space (`ttnn.L1_MEMORY_CONFIG`) avoids DRAM reads entirely, but this only helps when the data actually fits.

Check `PERF SUMMARY` for DRAM read/write volumes and effective bandwidth.

### 3. Increase DFB Block Size

Larger DFB shapes (block sizes) mean fewer DMA transfers and better throughput. Keep increasing `shape=(R, C)` on circular buffers until you run out of L1 (~1.5MB per core). This is often a big win.

## Iteration Flow

### Step 0: Verify Correctness

Run the kernel with `--hw` to confirm it produces correct results. If it fails, STOP and tell the user to fix their kernel before optimizing.

### Step 1: Establish Baseline

Run with `--perf --hw` to get baseline metrics. Record:
- **Wall time** (duration in us from `PERF SUMMARY`) -- this is your ground truth
- Grid size and core count
- DRAM read/write volumes
- Effective bandwidth
- Compute vs memory bound ratio

### Step 2: Identify Bottleneck

From the baseline, determine the primary bottleneck:
- **Underutilized cores**: grid is smaller than the data allows
- **Excess DRAM traffic**: multiple kernel calls, redundant reads, no streaming
- **Large transfer count with small transfer size**: DFB block size too small
- **Compute bound**: heavy math ops, possible to restructure
- **Memory bound**: data movement dominates, possible to overlap or reduce

### Step 3: Propose Plan

Present your optimization plan to the user **before making changes**. Include:
- What the bottleneck is and why
- What you plan to change
- Expected impact on wall time

Wait for user approval.

### Step 4: Implement and Measure

Make ONE change at a time. After each change:

1. Run with `--hw` to verify correctness (compare output to baseline)
2. Run with `--perf --hw` to measure performance
3. Compare wall time against baseline: did it improve? By how much?
4. If it regressed or broke, revert and try a different approach

**Wall time is the metric that matters.** Other metrics (cycles, BW) are diagnostic tools to understand why wall time changed.

Iterate as many times as needed. There is no limit on profiling runs. Keep going until you've exhausted the optimization targets or hit diminishing returns.

### Step 5: Report

Summarize:
- **Wall time: baseline vs final** (the most important number)
- Baseline vs final metrics (cycles, BW, core utilization)
- What changes were made and their individual wall time impact
- Any remaining bottlenecks

## Output

1. Baseline wall time and metrics
2. Final wall time and metrics
3. Wall time delta (speedup)
4. Summary of each change and its impact
