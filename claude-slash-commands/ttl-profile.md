---
description: Profile a TT-Lang kernel and report per-line cycle counts
argument-hint: <kernel-file>
---

## Prerequisites

All tools are installed at `~/.claude/commands/tools/`. Use `--help` on any tool to understand usage. Read `run-test.sh` if you need to understand the flags in detail.

Before doing anything else, run the smoke test to verify your remote setup:
```bash
~/.claude/commands/tools/smoke-test.sh
```
If the smoke test fails, STOP. Do NOT continue. Ask the user to fix their remote setup first.

For context on the TT-Lang programming model and kernel structure, refer to `/ttl-import`.

## Task

Profile a TT-Lang kernel using the auto-profiler. Report per-line cycle counts and performance metrics to the user.

**IMPORTANT: Do NOT attempt to optimize the kernel.** Your job is to profile and report only. If the user wants optimization, they should use `/ttl-optimize`.

## Input

$ARGUMENTS

## Constraint: One Kernel Invocation

The auto-profiler can only profile **one kernel invocation at a time**. Before profiling, read the file and check if there are multiple `@ttl.kernel` calls or if a kernel is called in a loop. If so, comment out extra invocations so only the target kernel runs once. If it's ambiguous which kernel to profile, ask the user.

## Process

### Step 1: Prepare and Run

Read the input file. Ensure only one kernel invocation will execute (see constraint above).

Run with `--auto-profile --hw`:
```bash
~/.claude/commands/tools/run-test.sh --auto-profile --hw /path/to/kernel.py
```

Then read the output log.

### Step 2: Analyze Results

The log contains several sections. Grep for these markers:

- `THREAD SUMMARY` -- Per-thread cycle counts, op counts, and a compute-vs-memory bound analysis with a visual bar
- `NOC PROFILER SUMMARY` -- Grid size, duration, DRAM read/write volumes, effective bandwidth, transfer sizes, barrier counts
- `DFB FLOW GRAPH` -- JSON describing circular buffer producer/consumer relationships and DMA ops
- `PIPE GRAPH` -- Inter-core pipe communication graph (may be empty if no pipes)

### Step 3: Report to User

Summarize:
- What was profiled (kernel name, grid size, tensor shapes)
- Whether the kernel is compute-bound or memory-bound, and by how much
- Per-thread cycle breakdown and hotspots
- DRAM bandwidth utilization and transfer patterns
- Any anomalies (imbalanced cores, unexpected stalls, low BW)

Be specific with numbers.

### Step 4: Provide Full Profiler Command

Give the user a command they can run directly on the server for the full auto-profiler result:
```
TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1 TTLANG_AUTO_PROFILE=1 python <path_to_kernel.py>
```

Do NOT include `TTLANG_PERF_DUMP=1` in this command.

## Output

1. Profile summary with cycle counts and bound analysis
2. Interesting findings called out
3. The server command for full profiler output
4. Reminder: use `/ttl-optimize` to improve performance
