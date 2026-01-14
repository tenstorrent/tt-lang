---
description: Profile a TT-Lang kernel and report per-line cycle counts
argument-hint: <kernel-file>
---

## Task

Run the auto-profiler on a TT-Lang kernel and display detailed performance metrics.

## Input

$ARGUMENTS

## Process

1. **Setup profiling environment**
   - Enable profiling flags and instrumentation
   - Configure cycle counter collection
   - Set up the profiling output capture

2. **Run the profiler**
   - Execute the kernel with profiling enabled
   - Collect cycle counts per operation
   - Gather memory transfer statistics

3. **Generate per-line report**
   - Map cycle counts back to source lines
   - Calculate percentage of total time per line
   - Identify the top N hotspots

4. **Display results**
   - Show annotated source with cycle counts
   - Highlight critical path operations
   - Display memory bandwidth metrics
   - Show compute vs memory time breakdown

## Output Format

```
Line  Cycles    %     Source
----  ------  -----   ------
  12  1,234   15.2%   dma_read(input_cb, tensor_a)
  15  2,567   31.5%   tile_matmul(a_tile, b_tile, acc)
  18    892   10.9%   dma_write(output_cb, result)
  ...
```

## Summary Statistics

- Total cycles
- Memory-bound vs compute-bound ratio
- Estimated throughput
- Bottleneck analysis
