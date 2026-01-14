---
description: Profile and optimize a TT-Lang kernel for better performance
argument-hint: <kernel-file>
---

## Task

Analyze kernel performance and apply optimizations to improve throughput.

## Input

$ARGUMENTS

## Process

1. **Profile the kernel**
   - Run the profiler to collect cycle counts
   - Identify hotspots and bottlenecks
   - Measure memory bandwidth utilization
   - Check compute unit utilization

2. **Analyze performance characteristics**
   - Identify memory-bound vs compute-bound regions
   - Check for NOC congestion patterns
   - Look for unnecessary data movement
   - Analyze tile reuse opportunities

3. **Apply optimizations**
   Consider and apply relevant optimizations:
   - **Tiling**: Adjust tile sizes for better cache utilization
   - **Pipelining**: Overlap data movement with compute
   - **Fusion**: Combine operations to reduce memory traffic
   - **Prefetching**: Hide memory latency with early DMA
   - **Parallelism**: Distribute work across more cores
   - **Memory layout**: Optimize data layout for access patterns

4. **Validate optimizations**
   - Re-profile after each change
   - Verify correctness is preserved
   - Quantify the performance improvement

5. **Iterate**
   - Repeat profiling and optimization until:
     - Target performance is achieved
     - No further improvements are found
     - Diminishing returns are observed

## Output

Provide:
- Performance analysis summary
- List of optimizations applied with rationale
- Before/after performance comparison
- The optimized kernel code
- Suggestions for further optimization if applicable
