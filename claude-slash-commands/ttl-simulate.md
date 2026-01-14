---
description: Run functional simulation and suggest improvements based on dynamic analysis
argument-hint: <kernel-file> [test-inputs]
---

## Task

Execute a TT-Lang kernel in the functional simulator and provide improvement suggestions based on runtime behavior.

## Input

$ARGUMENTS

## Process

1. **Setup simulation environment**
   - Configure the functional simulator
   - Prepare test inputs (use provided or generate defaults)
   - Enable tracing and debugging output

2. **Run the simulation**
   - Execute the kernel in the simulator
   - Capture execution trace
   - Record data flow and memory access patterns
   - Note any warnings or anomalies

3. **Analyze runtime behavior**
   - **Correctness**: Verify output matches expected results
   - **Data flow**: Track how data moves through the kernel
   - **Memory patterns**: Identify access patterns and potential issues
   - **Synchronization**: Check for race conditions or deadlocks
   - **Resource usage**: Monitor buffer utilization and capacity

4. **Generate improvement suggestions**

   Based on dynamic analysis, suggest improvements for:
   - **Correctness issues**: If outputs don't match, diagnose the bug
   - **Inefficient patterns**: Redundant data movement, poor locality
   - **Buffer sizing**: Over-provisioned or under-utilized buffers
   - **Synchronization**: Unnecessary barriers or missing synchronization
   - **Parallelism opportunities**: Sequential operations that could parallelize

5. **Provide actionable recommendations**
   - Prioritize suggestions by impact
   - Include code snippets showing proposed changes
   - Explain the expected benefit of each change

## Output

Provide:
- Simulation execution summary
- Correctness verification result
- Data flow visualization (if helpful)
- Ranked list of improvement suggestions
- Code changes to implement top suggestions
