# Claude Skills

> ⚠️ Skills are an experimental feature under active development; skills currently reference in-flight functionality that may not be available such as the matmul operator.

One of the easiest ways to get started with tt-lang is using [Claude Code](https://claude.com/claude-code) and an existing codebase. TT-Lang provides slash commands that guide Claude through kernel translation, testing, profiling, and optimization workflows.

## Example Workflow

```bash
# Clone a model you want to port
git clone https://github.com/karpathy/nanoGPT
cd nanoGPT

# Install TT-Lang slash commands (one-time setup)
cd /path/to/tt-lang/claude-slash-commands
./install.sh

# Open Claude Code in your project
cd /path/to/nanoGPT
claude

# Now type slash to use skills to translate kernels to TT-Lang:
#   /ttl-import model.py    "translate the attention kernel to TT-Lang DSL"
```

## Available Commands

Run `/ttl-help` in Claude Code to see all available commands. Here is a summary:

```
/ttl-import <kernel>
    Translate a CUDA, Triton, or PyTorch kernel to TT-Lang DSL. Analyzes the
    source kernel, maps GPU concepts to Tenstorrent equivalents, and iterates
    on testing until the translated kernel matches the original behavior.

/ttl-export <kernel>
    Export a TT-Lang kernel to TT-Metal C++ code. Runs the compiler pipeline,
    extracts the generated C++, and beautifies it by improving variable names
    and removing unnecessary casts for readable, production-ready output.

/ttl-optimize <kernel>
    Profile a kernel and apply performance optimizations. Identifies bottlenecks,
    suggests improvements like tiling, pipelining, and fusion, then validates
    that optimizations preserve correctness while improving throughput.

/ttl-profile <kernel>
    Run the profiler and display per-line cycle counts. Shows exactly where time
    is spent in the kernel with annotated source, hotspot highlighting, and
    memory vs compute breakdown.

/ttl-bug <reproducer>
    File a bug report for TT-Lang with a reproducer.

/ttl-help
    List all available TT-Lang slash commands with descriptions and examples.
```
