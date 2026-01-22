---
description: List available TT-Lang slash commands
---

## TT-Lang Slash Commands

The following commands are available for TT-Lang kernel development:

| Command | Description |
|---------|-------------|
| `/ttl-import <kernel>` | Translate a CUDA, Triton, or PyTorch kernel to TT-Lang DSL |
| `/ttl-export <kernel>` | Export TT-Lang kernel to TT-Metal C++ code |
| `/ttl-optimize <kernel>` | Profile and optimize kernel performance |
| `/ttl-profile <kernel>` | Run profiler with per-line cycle counts |
| `/ttl-simulate <kernel>` | Run functional simulator and suggest improvements |
| `/ttl-test <kernel>` | Generate test cases and test runner |
| `/ttl-help` | Show this help message |

### Usage Examples

```
/ttl-import my_cuda_kernel.cu
/ttl-export examples/matmul.py
/ttl-optimize my_kernel.py
/ttl-profile my_kernel.py
/ttl-simulate my_kernel.py
/ttl-test my_kernel.py "test edge cases for small matrices"
```

### Typical Workflow

1. **Import**: Start by importing an existing kernel with `/ttl-import`
2. **Test**: Generate tests with `/ttl-test` to validate correctness
3. **Simulate**: Run `/ttl-simulate` to verify behavior and find issues
4. **Profile**: Use `/ttl-profile` to identify performance bottlenecks
5. **Optimize**: Apply `/ttl-optimize` to improve performance
6. **Export**: Generate production C++ code with `/ttl-export`
