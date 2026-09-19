# Compiler suite on tt-emule

This is the September 19, 2026 test snapshot for
[`kostas/tt-lang-sim-emule-consolidated`](https://github.com/tenstorrent/tt-lang/tree/kostas/tt-lang-sim-emule-consolidated),
the branch proposed by [PR #1063](https://github.com/tenstorrent/tt-lang/pull/1063).
It used the exact compiler, tt-emule, tt-metal, image, and P150 target recorded
in `config/tt-lang-emule-stack.json`.

## What was run

The complete compiler suite was started with:

```bash
cmake --build build --target check-ttlang-all
```

Because that aggregate target stops after a failing group, the later end-to-end
and Python lit groups were then run separately. The combined result was:

| Group | Result |
| --- | ---: |
| MLIR | 528 passed |
| Python bindings | 5 passed |
| Packaging | 199 passed |
| Compiler and device pytest | 4,019 passed, 161 failed, 64 skipped, 8 expected failures |
| End-to-end | 859 passed, 34 expected failures |
| Python lit | 132 passed, 4 failed, 1 unsupported |
| **Total** | **5,742 passed, 165 failed, 65 skipped/unsupported, 42 expected failures** |

## What failed

The 165 failures reduce to five emulator/runtime gaps:

- **132 dynamic-buffer tests:** generated RISC-V inline assembly reaches the
  x86 host JIT and is rejected by Clang.
- **24 RMSNorm tests:** the emulated JIT environment does not provide
  `ckernel_sfpu_rsqrt.h`.
- **2 dynamic-buffer reuse tests:** execution completes, but 992 of 2,048
  output values remain zero.
- **3 collective tests:** one multicast loopback and two tree reductions
  produce incorrect output.
- **4 DPRINT lit tests:** kernels execute, but the expected device-side DPRINT
  payload is not emitted.

These are grouped root causes rather than 165 independent bugs. The compiler,
bindings, packaging, and end-to-end groups all passed completely.

## Reproduce or narrow the run

After installing the supported environment as described in
[Getting started with compiler-backed emulation](simulator-getting-started.md),
activate that compiler build environment and use the normal test tools:

```bash
cmake --build build --target check-ttlang-all
pytest -v test/python
pytest -v test/me2e
llvm-lit -v build/test/python
```

Normal pytest and lit selectors can be used to reproduce one file or case; no
separate `tt-lang-sim` test interface is required.
