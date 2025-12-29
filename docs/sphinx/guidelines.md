# Guidelines

- Follow LLVM C++ style; keep helpers local and avoid `using namespace` in headers.
- Use PEP 8 with black for Python; prefer type hints and explicit imports.
- Keep dialect design explicit: encode semantics in ops and types, avoid SSA chasing.
- In pattern rewrites, use `notifyMatchFailure` instead of `emitOpError`.
- Add concise comments that explain why when behavior is non-obvious.
- Include negative tests for new diagnostics and place them in `*_invalid.mlir`.
