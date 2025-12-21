# Contributor Guide

## Workflow
- Branch from `main` and keep changes focused.
- Reconfigure after rebases: `cmake -GNinja -B build .`.
- Build and lint before review: `cmake --build build` and `pre-commit run --all-files`.

## Validation
- Compiler tests: `cmake --build build --target check-ttlang`.
- Simulator tests: `pytest test/sim`.
- Targeted MLIR coverage: `llvm-lit test/ttlang/<path>.mlir`.

## Documentation
- Add new user-facing pages under `docs/src` and link them in `SUMMARY.md`.
- Keep contributor-only instructions in this guide or `guidelines.md`.
- Build docs with `cmake --build build --target ttlang-docs` or `make -C docs/sphinx html`.
