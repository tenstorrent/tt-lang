# Issue 1027 contiguous tensor/DFB copy handoff

Status date: 2026-09-28

Issue: https://github.com/tenstorrent/tt-lang/issues/1027

## Repository state

- Repository on the original machine: `/home/jackzhang/tt-lang`
- Branch: `jackzhang/coalesce-tensor-dfb-copies`
- Starting revision: `9d3e176c39b14e7189ac5848b80f1564339d3acb`
  (`origin/main` when the work began)
- All changes described here are committed on the branch. After the branch is
  pushed, resume on another machine with:

  ```sh
  git fetch origin
  git switch --track origin/jackzhang/coalesce-tensor-dfb-copies
  ```

## What is implemented

The TTL-to-TTKernel copy lowering now coalesces adjacent pages in the innermost
row of eligible tensor-to-DFB and DFB-to-tensor copies. The initial proof is
intentionally conservative:

- The tensor buffer must be L1 or L1 Small.
- The tensor memory layout must be `single_bank` or `height_sharded`.
- Coalescing stays within one innermost row, so it does not cross a
  height-shard/bank boundary.
- A row is split at the target NoC maximum burst size (16 KiB on Blackhole,
  8 KiB for the conservative/default target).
- Single-page bursts retain the existing tile operation.
- DRAM/interleaved and otherwise unproven layouts retain the original
  page-by-page lowering.

For the K3 MLA-shaped compact row in the issue, 224 pages of 64 bytes now lower
to one 14,336-byte transfer in each direction instead of 224 page transfers.

Two TTKernel operations carry the explicit contiguous byte count through the
pipeline:

- `ttkernel.noc_async_read_tensor`
- `ttkernel.noc_async_write_tensor`

They translate to the TensorAccessor overloads of `noc.async_read` and
`noc.async_write`, using the first page ID plus the explicit byte count. The
read-barrier canonicalization also recognizes the new read operation as a NoC
command.

## Changed files

- `include/ttlang/Dialect/TTKernel/IR/TTKernelOps.td`
- `lib/Dialect/TTKernel/IR/TTKernelOps.cpp`
- `lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp`
- `lib/Conversion/TTKernelToEmitC/TTKernelToEmitC.cpp`
- `test/ttlang/Conversion/TTLToTTKernel/coalesce_tensor_cb_copy.mlir`
- `test/ttlang/Translate/TTLToCpp/coalesce_tensor_cb_copy.mlir`

## Build notes for the original machine

The machine did not have the expected prebuilt tt-lang toolchain or an MLIR
development package, so the pinned LLVM/MLIR and tt-metal submodules were built
under `build/`. CMake 4.0.2 needed its executable-format cache corrected to
`ELF`; Clang 20 with libstdc++ 12 also needed an unrelated deprecation warning
suppressed because the project enables `-Werror`.

The successful configuration was:

```sh
cmake -G Ninja -B build \
  -DCMAKE_EXECUTABLE_FORMAT:STRING=ELF \
  -DCMAKE_CXX_FLAGS:STRING=-Wno-deprecated-declarations \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang-20 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-20 \
  -DTTLANG_USE_TOOLCHAIN=OFF
cmake --build build --target ttlang-opt ttlang-translate -j 24
```

The repository-pinned clang-format hook passes on the modified C++ files.

## Validation completed

The following passed on `bh-lb-120-a07u24`:

- Build of `ttlang-opt` and `ttlang-translate`.
- New TTL-to-TTKernel regression covering:
  - a 224-page Blackhole row coalesced to 14,336 bytes;
  - both read and write directions;
  - a 300-page row split into 16,384-byte and 2,816-byte bursts;
  - interleaved DRAM falling back to tile-by-tile copies.
- New end-to-end TTL-to-C++ regression for both byte-counted read and write.
- Existing `rank_reducing_copy.mlir` checks, including its address checks.
- Existing `dma_single_core.mlir` lowering checks.
- `git diff --check`.

`llvm-lit` itself could not load `test/lit.cfg.py` because the optional Python
extension `ttl._mlir_libs._ttlang` was not built. The exact compiler and
FileCheck pipelines from the tests were therefore run directly and passed.

## Remaining work

- Run the broader CI/lit suite in a standard tt-lang developer environment.
- Run an actual K3 model workload from `tt-lang-ops-and-models` against this
  compiler branch and confirm the generated kernels and numerical results.
- Measure the affected MLA layers to confirm the expected reduction from 224
  page commands to one bulk command is visible in runtime performance.
- Consider extending the proof to additional layouts only when bank identity,
  address progression, and boundary constraints can be established.

The optimization belongs in `tt-lang`; the K3 repository is the right place
for model-level reproduction and performance validation, not for the generic
lowering itself.
