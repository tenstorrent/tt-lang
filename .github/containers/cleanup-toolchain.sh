#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Clean up unnecessary files from tt-mlir toolchain to reduce Docker image size
# Keep only what's needed for running tt-lang

set -e

TOOLCHAIN_DIR="${1:?Usage: $0 <toolchain-dir>}"

echo "Cleaning up toolchain at: $TOOLCHAIN_DIR"

# Remove duplicate lib64 directory in venv (replace with symlink to lib)
if [ -d "$TOOLCHAIN_DIR/venv/lib64" ] && [ -d "$TOOLCHAIN_DIR/venv/lib" ]; then
    echo "Removing duplicate venv/lib64 directory"
    rm -rf "$TOOLCHAIN_DIR/venv/lib64"
    ln -s lib "$TOOLCHAIN_DIR/venv/lib64"
fi

# Remove unnecessary LLVM/MLIR binaries
# Keep: FileCheck, llvm-lit, *tblgen*, not, count, split-file (lit utilities)
# Keep: ttmlir-*, ttlang-* (project tools)
# Keep: mlir-opt, mlir-translate, *-lsp-server, mlir-tblgen (useful MLIR tools)
cd "$TOOLCHAIN_DIR/bin"
rm -f llc lli lli-child-target opt bugpoint \
      llvm-PerfectShuffle llvm-addr2line llvm-ar llvm-as llvm-bcanalyzer \
      llvm-bitcode-strip llvm-c-test llvm-cat llvm-cfi-verify llvm-cgdata \
      llvm-config llvm-cov llvm-ctxprof-util llvm-cvtres llvm-cxxdump \
      llvm-cxxfilt llvm-cxxmap llvm-debuginfo-analyzer llvm-debuginfod \
      llvm-debuginfod-find llvm-diff llvm-dis llvm-dlltool llvm-dwarfdump \
      llvm-dwarfutil llvm-dwp llvm-exegesis llvm-extract llvm-gsymutil \
      llvm-ifs llvm-install-name-tool llvm-ir2vec llvm-jitlink \
      llvm-jitlink-executor llvm-lib llvm-libtool-darwin llvm-link \
      llvm-lipo llvm-lto llvm-lto2 llvm-mc llvm-mca llvm-ml llvm-ml64 \
      llvm-modextract llvm-mt llvm-nm llvm-objcopy llvm-objdump \
      llvm-offload-wrapper llvm-opt-report llvm-otool llvm-pdbutil \
      llvm-profdata llvm-profgen llvm-ranlib llvm-rc llvm-readelf \
      llvm-readobj llvm-readtapi llvm-reduce llvm-remarkutil llvm-rtdyld \
      llvm-sim llvm-size llvm-split llvm-stress llvm-strings llvm-strip \
      llvm-symbolizer llvm-test-mustache-spec llvm-tli-checker \
      llvm-undname llvm-windres llvm-xray \
      mlir-cat mlir-linalg-ods-yaml-gen \
      mlir-minimal-opt mlir-minimal-opt-canonicalize mlir-pdll \
      mlir-query mlir-reduce mlir-rewrite mlir-runner mlir-transform-opt \
      dsymutil sancov sanstats obj2yaml yaml2obj verify-uselistorder \
      yaml-bench flatc reduce-chunk-list run-clang-tidy.py 2>/dev/null || true

echo "Toolchain cleanup complete"
