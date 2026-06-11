# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Emit the generated metal kernels for each subblock shape (no device run).

For every subblock of a kind (add | matmul) this compiles with that forced
subblock and copies the generated **compute** kernel C++ to
``<out>/compute_<sR>x<sC>.cpp`` so you can diff the codegen across subblock
shapes. The reader/writer are the same for every shape and are copied once.

Uses TTLANG_COMPILE_ONLY=1, so it only *generates* the kernels (tt-lang writes
each thread's C++ to /tmp/<user>/ and prints its path) -- no kernel execution and
no profiler. A device is still opened to build the input tensors.

    python3 -m benchmarks.cycles.subblock.emit --kind add  [out_dir]
    python3 -m benchmarks.cycles.subblock.emit --kind matmul
"""

import os

# Generate kernels without running them on device.
os.environ["TTLANG_COMPILE_ONLY"] = "1"

import argparse
import contextlib
import io
import re
import shutil
from pathlib import Path

import ttnn

from .kinds import DST_FULL_SYNC_EN, KINDS

# tt-lang prints one line per generated kernel: "=== <thread> kernel written to <path> ==="
_KERNEL_LINE = re.compile(r"=== (\w+) kernel written to (\S+\.cpp) ===")


def _kernel_paths(stdout_text):
    """{thread: path} parsed from a compile's stdout."""
    return {m.group(1): m.group(2) for m in _KERNEL_LINE.finditer(stdout_text)}


def emit(kind_name, out_dir, dst_full_sync_en=DST_FULL_SYNC_EN):
    kind = KINDS[kind_name]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = ttnn.open_device(device_id=0)
    try:
        tensors = kind.make_tensors(device)
        wrote_dm = False
        for subblock in kind.subblocks():
            label = "auto" if subblock is None else "x".join(str(s) for s in subblock)
            op = kind.make_op(subblock, dst_full_sync_en)
            # Capture the compile's stdout (suppresses the big kernel-source dump
            # and gives the exact generated-kernel paths for this config).
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    op(*tensors)  # COMPILE_ONLY: generates + writes kernels, no run
            except Exception as e:
                from .sweep import _error_summary
                print(f"{kind.name} subblock {label}: invalid "
                      f"(compiler rejected: {_error_summary(e)})", flush=True)
                continue
            paths = _kernel_paths(buf.getvalue())

            if "compute" in paths:
                dst = out / f"compute_{label}.cpp"
                shutil.copy(paths["compute"], dst)
                print(f"{kind.name} subblock {label} -> {dst}", flush=True)
            else:
                print(f"{kind.name} subblock {label}: no compute kernel path found", flush=True)

            # reader/writer are identical across subblocks; copy them once.
            if not wrote_dm:
                for thread in ("read", "write"):
                    if thread in paths:
                        shutil.copy(paths[thread], out / f"{thread}.cpp")
                wrote_dm = True
    finally:
        ttnn.close_device(device)

    print(f"\n{kind.name} kernels emitted to {out}/", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="emit generated kernels per subblock (add | matmul)")
    ap.add_argument("--kind", choices=sorted(KINDS), default="add")
    ap.add_argument("out_dir", nargs="?", default=None)
    args = ap.parse_args(argv)
    out = args.out_dir or str(Path(__file__).parent / "out" / "kernels" / args.kind)
    emit(args.kind, out)


if __name__ == "__main__":
    main()
