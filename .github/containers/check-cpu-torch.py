# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Verify the CPU-only Python environment packaged in tt-lang containers."""

from importlib import metadata
import re

import torch


def main() -> None:
    if torch.version.cuda is not None or torch.version.hip is not None:
        raise SystemExit("Expected CPU-only PyTorch in the packaged toolchain venv")

    gpu_packages = []
    for distribution in metadata.distributions():
        name = distribution.metadata["Name"]
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized.startswith(("nvidia-", "cuda-")) or normalized == "triton":
            gpu_packages.append(name)
    if gpu_packages:
        raise SystemExit(
            "Unexpected GPU dependencies in the packaged toolchain venv: "
            + ", ".join(sorted(gpu_packages))
        )
    print(f"CPU-only PyTorch {torch.__version__}; no CUDA or Triton packages")


if __name__ == "__main__":
    main()
