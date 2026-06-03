# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Result helpers shared by the benchmarks: correctness check and CSV output."""

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch


def pcc(result, ref) -> float:
    """Pearson correlation between a result tensor and its reference."""
    a = result.flatten().float()
    b = ref.flatten().float()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def write_csv(path, fields: Sequence[str], rows: Iterable[Mapping]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    return p
