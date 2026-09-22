# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Pinned AGMM cases imported from TT-Metal's model sweep table.

The source tuple uses full K and per-device N.  The four-device runner converts
it to M tiles, K tiles per device, and global N tiles before constructing inputs.
Rows with operation kinds or fused epilogues that this benchmark cannot model
are retained so the sweep reports them explicitly instead of relabeling them.
"""

from dataclasses import dataclass

TT_METAL_SWEEP_REVISION = "0e9d200db976120c129ab0deb13aa3f6d972b723"


@dataclass(frozen=True)
class AGMMCase:
    m_elements: int
    full_k_elements: int
    n_elements_per_device: int
    compute_grid: tuple[int, int]
    use_case: str
    operation_kind: str = "agmm"

    @property
    def case_id(self) -> str:
        return (
            f"{self.m_elements}x{self.full_k_elements}x"
            f"{self.n_elements_per_device}_{self.compute_grid[0]}x"
            f"{self.compute_grid[1]}_{self.operation_kind}_{self.use_case}"
        )

    @property
    def comparison_id(self) -> str:
        return (
            f"{self.m_elements}x{self.full_k_elements}x"
            f"{self.n_elements_per_device}_{self.operation_kind}_{self.use_case}"
        )

    @property
    def m_tiles(self) -> int:
        if self.m_elements % 32:
            raise ValueError(f"{self.case_id}: M must be tile aligned")
        return self.m_elements // 32

    def k_tiles_per_device(self, device_count: int) -> int:
        if self.full_k_elements % (32 * device_count):
            raise ValueError(f"{self.case_id}: full K is not divisible by device count")
        return self.full_k_elements // (32 * device_count)

    @property
    def n_tiles_per_device(self) -> int:
        if self.n_elements_per_device % 32:
            raise ValueError(f"{self.case_id}: per-device N must be tile aligned")
        return self.n_elements_per_device // 32


# (M, K, N, grid_x, grid_y, use_case, operation_kind)
UPSTREAM_AGMM_CASES = (
    AGMMCase(9472, 5120, 3840, (12, 9), "qkv", "agmm"),
    AGMMCase(9472, 5120, 1280, (12, 9), "to_out", "agmm"),
    AGMMCase(9472, 5120, 3456, (12, 9), "ff1_gelu", "agmm"),
    AGMMCase(3072, 5120, 3840, (8, 8), "plain", "agmm"),
    AGMMCase(3072, 5120, 1280, (8, 8), "plain", "agmm"),
    AGMMCase(3072, 5120, 3456, (8, 8), "plain_gelu", "agmm"),
    AGMMCase(4768, 5376, 5376, (12, 9), "qkv", "agmm"),
    AGMMCase(4768, 7168, 1344, (12, 9), "plain", "agmm"),
    AGMMCase(4768, 5376, 7168, (12, 9), "ff1_swiglu", "agmm"),
    AGMMCase(1024, 768, 4608, (12, 9), "plain", "agmm"),
    AGMMCase(1152, 768, 4608, (12, 9), "plain", "agmm"),
    AGMMCase(128, 768, 4608, (12, 9), "plain", "agmm"),
    AGMMCase(4096, 768, 4608, (12, 9), "plain", "agmm"),
    AGMMCase(4224, 768, 4608, (12, 9), "plain", "agmm"),
    AGMMCase(4096, 768, 2304, (12, 9), "qkv", "agmm"),
    AGMMCase(16384, 768, 4608, (12, 9), "plain", "agmm"),
    AGMMCase(16512, 768, 4608, (12, 9), "plain", "agmm"),
    AGMMCase(16384, 768, 2304, (12, 9), "qkv", "agmm"),
    AGMMCase(16384, 6144, 4608, (12, 9), "plain", "agmm"),
    AGMMCase(16512, 6144, 4608, (12, 9), "plain", "agmm"),
    AGMMCase(16384, 6144, 2304, (12, 9), "qkv", "agmm"),
    AGMMCase(16384, 6144, 768, (12, 9), "to_out", "agmm"),
    AGMMCase(4096, 6144, 4608, (12, 9), "plain", "agmm"),
    AGMMCase(4224, 6144, 4608, (12, 9), "plain", "agmm"),
    AGMMCase(4096, 6144, 2304, (12, 9), "qkv", "agmm"),
    AGMMCase(4096, 6144, 768, (12, 9), "to_out", "agmm"),
    AGMMCase(1024, 6144, 4608, (12, 9), "plain", "agmm"),
    AGMMCase(1152, 6144, 4608, (12, 9), "plain", "agmm"),
    AGMMCase(1024, 6144, 2304, (12, 9), "qkv", "agmm"),
    AGMMCase(1152, 6144, 2304, (12, 9), "qkv", "agmm"),
    AGMMCase(1024, 6144, 768, (12, 9), "to_out", "agmm"),
    AGMMCase(128, 6144, 2304, (12, 9), "qkv", "agmm"),
    AGMMCase(128, 6144, 768, (12, 9), "to_out", "agmm"),
    AGMMCase(128, 6144, 4608, (12, 9), "plain", "agmm"),
    AGMMCase(8192, 6144, 9216, (12, 9), "plain", "agmm"),
    AGMMCase(8256, 6144, 9216, (12, 9), "plain", "agmm"),
    AGMMCase(8192, 6144, 4608, (12, 9), "qkv", "agmm"),
    AGMMCase(8192, 6144, 1536, (12, 9), "to_out", "agmm"),
    AGMMCase(2048, 6144, 9216, (12, 9), "plain", "agmm"),
    AGMMCase(2112, 6144, 9216, (12, 9), "plain", "agmm"),
    AGMMCase(2048, 6144, 4608, (12, 9), "qkv", "agmm"),
    AGMMCase(576, 6144, 9216, (12, 9), "plain", "agmm"),
    AGMMCase(512, 6144, 9216, (12, 9), "plain", "agmm"),
    AGMMCase(64, 6144, 9216, (12, 9), "plain", "agmm"),
    AGMMCase(512, 6144, 4608, (12, 9), "qkv", "agmm"),
    AGMMCase(64, 6144, 4608, (12, 9), "qkv", "agmm"),
    AGMMCase(2048, 6144, 1536, (12, 9), "to_out", "agmm"),
    AGMMCase(512, 6144, 1536, (12, 9), "to_out", "agmm"),
    AGMMCase(64, 6144, 1536, (12, 9), "to_out", "agmm"),
    AGMMCase(1152, 6144, 4608, (12, 8), "plain", "agmm"),
    AGMMCase(1152, 6144, 4608, (12, 7), "plain", "agmm"),
    AGMMCase(1152, 6144, 4608, (12, 6), "plain", "agmm"),
    AGMMCase(1152, 6144, 4608, (12, 5), "plain", "agmm"),
    AGMMCase(1024, 6144, 4608, (12, 8), "plain", "agmm"),
    AGMMCase(1024, 6144, 4608, (12, 7), "plain", "agmm"),
    AGMMCase(1024, 6144, 4608, (12, 6), "plain", "agmm"),
    AGMMCase(1024, 6144, 4608, (12, 5), "plain", "agmm"),
    AGMMCase(128, 6144, 4608, (12, 8), "plain", "agmm"),
    AGMMCase(128, 6144, 4608, (12, 7), "plain", "agmm"),
    AGMMCase(128, 6144, 4608, (12, 6), "plain", "agmm"),
    AGMMCase(128, 6144, 4608, (12, 5), "plain", "agmm"),
    AGMMCase(16384, 6144, 2304, (12, 8), "qkv", "agmm"),
    AGMMCase(16384, 6144, 2304, (12, 7), "qkv", "agmm"),
    AGMMCase(16384, 6144, 2304, (12, 6), "qkv", "agmm"),
    AGMMCase(16384, 6144, 2304, (12, 5), "qkv", "agmm"),
    AGMMCase(4096, 6144, 2304, (12, 8), "qkv", "agmm"),
    AGMMCase(4096, 6144, 2304, (12, 7), "qkv", "agmm"),
    AGMMCase(4096, 6144, 2304, (12, 6), "qkv", "agmm"),
    AGMMCase(4096, 6144, 2304, (12, 5), "qkv", "agmm"),
    AGMMCase(1024, 6144, 2304, (12, 8), "qkv", "agmm"),
    AGMMCase(1024, 6144, 2304, (12, 7), "qkv", "agmm"),
    AGMMCase(1024, 6144, 2304, (12, 6), "qkv", "agmm"),
    AGMMCase(1024, 6144, 2304, (12, 5), "qkv", "agmm"),
    AGMMCase(1152, 6144, 2304, (12, 8), "qkv", "agmm"),
    AGMMCase(1152, 6144, 2304, (12, 7), "qkv", "agmm"),
    AGMMCase(1152, 6144, 2304, (12, 6), "qkv", "agmm"),
    AGMMCase(1152, 6144, 2304, (12, 5), "qkv", "agmm"),
    AGMMCase(128, 6144, 2304, (12, 8), "qkv", "agmm"),
    AGMMCase(128, 6144, 2304, (12, 7), "qkv", "agmm"),
    AGMMCase(128, 6144, 2304, (12, 6), "qkv", "agmm"),
    AGMMCase(128, 6144, 2304, (12, 5), "qkv", "agmm"),
    AGMMCase(16384, 6144, 768, (12, 8), "to_out", "agmm"),
    AGMMCase(16384, 6144, 768, (12, 7), "to_out", "agmm"),
    AGMMCase(16384, 6144, 768, (12, 6), "to_out", "agmm"),
    AGMMCase(16384, 6144, 768, (12, 5), "to_out", "agmm"),
    AGMMCase(4096, 6144, 768, (12, 8), "to_out", "agmm"),
    AGMMCase(4096, 6144, 768, (12, 7), "to_out", "agmm"),
    AGMMCase(4096, 6144, 768, (12, 6), "to_out", "agmm"),
    AGMMCase(4096, 6144, 768, (12, 5), "to_out", "agmm"),
    AGMMCase(1024, 6144, 768, (12, 8), "to_out", "agmm"),
    AGMMCase(1024, 6144, 768, (12, 7), "to_out", "agmm"),
    AGMMCase(1024, 6144, 768, (12, 6), "to_out", "agmm"),
    AGMMCase(1024, 6144, 768, (12, 5), "to_out", "agmm"),
    AGMMCase(128, 6144, 768, (12, 8), "to_out", "agmm"),
    AGMMCase(128, 6144, 768, (12, 7), "to_out", "agmm"),
    AGMMCase(128, 6144, 768, (12, 6), "to_out", "agmm"),
    AGMMCase(128, 6144, 768, (12, 5), "to_out", "agmm"),
    AGMMCase(8192, 6144, 4608, (12, 8), "qkv", "agmm"),
    AGMMCase(8192, 6144, 4608, (12, 7), "qkv", "agmm"),
    AGMMCase(8192, 6144, 4608, (12, 6), "qkv", "agmm"),
    AGMMCase(8192, 6144, 4608, (12, 5), "qkv", "agmm"),
    AGMMCase(2048, 6144, 4608, (12, 8), "qkv", "agmm"),
    AGMMCase(2048, 6144, 4608, (12, 7), "qkv", "agmm"),
    AGMMCase(2048, 6144, 4608, (12, 6), "qkv", "agmm"),
    AGMMCase(2048, 6144, 4608, (12, 5), "qkv", "agmm"),
    AGMMCase(512, 6144, 4608, (12, 8), "qkv", "agmm"),
    AGMMCase(512, 6144, 4608, (12, 7), "qkv", "agmm"),
    AGMMCase(512, 6144, 4608, (12, 6), "qkv", "agmm"),
    AGMMCase(512, 6144, 4608, (12, 5), "qkv", "agmm"),
    AGMMCase(64, 6144, 4608, (12, 8), "qkv", "agmm"),
    AGMMCase(64, 6144, 4608, (12, 7), "qkv", "agmm"),
    AGMMCase(64, 6144, 4608, (12, 6), "qkv", "agmm"),
    AGMMCase(64, 6144, 4608, (12, 5), "qkv", "agmm"),
    AGMMCase(8192, 6144, 1536, (12, 8), "to_out", "agmm"),
    AGMMCase(8192, 6144, 1536, (12, 7), "to_out", "agmm"),
    AGMMCase(8192, 6144, 1536, (12, 6), "to_out", "agmm"),
    AGMMCase(8192, 6144, 1536, (12, 5), "to_out", "agmm"),
    AGMMCase(2048, 6144, 1536, (12, 8), "to_out", "agmm"),
    AGMMCase(2048, 6144, 1536, (12, 7), "to_out", "agmm"),
    AGMMCase(2048, 6144, 1536, (12, 6), "to_out", "agmm"),
    AGMMCase(2048, 6144, 1536, (12, 5), "to_out", "agmm"),
    AGMMCase(512, 6144, 1536, (12, 8), "to_out", "agmm"),
    AGMMCase(512, 6144, 1536, (12, 7), "to_out", "agmm"),
    AGMMCase(512, 6144, 1536, (12, 6), "to_out", "agmm"),
    AGMMCase(512, 6144, 1536, (12, 5), "to_out", "agmm"),
    AGMMCase(64, 6144, 1536, (12, 8), "to_out", "agmm"),
    AGMMCase(64, 6144, 1536, (12, 7), "to_out", "agmm"),
    AGMMCase(64, 6144, 1536, (12, 6), "to_out", "agmm"),
    AGMMCase(64, 6144, 1536, (12, 5), "to_out", "agmm"),
    AGMMCase(1152, 24576, 768, (12, 9), "to_out", "agmm"),
    AGMMCase(1152, 24576, 768, (12, 8), "to_out", "agmm"),
    AGMMCase(1024, 24576, 768, (12, 9), "to_out", "agmm"),
    AGMMCase(576, 1536, 9216, (12, 9), "plain", "agmm"),
    AGMMCase(512, 1536, 9216, (12, 9), "plain", "agmm"),
    AGMMCase(64, 1536, 9216, (12, 9), "plain", "agmm"),
    AGMMCase(2048, 1536, 9216, (12, 9), "plain", "agmm"),
    AGMMCase(2112, 1536, 9216, (12, 9), "plain", "agmm"),
    AGMMCase(2048, 1536, 4608, (12, 9), "plain", "agmm"),
    AGMMCase(8192, 1536, 9216, (12, 9), "plain", "agmm"),
    AGMMCase(8256, 1536, 9216, (12, 9), "plain", "agmm"),
    AGMMCase(8192, 1536, 4608, (12, 9), "plain", "agmm"),
    AGMMCase(1152, 6144, 4608, (12, 8), "ff1_swiglu", "agmm"),
    AGMMCase(1024, 6144, 4608, (12, 8), "ff1_swiglu", "agmm"),
    AGMMCase(128, 6144, 4608, (12, 8), "ff1_swiglu", "agmm"),
    AGMMCase(1152, 6144, 4608, (12, 9), "ff1_swiglu", "agmm"),
    AGMMCase(1024, 6144, 4608, (12, 9), "ff1_swiglu", "agmm"),
    AGMMCase(128, 6144, 4608, (12, 9), "ff1_swiglu", "agmm"),
    AGMMCase(1152, 6144, 4608, (12, 7), "ff1_swiglu", "agmm"),
    AGMMCase(1024, 6144, 4608, (12, 7), "ff1_swiglu", "agmm"),
    AGMMCase(128, 6144, 4608, (12, 7), "ff1_swiglu", "agmm"),
    AGMMCase(1152, 6144, 4608, (12, 8), "ff1_swiglu", "sagmm"),
    AGMMCase(1024, 6144, 4608, (12, 8), "ff1_swiglu", "sagmm"),
    AGMMCase(128, 6144, 4608, (12, 8), "ff1_swiglu", "sagmm"),
    AGMMCase(1152, 6144, 4608, (12, 7), "ff1_swiglu", "sagmm"),
    AGMMCase(1024, 6144, 4608, (12, 7), "ff1_swiglu", "sagmm"),
    AGMMCase(128, 6144, 4608, (12, 7), "ff1_swiglu", "sagmm"),
)


COMPARABLE_OPERATION_KINDS = frozenset({"agmm"})
# QKV chunking partitions the same matmul result that TT-Lang returns as one
# tensor.  Addcmul and fused activations require matching TT-Lang epilogues.
COMPARABLE_USE_CASES = frozenset({"plain", "qkv"})
NATIVE_SUPPORTED_USE_CASES = frozenset(
    {"plain", "qkv", "to_out", "ff1_gelu", "plain_gelu", "ff1_swiglu"}
)
