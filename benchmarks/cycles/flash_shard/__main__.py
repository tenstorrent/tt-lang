# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Flash-shard cycle A/B: run both variants and print the ratio.

TT_METAL_DEVICE_PROFILER=1 python3 -m benchmarks.cycles.flash_shard
"""

from . import metal_sdpa, shapes, ttl_shard


def main():
    n = shapes.N_CHUNKS
    print(f"=== flash-shard single-core cycle A/B  (N_CHUNKS={n}) ===", flush=True)

    ttl = ttl_shard.run(n)
    shapes.print_result("flash_shard (ttl)", ttl, n)

    metal = metal_sdpa.run(n)
    shapes.print_result("compute_sdpa_chunk (metal)", metal, n)

    ratio = ttl["cycles"] / metal["cycles"]
    print(
        f"\nttl / metal: {ratio:.2f}x slower  "
        f"({ttl['us'] / n:.2f} vs {metal['us'] / n:.2f} us/chunk)",
        flush=True,
    )


if __name__ == "__main__":
    main()
