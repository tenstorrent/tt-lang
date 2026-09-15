# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Tile-parallel BLAKE3 using the Tensix SFPU.

Each 32x32 Int32 tile holds one BLAKE3 word across 1024 independent hashes.
Groups of 1024 hashes are stacked and distributed across the full worker
grid. Data-movement kernels stream those tiles; the compute kernel runs
the compression function with SFPU integer add, XOR, and shift.

A second mode brute-forces ASCII passwords of at most 16 bytes by hashing
one candidate per SFPU lane on the full grid.
"""

from __future__ import annotations

import argparse
import os
import string
from pathlib import Path

import numpy as np
import torch
import ttl
import ttnn

BLOCK_BYTES = 64
CHUNK_BYTES = 1024
CV_WORDS = 8
WORD_COUNT = 16
TILE = 32
LANES = TILE * TILE
MAX_INPUT_BYTES = (1 << 31) - 1
MAX_PASSWORD_BYTES = 16
KERNEL_HEADER = os.path.join(os.path.dirname(__file__), "blake3_kernel.hpp")

CHUNK_START = 1
CHUNK_END = 2
PARENT = 4
ROOT = 8

IV = [
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
]

CHARSETS = {
    "digits": string.digits,
    "digits+letters": string.digits + string.ascii_letters,
    "digits+letters+symbols": string.digits
    + string.ascii_letters
    + string.punctuation,
}


def _u32_tile(values) -> torch.Tensor:
    lane_values = np.zeros(LANES, dtype=np.uint32)
    count = min(len(values), LANES)
    lane_values[:count] = np.asarray(values[:count], dtype=np.uint32)
    return torch.from_numpy(lane_values.view(np.int32)).view(TILE, TILE)


def _worker_count(device) -> int:
    grid = device.compute_with_storage_grid_size()
    return int(grid.x) * int(grid.y)


def _pack_word_groups(word_lanes, group_count: int) -> torch.Tensor:
    lane_count = len(word_lanes[0])
    live_groups = min(group_count, max(1, (lane_count + LANES - 1) // LANES))
    tiles = [
        _u32_tile(values[group_start : group_start + LANES])
        for group_start in range(0, live_groups * LANES, LANES)
        for values in word_lanes
    ]
    packed = torch.cat(tiles, dim=0)
    missing_rows = (group_count - live_groups) * len(word_lanes) * TILE
    if missing_rows:
        packed = torch.cat(
            [packed, torch.zeros((missing_rows, TILE), dtype=torch.int32)], dim=0
        )
    return packed


def _to_device_tiles(host: torch.Tensor, device):
    return ttnn.from_torch(
        host,
        dtype=ttnn.int32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _empty_word_tiles(word_count: int, device):
    return _to_device_tiles(
        torch.zeros((word_count * TILE, TILE), dtype=torch.int32), device
    )


def _live_count_tensor(live_groups: int, device):
    return _to_device_tiles(
        torch.full((TILE, TILE), live_groups, dtype=torch.int32), device
    )


@ttl.operation(grid="full", fp32_dest_acc_en=True, dst_full_sync_en=True)
def _compress_tiles(msg_tensor, cv_tensor, meta_tensor, out_tensor, live_tensor):
    count_c = ttl.make_dataflow_buffer_like(live_tensor, shape=(1, 1), block_count=2)
    count_w = ttl.make_dataflow_buffer_like(live_tensor, shape=(1, 1), block_count=2)
    msg_dfb = ttl.make_dataflow_buffer_like(
        msg_tensor, shape=(WORD_COUNT, 1), block_count=2
    )
    cv_dfb = ttl.make_dataflow_buffer_like(
        cv_tensor, shape=(CV_WORDS, 1), block_count=2
    )
    meta_dfb = ttl.make_dataflow_buffer_like(
        meta_tensor, shape=(4, 1), block_count=2
    )
    out_dfb = ttl.make_dataflow_buffer_like(
        out_tensor, shape=(CV_WORDS, 1), block_count=2
    )
    scratch = ttl.make_dfb("int32", shape=(WORD_COUNT, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_index = ttl.node(dims=1)
        with count_c.wait():
            live_groups = ttl.call_extern_func(
                KERNEL_HEADER,
                "ttlang_blake3_peek_read_u32",
                template_args=[ttl.get_dfb_id(count_c)],
                func_args=[count_c],
                result_type=ttl.ScalarType.I32,
            )
        if node_index < live_groups:
            with (
                msg_dfb.wait(),
                cv_dfb.wait(),
                meta_dfb.wait(),
                out_dfb.reserve(),
            ):
                ttl.call_extern_func(
                    KERNEL_HEADER,
                    "ttlang_blake3_compress_tiles",
                    template_args=[
                        ttl.get_dfb_id(msg_dfb),
                        ttl.get_dfb_id(cv_dfb),
                        ttl.get_dfb_id(meta_dfb),
                        ttl.get_dfb_id(out_dfb),
                        ttl.dfb_descriptor(scratch),
                    ],
                    func_args=[msg_dfb, cv_dfb, meta_dfb, out_dfb, scratch],
                    unknown_dfb_access=True,
                )

    @ttl.datamovement()
    def read():
        node_index = ttl.node(dims=1)
        with count_c.reserve() as count_blk:
            ttl.copy(live_tensor[0:1, 0:1], count_blk).wait()
            live_groups = ttl.call_extern_func(
                KERNEL_HEADER,
                "ttlang_blake3_peek_write_u32",
                template_args=[ttl.get_dfb_id(count_c)],
                func_args=[count_c],
                result_type=ttl.ScalarType.I32,
            )
        with count_w.reserve() as count_blk:
            ttl.copy(live_tensor[0:1, 0:1], count_blk).wait()
        if node_index < live_groups:
            msg_row = node_index * WORD_COUNT
            cv_row = node_index * CV_WORDS
            meta_row = node_index * 4
            with msg_dfb.reserve() as msg_blk:
                ttl.copy(
                    msg_tensor[msg_row : msg_row + WORD_COUNT, 0:1], msg_blk
                ).wait()
            with cv_dfb.reserve() as cv_blk:
                ttl.copy(cv_tensor[cv_row : cv_row + CV_WORDS, 0:1], cv_blk).wait()
            with meta_dfb.reserve() as meta_blk:
                ttl.copy(meta_tensor[meta_row : meta_row + 4, 0:1], meta_blk).wait()

    @ttl.datamovement()
    def write():
        node_index = ttl.node(dims=1)
        with count_w.wait():
            live_groups = ttl.call_extern_func(
                KERNEL_HEADER,
                "ttlang_blake3_peek_read_u32",
                template_args=[ttl.get_dfb_id(count_w)],
                func_args=[count_w],
                result_type=ttl.ScalarType.I32,
            )
        if node_index < live_groups:
            out_row = node_index * CV_WORDS
            with out_dfb.wait() as out_blk:
                ttl.copy(
                    out_blk, out_tensor[out_row : out_row + CV_WORDS, 0:1]
                ).wait()


def _chunk_plan(data: bytes) -> list[dict]:
    chunk_count = max(1, (len(data) + CHUNK_BYTES - 1) // CHUNK_BYTES)
    chunks = []
    for chunk_index in range(chunk_count):
        start = chunk_index * CHUNK_BYTES
        chunk = data[start : start + CHUNK_BYTES]
        block_count = max(1, (len(chunk) + BLOCK_BYTES - 1) // BLOCK_BYTES)
        if not chunk:
            block_count = 1
        chunks.append(
            {
                "index": chunk_index,
                "data": chunk,
                "block_count": block_count,
            }
        )
    return chunks


def _message_words(chunk: bytes, block_index: int) -> list[int]:
    start = block_index * BLOCK_BYTES
    block = chunk[start : start + BLOCK_BYTES]
    block = block + bytes(BLOCK_BYTES - len(block))
    return [
        int.from_bytes(block[i : i + 4], "little") for i in range(0, BLOCK_BYTES, 4)
    ]


def _block_meta(
    chunk: dict, block_index: int, chunk_count: int
) -> tuple[int, int, int]:
    data = chunk["data"]
    if block_index >= chunk["block_count"]:
        return 0, 0, 0
    flags = CHUNK_START if block_index == 0 else 0
    last = block_index + 1 == chunk["block_count"]
    if last:
        flags |= CHUNK_END
        if chunk_count == 1:
            flags |= ROOT
        block_len = len(data) - block_index * BLOCK_BYTES if data else 0
    else:
        block_len = BLOCK_BYTES
    return flags, block_len, 0xFFFFFFFF


def _unpack_cv_words(cv_tensor, lane_count: int) -> list[list[int]]:
    host = ttnn.to_torch(cv_tensor).to(torch.int32).cpu().numpy()
    bits = host.view(np.uint32).reshape(-1, TILE, TILE)
    cvs = []
    for lane in range(lane_count):
        group, local = divmod(lane, LANES)
        row, col = divmod(local, TILE)
        base = group * CV_WORDS
        cvs.append([int(bits[base + word, row, col]) for word in range(CV_WORDS)])
    return cvs


def _compress_group(messages, cvs, counters, block_lens, flags, actives, device):
    workers = _worker_count(device)
    lane_count = len(actives)
    wave_lanes = workers * LANES
    results = []
    for wave_start in range(0, max(lane_count, 1), wave_lanes):
        wave_end = min(wave_start + wave_lanes, lane_count)
        wave = slice(wave_start, wave_end)
        msg = _to_device_tiles(
            _pack_word_groups([m[wave] for m in messages], workers), device
        )
        cv = _to_device_tiles(
            _pack_word_groups([c[wave] for c in cvs], workers), device
        )
        meta = _to_device_tiles(
            _pack_word_groups(
                [counters[wave], block_lens[wave], flags[wave], actives[wave]],
                workers,
            ),
            device,
        )
        out = _empty_word_tiles(CV_WORDS * workers, device)
        live_groups = max(1, (wave_end - wave_start + LANES - 1) // LANES)
        _compress_tiles(msg, cv, meta, out, _live_count_tensor(live_groups, device))
        results.extend(_unpack_cv_words(out, wave_end - wave_start))
    return results


def _hash_chunks(data: bytes, device) -> list[list[int]]:
    chunks = _chunk_plan(data)
    chunk_count = len(chunks)
    cvs = [list(IV) for _ in chunks]
    max_blocks = max(chunk["block_count"] for chunk in chunks)
    for block_index in range(max_blocks):
        messages = [
            [
                (
                    _message_words(chunk["data"], block_index)[word]
                    if block_index < chunk["block_count"]
                    else 0
                )
                for chunk in chunks
            ]
            for word in range(WORD_COUNT)
        ]
        cv_words = [[cv[word] for cv in cvs] for word in range(CV_WORDS)]
        counters = [chunk["index"] for chunk in chunks]
        flags = []
        block_lens = []
        actives = []
        for chunk in chunks:
            flag, length, active = _block_meta(chunk, block_index, chunk_count)
            flags.append(flag)
            block_lens.append(length)
            actives.append(active)
        cvs = _compress_group(
            messages, cv_words, counters, block_lens, flags, actives, device
        )
    return cvs


def _hash_parents(child_cvs: list[list[int]], is_root: bool, device) -> list[list[int]]:
    parent_count = (len(child_cvs) + 1) // 2
    parents = [[] for _ in range(parent_count)]
    if len(child_cvs) % 2:
        parents[-1] = child_cvs[-1]

    pairs = [
        (index // 2, child_cvs[index], child_cvs[index + 1])
        for index in range(0, len(child_cvs) - len(child_cvs) % 2, 2)
    ]
    flags = PARENT | (ROOT if is_root else 0)
    if pairs:
        messages = [
            [
                pair[1][word] if word < CV_WORDS else pair[2][word - CV_WORDS]
                for pair in pairs
            ]
            for word in range(WORD_COUNT)
        ]
        updated = _compress_group(
            messages,
            [[word] * len(pairs) for word in IV],
            [0] * len(pairs),
            [BLOCK_BYTES] * len(pairs),
            [flags] * len(pairs),
            [0xFFFFFFFF] * len(pairs),
            device,
        )
        for lane, cv in enumerate(updated):
            parents[pairs[lane][0]] = cv
    return parents


def blake3(data: bytes, device) -> bytes:
    """Return the standard 32-byte unkeyed BLAKE3 digest."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError(f"prototype supports at most {MAX_INPUT_BYTES} input bytes")

    cvs = _hash_chunks(data, device)
    while len(cvs) > 1:
        cvs = _hash_parents(cvs, len(cvs) == 2, device)

    digest = np.array(cvs[0], dtype=np.uint32)
    return digest.tobytes()


def _resolve_charset(charset: str) -> str:
    if charset in CHARSETS:
        return CHARSETS[charset]
    raise ValueError(
        "charset must be one of: " + ", ".join(sorted(CHARSETS))
    )


def _index_to_password(index: int, length: int, charset: str) -> str:
    if length == 0:
        return ""
    chars = []
    remaining = index
    base = len(charset)
    for _ in range(length):
        remaining, digit = divmod(remaining, base)
        chars.append(charset[digit])
    return "".join(reversed(chars))


def _unpack_cv_array(cv_tensor, lane_count: int) -> np.ndarray:
    host = ttnn.to_torch(cv_tensor).to(torch.int32).cpu().numpy()
    tiles = host.view(np.uint32).reshape(-1, CV_WORDS, TILE, TILE)
    lanes = np.arange(lane_count)
    group = lanes // LANES
    local = lanes % LANES
    row = local // TILE
    col = local % TILE
    return tiles[group[:, None], np.arange(CV_WORDS)[None, :], row[:, None], col[:, None]]


def _candidate_messages(start: int, count: int, length: int, charset: str):
    ids = np.arange(start, start + count, dtype=np.uint64)
    block = np.zeros((count, BLOCK_BYTES), dtype=np.uint8)
    if length:
        table = np.frombuffer(charset.encode("ascii"), dtype=np.uint8)
        remaining = ids
        base = np.uint64(len(charset))
        for position in range(length - 1, -1, -1):
            remaining, digit = np.divmod(remaining, base)
            block[:, position] = table[digit]
    words = block.view("<u4").reshape(count, WORD_COUNT)
    messages = [words[:, word] for word in range(WORD_COUNT)]
    iv = [np.full(count, word, dtype=np.uint32) for word in IV]
    zeros = np.zeros(count, dtype=np.uint32)
    block_lens = np.full(count, length, dtype=np.uint32)
    flags = np.full(count, CHUNK_START | CHUNK_END | ROOT, dtype=np.uint32)
    actives = np.full(count, 0xFFFFFFFF, dtype=np.uint32)
    return messages, iv, zeros, block_lens, flags, actives


def _bruteforce_wave(
    target_words: np.ndarray,
    start: int,
    count: int,
    length: int,
    charset: str,
    device,
) -> str | None:
    messages, iv, counters, block_lens, flags, actives = _candidate_messages(
        start, count, length, charset
    )
    workers = _worker_count(device)
    msg = _to_device_tiles(_pack_word_groups(messages, workers), device)
    cv = _to_device_tiles(_pack_word_groups(iv, workers), device)
    meta = _to_device_tiles(
        _pack_word_groups([counters, block_lens, flags, actives], workers),
        device,
    )
    out = _empty_word_tiles(CV_WORDS * workers, device)
    live_groups = max(1, (count + LANES - 1) // LANES)
    _compress_tiles(msg, cv, meta, out, _live_count_tensor(live_groups, device))
    cvs = _unpack_cv_array(out, count)
    matches = np.flatnonzero(np.all(cvs == target_words, axis=1))
    if matches.size == 0:
        return None
    return _index_to_password(start + int(matches[0]), length, charset)


def blake3_bruteforce(
    target: bytes,
    charset: str,
    device,
    *,
    min_length: int = 0,
    max_length: int = MAX_PASSWORD_BYTES,
) -> str | None:
    """Return the first password whose BLAKE3 digest matches ``target``.

    Candidates are ASCII strings from ``charset`` with length in
    ``[min_length, max_length]``. Each 32x32 tile hashes 1024 candidates, and
    tiles are launched across the full worker grid.
    """
    if not isinstance(target, bytes):
        raise TypeError("target must be bytes")
    if len(target) != 32:
        raise ValueError("target must be a 32-byte BLAKE3 digest")
    if min_length < 0 or max_length > MAX_PASSWORD_BYTES or min_length > max_length:
        raise ValueError(
            f"password length must satisfy 0 <= min_length <= max_length <= {MAX_PASSWORD_BYTES}"
        )

    alphabet = _resolve_charset(charset)
    target_words = np.frombuffer(target, dtype="<u4")
    wave_lanes = _worker_count(device) * LANES
    for length in range(min_length, max_length + 1):
        space = 1 if length == 0 else len(alphabet) ** length
        for start in range(0, space, wave_lanes):
            count = min(wave_lanes, space - start)
            found = _bruteforce_wave(
                target_words, start, count, length, alphabet, device
            )
            if found is not None:
                return found
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--text", default="", help="UTF-8 text to hash")
    source.add_argument("--file", help="File to hash")
    source.add_argument("--crack", help="32-byte BLAKE3 digest as hex to brute-force")
    parser.add_argument(
        "--charset",
        choices=sorted(CHARSETS),
        default="digits",
        help="Candidate alphabet for --crack",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=0,
        help="Minimum password length for --crack",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_PASSWORD_BYTES,
        help="Maximum password length for --crack",
    )
    args = parser.parse_args()

    device = ttnn.open_device(device_id=0)
    try:
        if args.crack is not None:
            found = blake3_bruteforce(
                bytes.fromhex(args.crack),
                args.charset,
                device,
                min_length=args.min_length,
                max_length=args.max_length,
            )
            print("not found" if found is None else found)
            return
        data = (
            Path(args.file).read_bytes()
            if args.file is not None
            else args.text.encode("utf-8")
        )
        print(blake3(data, device).hex())
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
