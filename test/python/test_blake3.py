# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device tests for the parallel BLAKE3 example."""

from __future__ import annotations

import importlib.util
import struct
import time
from pathlib import Path

import pytest

pytest.importorskip("ttnn", exc_type=ImportError)

EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "examples" / "blake3.py"
SPEC = importlib.util.spec_from_file_location("ttlang_blake3_example", EXAMPLE_PATH)
assert SPEC is not None and SPEC.loader is not None
BLAKE3_EXAMPLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BLAKE3_EXAMPLE)

REPO_ROOT = Path(__file__).resolve().parents[2]
BIN_DIR = REPO_ROOT / "build" / "bin"
PERF_FILES = (
    "ttlang-execution-count-test",
    "ttlang-value-origin-test",
    "ttlang-opt",
)

# hashlib provides BLAKE2, not BLAKE3. This host path is a compact unkeyed
# BLAKE3 so the device kernel can be checked without extra packages.
_MASK = 0xFFFFFFFF
_IV = (
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
)
_PERM = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)
_CHUNK_START = 1
_CHUNK_END = 2
_PARENT = 4
_ROOT = 8
_BLOCK_BYTES = 64
_CHUNK_BYTES = 1024


def _rotr(value: int, amount: int) -> int:
    return ((value >> amount) | (value << (32 - amount))) & _MASK


def _mix(state: list[int], a: int, b: int, c: int, d: int, x: int, y: int) -> None:
    state[a] = (state[a] + state[b] + x) & _MASK
    state[d] = _rotr(state[d] ^ state[a], 16)
    state[c] = (state[c] + state[d]) & _MASK
    state[b] = _rotr(state[b] ^ state[c], 12)
    state[a] = (state[a] + state[b] + y) & _MASK
    state[d] = _rotr(state[d] ^ state[a], 8)
    state[c] = (state[c] + state[d]) & _MASK
    state[b] = _rotr(state[b] ^ state[c], 7)


def _compress(
    cv: list[int], block: list[int], counter: int, block_len: int, flags: int
) -> list[int]:
    state = cv + list(_IV[:4]) + [counter & _MASK, counter >> 32, block_len, flags]
    message = list(block)
    for round_index in range(7):
        _mix(state, 0, 4, 8, 12, message[0], message[1])
        _mix(state, 1, 5, 9, 13, message[2], message[3])
        _mix(state, 2, 6, 10, 14, message[4], message[5])
        _mix(state, 3, 7, 11, 15, message[6], message[7])
        _mix(state, 0, 5, 10, 15, message[8], message[9])
        _mix(state, 1, 6, 11, 12, message[10], message[11])
        _mix(state, 2, 7, 8, 13, message[12], message[13])
        _mix(state, 3, 4, 9, 14, message[14], message[15])
        if round_index != 6:
            message = [message[index] for index in _PERM]
    return [(state[i] ^ state[i + 8]) & _MASK for i in range(8)]


def _words_from_bytes(data: bytes) -> list[int]:
    padded = data + bytes(_BLOCK_BYTES - len(data))
    return list(struct.unpack("<16I", padded))


def _hash_chunk(chunk: bytes, counter: int, is_root: bool) -> list[int]:
    cv = list(_IV)
    block_count = max(1, (len(chunk) + _BLOCK_BYTES - 1) // _BLOCK_BYTES)
    if not chunk:
        block_count = 1
    for block_index in range(block_count):
        start = block_index * _BLOCK_BYTES
        block = chunk[start : start + _BLOCK_BYTES]
        flags = _CHUNK_START if block_index == 0 else 0
        last = block_index + 1 == block_count
        if last:
            flags |= _CHUNK_END
            if is_root:
                flags |= _ROOT
            block_len = len(block)
        else:
            block_len = _BLOCK_BYTES
        cv = _compress(cv, _words_from_bytes(block), counter, block_len, flags)
    return cv


def host_blake3(data: bytes) -> bytes:
    """Standard 32-byte unkeyed BLAKE3 digest computed on the host."""
    chunk_count = max(1, (len(data) + _CHUNK_BYTES - 1) // _CHUNK_BYTES)
    cvs = [
        _hash_chunk(
            data[index * _CHUNK_BYTES : (index + 1) * _CHUNK_BYTES],
            index,
            chunk_count == 1,
        )
        for index in range(chunk_count)
    ]
    while len(cvs) > 1:
        parents = []
        is_root = len(cvs) == 2
        for index in range(0, len(cvs), 2):
            if index + 1 >= len(cvs):
                parents.append(cvs[index])
                continue
            block = cvs[index] + cvs[index + 1]
            parents.append(
                _compress(
                    list(_IV),
                    block,
                    0,
                    _BLOCK_BYTES,
                    _PARENT | (_ROOT if is_root else 0),
                )
            )
        cvs = parents
    return struct.pack("<8I", *cvs[0])


# The official vectors use byte i % 251 and provide 131 output bytes. Only the
# first 32 bytes are listed because this prototype implements the standard
# digest rather than XOF output.
OFFICIAL_VECTORS = [
    (0, "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"),
    (1, "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"),
    (63, "e9bc37a594daad83be9470df7f7b3798297c3d834ce80ba85d6e207627b7db7b"),
    (64, "4eed7141ea4a5cd4b788606bd23f46e212af9cacebacdc7d1f4c6dc7f2511b98"),
    (65, "de1e5fa0be70df6d2be8fffd0e99ceaa8eb6e8c93a63f2d8d1c30ecb6b263dee"),
    (1023, "10108970eeda3eb932baac1428c7a2163b0e924c9a9e25b35bba72b28f70bd11"),
    (1024, "42214739f095a406f3fc83deb889744ac00df831c10daa55189b5d121c855af7"),
    (1025, "d00278ae47eb27b34faecf67b4fe263f82d5412916c1ffd97c8cb7fb814b8444"),
    (2049, "5f4d72f40d7a5f82b15ca2b2e44b1de3c2ef86c426c95c1af0b6879522563030"),
    (6145, "f1323a8631446cc50536a9f705ee5cb619424d46887f3c376c695b70e0f0507f"),
]


@pytest.mark.parametrize(
    ("input_length", "expected"),
    OFFICIAL_VECTORS,
    ids=[f"len-{input_length}" for input_length, _ in OFFICIAL_VECTORS],
)
def test_host_blake3_official_vectors(input_length, expected):
    """The host reference matches the official BLAKE3 digest vectors."""
    data = bytes(index % 251 for index in range(input_length))
    assert host_blake3(data).hex() == expected


@pytest.mark.parametrize(
    ("input_length", "expected"),
    OFFICIAL_VECTORS,
    ids=[f"len-{input_length}" for input_length, _ in OFFICIAL_VECTORS],
)
def test_blake3_official_vectors(device, input_length, expected):
    """Block, chunk, and odd-tree boundaries match the BLAKE3 vectors."""
    data = bytes(index % 251 for index in range(input_length))
    assert BLAKE3_EXAMPLE.blake3(data, device).hex() == expected


def _perf_file(name: str) -> Path:
    path = BIN_DIR / name
    if not path.is_file():
        pytest.skip(f"{path} is not present")
    return path


@pytest.mark.parametrize("filename", PERF_FILES)
def test_blake3_file_perf_against_host(device, filename):
    """Hash a built binary on device and host, then compare digest and time."""
    path = _perf_file(filename)
    data = path.read_bytes()

    host_started = time.perf_counter()
    host_digest = host_blake3(data)
    host_seconds = time.perf_counter() - host_started

    BLAKE3_EXAMPLE.blake3(data, device)
    device_started = time.perf_counter()
    device_digest = BLAKE3_EXAMPLE.blake3(data, device)
    device_seconds = time.perf_counter() - device_started

    assert device_digest == host_digest

    mebibytes = len(data) / (1024 * 1024)
    host_rate = mebibytes / host_seconds if host_seconds else float("inf")
    device_rate = mebibytes / device_seconds if device_seconds else float("inf")
    print(
        f"\nBLAKE3 {path.name}: {mebibytes:.1f} MiB, "
        f"host {host_seconds:.3f}s ({host_rate:.2f} MiB/s), "
        f"tt-lang {device_seconds:.3f}s ({device_rate:.2f} MiB/s)"
    )


def _crack(device, password: str, charset: str, **kwargs) -> str | None:
    digest = host_blake3(password.encode("ascii"))
    return BLAKE3_EXAMPLE.blake3_bruteforce(digest, charset, device, **kwargs)


@pytest.mark.parametrize(
    ("password", "charset", "max_length"),
    [
        ("", "digits", 0),
        ("7", "digits", 1),
        ("42", "digits", 2),
        ("1024", "digits", 4),
        ("a1", "digits+letters", 2),
        ("A!", "digits+letters+symbols", 2),
    ],
    ids=["empty", "digit", "pin", "multi-group", "alnum", "symbol"],
)
def test_blake3_bruteforce_recovers_password(device, password, charset, max_length):
    """Full-grid candidate hashing recovers a known password."""
    assert _crack(device, password, charset, max_length=max_length) == password


def test_blake3_bruteforce_max_length_sixteen(device):
    """A 16-character candidate still occupies a single BLAKE3 block."""
    password = "0" * 16
    assert (
        _crack(
            device,
            password,
            "digits",
            min_length=16,
            max_length=16,
        )
        == password
    )


def test_blake3_bruteforce_missing_password(device):
    """A digest outside the searched alphabet returns no match."""
    digest = host_blake3(b"zz")
    assert (
        BLAKE3_EXAMPLE.blake3_bruteforce(digest, "digits", device, max_length=2)
        is None
    )


def _host_crack_range(target: bytes, start: int, count: int, length: int, charset: str):
    for index in range(start, start + count):
        password = BLAKE3_EXAMPLE._index_to_password(index, length, charset)
        if host_blake3(password.encode("ascii")) == target:
            return password
    return None


def _password_to_index(password: str, charset: str) -> int:
    index = 0
    base = len(charset)
    for symbol in password:
        index = index * base + charset.index(symbol)
    return index


def _candidates_hashed_on_device(index: int, space: int, wave_lanes: int) -> int:
    hashed = 0
    for start in range(0, space, wave_lanes):
        count = min(wave_lanes, space - start)
        hashed += count
        if start <= index < start + count:
            return hashed
    return hashed


@pytest.mark.parametrize(
    ("password", "charset"),
    [
        pytest.param(None, "digits", id="full-grid-digits"),
        pytest.param("42", "digits", id="digits-42"),
        pytest.param("1024", "digits", id="digits-1024"),
        pytest.param("a1", "digits+letters", id="alnum-a1"),
        pytest.param("A!", "digits+letters+symbols", id="symbol-A!"),
        pytest.param("2V&", "digits+letters+symbols", id="symbol-2V&"),
    ],
)
def test_blake3_bruteforce_full_grid_perf_against_host(device, password, charset):
    """Compare host and device brute-force rate for a password and charset.

    ``password=None`` uses the last candidate of four full-grid waves so every
    core is occupied. Add cases as ``(password, charset)`` pairs.
    """
    workers = BLAKE3_EXAMPLE._worker_count(device)
    wave_lanes = workers * BLAKE3_EXAMPLE.LANES
    alphabet = BLAKE3_EXAMPLE.CHARSETS[charset]
    if password is None:
        length = 8
        password = BLAKE3_EXAMPLE._index_to_password(
            4 * wave_lanes - 1, length, alphabet
        )
    else:
        length = len(password)

    index = _password_to_index(password, alphabet)
    space = 1 if length == 0 else len(alphabet) ** length
    host_count = index + 1
    device_count = _candidates_hashed_on_device(index, space, wave_lanes)
    target = host_blake3(password.encode("ascii"))

    BLAKE3_EXAMPLE.blake3(b"", device)

    host_started = time.perf_counter()
    host_found = _host_crack_range(target, 0, host_count, length, alphabet)
    host_seconds = time.perf_counter() - host_started

    device_started = time.perf_counter()
    device_found = BLAKE3_EXAMPLE.blake3_bruteforce(
        target,
        charset,
        device,
        min_length=length,
        max_length=length,
    )
    device_seconds = time.perf_counter() - device_started

    assert host_found == device_found == password

    host_rate = host_count / host_seconds if host_seconds else float("inf")
    device_rate = device_count / device_seconds if device_seconds else float("inf")
    print(
        f"\nBLAKE3 brute-force {password!r} charset={charset}, "
        f"{workers} cores x {BLAKE3_EXAMPLE.LANES} lanes, "
        f"host {host_count} in {host_seconds:.3f}s ({host_rate:.0f} H/s), "
        f"tt-lang {device_count} in {device_seconds:.3f}s ({device_rate:.0f} H/s)"
    )
