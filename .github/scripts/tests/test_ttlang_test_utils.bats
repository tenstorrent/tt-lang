#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for shared tt-lang Python test utilities used by CI runners.

load test_helper

setup() {
    unset TT_VISIBLE_DEVICES
    unset TT_METAL_CACHE
    unset TTLANG_PIN_XDIST_WORKERS_TO_DEVICES
    unset TTLANG_XDIST_TT_METAL_CACHE_ROOT
}

@test "xdist worker pinning assigns one visible chip and cache directory" {
    run env \
        PYTHONPATH="$TTLANG_REPO_ROOT/test" \
        TTLANG_PIN_XDIST_WORKERS_TO_DEVICES=1 \
        PYTEST_XDIST_WORKER=gw3 \
        TTLANG_XDIST_TT_METAL_CACHE_ROOT="$BATS_TEST_TMPDIR/cache" \
        python3 -c 'import os; from ttlang_test_utils import pin_xdist_worker_to_device; pin_xdist_worker_to_device(); print(os.environ["TT_VISIBLE_DEVICES"]); print(os.environ["TT_METAL_CACHE"]); assert os.path.isdir(os.environ["TT_METAL_CACHE"])'

    assert_success
    assert_line "3"
    assert_line "$BATS_TEST_TMPDIR/cache/worker-3"
}

@test "xdist worker pinning preserves an existing visible chip and still isolates cache" {
    run env \
        PYTHONPATH="$TTLANG_REPO_ROOT/test" \
        TTLANG_PIN_XDIST_WORKERS_TO_DEVICES=1 \
        PYTEST_XDIST_WORKER=gw3 \
        TT_VISIBLE_DEVICES=7 \
        TTLANG_XDIST_TT_METAL_CACHE_ROOT="$BATS_TEST_TMPDIR/cache" \
        python3 -c 'import os; from ttlang_test_utils import pin_xdist_worker_to_device; pin_xdist_worker_to_device(); print(os.environ["TT_VISIBLE_DEVICES"]); print(os.environ["TT_METAL_CACHE"])'

    assert_success
    assert_line "7"
    assert_line "$BATS_TEST_TMPDIR/cache/worker-3"
}

@test "xdist worker pinning converts a relative cache root to an absolute path" {
    run env \
        PYTHONPATH="$TTLANG_REPO_ROOT/test" \
        TTLANG_PIN_XDIST_WORKERS_TO_DEVICES=1 \
        PYTEST_XDIST_WORKER=gw2 \
        TTLANG_XDIST_TT_METAL_CACHE_ROOT=relative-cache \
        python3 -c 'import os; from ttlang_test_utils import pin_xdist_worker_to_device; pin_xdist_worker_to_device(); print(os.environ["TT_METAL_CACHE"])'

    assert_success
    assert_line "$PWD/relative-cache/worker-2"
}
