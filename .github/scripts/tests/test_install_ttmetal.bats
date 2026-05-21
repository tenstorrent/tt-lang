#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for scripts/install-ttmetal.sh.
#
# Focus: the .so binaries (_ttnn.so, _ttnncpp.so) come from $BUILD/ttnn/,
# never from $SRC/ttnn/ttnn/. install-ttmetal.sh used to copy from $SRC,
# which silently installed stale extensions left in the source tree by a
# previous tt-metal build (see the v0.71.0-rc2 uplift debugging session).

load test_helper

# Build a fake tt-metal source tree and a matching build dir.
# Echoes the temp dir holding both ($DIR/src and $DIR/build). The build
# dir always exists (an empty $BUILD is a realistic state when tt-metal
# configure ran but link failed); only the contents differ.
mkfake_ttmetal() {
    local include_so="$1"     # "with-so" or "no-so" (in $BUILD/ttnn/)
    local has_stale_so="$2"   # "stale" or "fresh" (in $SRC/ttnn/ttnn/)
    local dir
    dir=$(mktemp -d "${BATS_TEST_TMPDIR:-/tmp}/ttmetal.XXXXXX")

    mkdir -p "$dir/src/ttnn/ttnn" "$dir/src/ttnn/cpp" "$dir/src/tt_metal/api" \
        "$dir/src/tools/tracy" "$dir/src/tt_metal/pre-compiled/stale"
    echo "from . import _ttnn" > "$dir/src/ttnn/ttnn/__init__.py"
    echo "version = '0.x'" > "$dir/src/ttnn/ttnn/version.py"
    echo "// cpp header" > "$dir/src/ttnn/cpp/placeholder.h"
    echo "// header" > "$dir/src/tt_metal/api/sample.h"
    echo "stale firmware" > "$dir/src/tt_metal/pre-compiled/stale/fw.elf"
    echo "tracy_module" > "$dir/src/tools/tracy/__init__.py"

    if [[ "$has_stale_so" == "stale" ]]; then
        printf '\x7fELF stale ttnn' > "$dir/src/ttnn/ttnn/_ttnn.so"
        printf '\x7fELF stale ttnncpp' > "$dir/src/ttnn/ttnn/_ttnncpp.so"
    fi

    mkdir -p "$dir/build/ttnn" "$dir/build/lib"
    if [[ "$include_so" == "with-so" ]]; then
        printf '\x7fELF fresh ttnn' > "$dir/build/ttnn/_ttnn.so"
        printf '\x7fELF fresh ttnncpp' > "$dir/build/ttnn/_ttnncpp.so"
        printf '\x7fELF libdevice' > "$dir/build/lib/libdevice.so"
    fi

    echo "$dir"
}

run_install() {
    local src="$1"
    local build="$2"
    local install="$3"
    bash "$TTLANG_REPO_ROOT/scripts/install-ttmetal.sh" "$src" "$build" "$install"
}

@test "install copies _ttnn.so from \$BUILD, not from \$SRC stale copy" {
    local d
    d=$(mkfake_ttmetal "with-so" "stale")
    local install="$BATS_TEST_TMPDIR/install"
    run run_install "$d/src" "$d/build" "$install"
    assert_success
    # Installed .so must match the fresh build copy, not the stale source copy.
    assert_equal "$(cat "$install/python_packages/ttnn/ttnn/_ttnn.so")" \
                 "$(cat "$d/build/ttnn/_ttnn.so")"
    refute_output --partial "stale ttnn"
}

@test "install fails when \$BUILD/ttnn/_ttnn.so is missing" {
    local d
    d=$(mkfake_ttmetal "no-so" "stale")
    local install="$BATS_TEST_TMPDIR/install"
    run run_install "$d/src" "$d/build" "$install"
    assert_failure
    assert_output --partial "not found"
    assert_output --partial "tt-metal build did not produce ttnn extensions"
}

@test "install copies Python sources from \$SRC" {
    local d
    d=$(mkfake_ttmetal "with-so" "fresh")
    local install="$BATS_TEST_TMPDIR/install"
    run run_install "$d/src" "$d/build" "$install"
    assert_success
    [ -f "$install/python_packages/ttnn/ttnn/__init__.py" ]
    [ -f "$install/python_packages/ttnn/ttnn/version.py" ]
}

@test "install fails when neither stale .so nor fresh .so exist" {
    local d
    d=$(mkfake_ttmetal "no-so" "fresh")
    local install="$BATS_TEST_TMPDIR/install"
    run run_install "$d/src" "$d/build" "$install"
    assert_failure
    assert_output --partial "not found"
}

@test "install never copies .so files originally in \$SRC" {
    # If the build dir is missing the .so but the source tree has them
    # (stale), install must NOT silently use the source-tree copy.
    local d
    d=$(mkfake_ttmetal "no-so" "stale")
    local install="$BATS_TEST_TMPDIR/install"
    run run_install "$d/src" "$d/build" "$install"
    assert_failure
    # The install dir must not contain stale .so files. Either the dir was
    # not populated at all, or it was wiped before the error.
    [ ! -f "$install/python_packages/ttnn/ttnn/_ttnn.so" ] \
        || ! grep -q "stale" "$install/python_packages/ttnn/ttnn/_ttnn.so"
}

@test "install excludes pre-compiled firmware from JIT source tree" {
    local d
    d=$(mkfake_ttmetal "with-so" "fresh")
    local install="$BATS_TEST_TMPDIR/install"
    run run_install "$d/src" "$d/build" "$install"
    assert_success
    [ -f "$install/tt_metal/api/sample.h" ]
    [ ! -e "$install/tt_metal/pre-compiled" ]
}

@test "install includes pre-compiled firmware when precompile stamp exists" {
    local d
    d=$(mkfake_ttmetal "with-so" "fresh")
    mkdir -p "$d/build/tt_metal/pre-compiled"
    : > "$d/build/tt_metal/pre-compiled/.stamp"
    local install="$BATS_TEST_TMPDIR/install"
    run run_install "$d/src" "$d/build" "$install"
    assert_success
    [ -f "$install/tt_metal/pre-compiled/stale/fw.elf" ]
}
