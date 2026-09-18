#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

load test_helper

@test "dev version resolves to its year-month view" {
    output_file="$BATS_TEST_TMPDIR/output"
    run env GITHUB_OUTPUT="$output_file" \
        "$SCRIPTS_DIR/resolve-s3-publish-prefix.sh" \
        1.2.3.dev20260726

    assert_success
    grep -qx 'prefix=tt-lang/2026-07' "$output_file"
}

@test "final version resolves to the releases view" {
    output_file="$BATS_TEST_TMPDIR/output"
    run env GITHUB_OUTPUT="$output_file" \
        "$SCRIPTS_DIR/resolve-s3-publish-prefix.sh" \
        1.2.3

    assert_success
    grep -qx 'prefix=tt-lang/releases' "$output_file"
}

@test "local label text does not classify a final version as dev" {
    output_file="$BATS_TEST_TMPDIR/output"
    run env GITHUB_OUTPUT="$output_file" \
        "$SCRIPTS_DIR/resolve-s3-publish-prefix.sh" \
        1.2.3+foo.dev1

    assert_success
    grep -qx 'prefix=tt-lang/releases' "$output_file"
}

@test "dev version rejects an out-of-range month" {
    output_file="$BATS_TEST_TMPDIR/output"
    run env GITHUB_OUTPUT="$output_file" \
        "$SCRIPTS_DIR/resolve-s3-publish-prefix.sh" \
        1.2.3.dev20261332

    assert_failure 2
    assert_output --partial "Invalid calendar month in dev version"
    [[ ! -e "$output_file" ]]
}

@test "dev version rejects a non-date dev number" {
    output_file="$BATS_TEST_TMPDIR/output"
    run env GITHUB_OUTPUT="$output_file" \
        "$SCRIPTS_DIR/resolve-s3-publish-prefix.sh" \
        1.2.3.dev1

    assert_failure 2
    assert_output --partial \
        "S3 dev versions must use an 8-digit YYYYMMDD dev number"
    [[ ! -e "$output_file" ]]
}

@test "selected publish preparation rejects an empty artifact root" {
    artifact_root="$BATS_TEST_TMPDIR/artifacts"
    mkdir -p "$artifact_root"

    run "$SCRIPTS_DIR/prepare-selected-s3-publish-dist.sh" \
        1.2.3.dev20260726 \
        "$artifact_root" \
        "$BATS_TEST_TMPDIR/dist"

    assert_failure
    assert_output --partial "No selected wheel artifacts found"
}

@test "selected publish preparation combines bundled and light artifacts" {
    artifact_root="$BATS_TEST_TMPDIR/artifacts"
    publish_dir="$BATS_TEST_TMPDIR/dist"
    mkdir -p "$artifact_root/bundled" "$artifact_root/light"
    touch \
        "$artifact_root/bundled/$(whl 1.2.3.dev20260726)" \
        "$artifact_root/bundled/$(whl_sim 1.2.3.dev20260726)" \
        "$artifact_root/light/$(whl_light_core_cp310 1.2.3.dev20260726)" \
        "$artifact_root/light/$(whl_light_core_cp312 1.2.3.dev20260726)" \
        "$artifact_root/light/$(whl_light 1.2.3.dev20260726)"

    run "$SCRIPTS_DIR/prepare-selected-s3-publish-dist.sh" \
        1.2.3.dev20260726 \
        "$artifact_root" \
        "$publish_dir"

    assert_success
    run find "$publish_dir" -maxdepth 1 -name '*.whl' -print
    assert_success
    assert_line --partial "tt_lang-1.2.3.dev20260726"
    assert_line --partial "tt_lang-1.2.3.dev20260726+light"
    assert_line --partial "tt_lang_light-1.2.3.dev20260726"
}
