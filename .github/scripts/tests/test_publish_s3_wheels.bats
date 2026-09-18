#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/publish-s3-wheels.sh.

load test_helper

make_aws_mock() {
    cat > "$BINDIR/aws" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_AWS_ARGS"
if [[ "$1 $2" == "s3 ls" ]]; then
    case "$3" in
        *tt-lang/) cat "$LS_PARENT" ;;
    esac
elif [[ "$1 $2" == "s3api head-object" ]]; then
    if [[ "${HEAD_OBJECT_EXISTS:-false}" == true ]]; then
        exit 0
    fi
    if [[ -n "${HEAD_OBJECT_EXISTS_KEY:-}" && "$*" == *"$HEAD_OBJECT_EXISTS_KEY"* ]]; then
        exit 0
    fi
    echo "An error occurred (404) when calling the HeadObject operation: Not Found" >&2
    exit 1
elif [[ "$1 $2" == "s3api put-object" ]]; then
    shift 2
    key="" body=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --key) key="$2"; shift 2 ;;
            --body) body="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    cp "$body" "$PUT_BODIES_DIR/${key//\//_}"
fi
EOF
    chmod +x "$BINDIR/aws"
}

# Create a temp dist dir containing the named (empty) wheel files. Echoes the
# dir path.
make_dist_dir() {
    local dir
    dir=$(mktemp -d "$BATS_TEST_TMPDIR/dist.XXXXXX")
    for name in "$@"; do
        : > "$dir/$name"
    done
    echo "$dir"
}

setup() {
    SCRIPT="$SCRIPTS_DIR/publish-s3-wheels.sh"
    FAKE_AWS_ARGS="$BATS_TEST_TMPDIR/aws_args"
    LS_PARENT="$BATS_TEST_TMPDIR/ls_parent"
    PUT_BODIES_DIR="$BATS_TEST_TMPDIR/put_bodies"
    : > "$FAKE_AWS_ARGS"
    mkdir -p "$PUT_BODIES_DIR"
    cat > "$LS_PARENT" <<EOF
                           PRE 2026-07/
                           PRE releases/
2026-07-04 00:00:00       1234 tt_lang-0.0.1-py3-none-any.whl
2026-07-04 00:00:00       1234 tt_lang-0.0.1.dev20260704-py3-none-any.whl
EOF
    export FAKE_AWS_ARGS LS_PARENT PUT_BODIES_DIR
    export HEAD_OBJECT_EXISTS=false
    export HEAD_OBJECT_EXISTS_KEY=""
    BINDIR="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$BINDIR"
    make_aws_mock
    export PATH="$BINDIR:$PATH"
}

@test "no arguments -> usage error (exit 2)" {
    run -2 "$SCRIPT"
}

@test "too many arguments -> usage error (exit 2)" {
    run -2 "$SCRIPT" dist extra
}

@test "empty dist dir -> error (exit 1)" {
    dir=$(make_dist_dir)
    run -1 "$SCRIPT" --prefix tt-lang/2026-07 "$dir"
    assert_output --partial "No wheels found under $dir"
}

@test "--prefix is required" {
    dir=$(make_dist_dir "tt_lang-1.0-py3-none-any.whl")
    run -2 "$SCRIPT" "$dir"
    assert_output --partial "Usage:"
}

@test "--prefix must be a supported generated tt-lang wheel view" {
    dir=$(make_dist_dir "tt_lang-1.0-py3-none-any.whl")
    run -2 "$SCRIPT" --prefix tt-lang-light/2026-07 "$dir"
    assert_output --partial "Publish prefix must be tt-lang/<YYYY-MM> or tt-lang/releases"
    run -2 "$SCRIPT" --prefix tt-lang/ttmetal "$dir"
    assert_output --partial "Publish prefix must be tt-lang/<YYYY-MM> or tt-lang/releases"
    run -2 "$SCRIPT" --prefix tt-lang/2026-13 "$dir"
    assert_output --partial "Publish prefix must be tt-lang/<YYYY-MM> or tt-lang/releases"
}

@test "--prefix publishes top-level wheels and regenerates a month view" {
    dir=$(make_dist_dir \
        "tt_lang-1.0.0.dev20260705-py3-none-any.whl" \
        "tt_lang_light-1.0.0.dev20260705-py3-none-any.whl")
    cat > "$LS_PARENT" <<EOF
                           PRE 2026-07/
                           PRE releases/
2026-07-04 00:00:00       1234 tt_lang-0.0.1-py3-none-any.whl
2026-07-04 00:00:00       1234 tt_lang-0.9.0.dev20260704-py3-none-any.whl
2026-07-05 00:00:00       1234 tt_lang-1.0.0.dev20260705-py3-none-any.whl
2026-07-05 00:00:00       1234 tt_lang_light-1.0.0.dev20260705-py3-none-any.whl
EOF
    run -0 "$SCRIPT" --prefix tt-lang/2026-07 "$dir"

    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3 cp $dir/tt_lang-1.0.0.dev20260705-py3-none-any.whl s3://tenstorrent-pypi/tt-lang/tt_lang-1.0.0.dev20260705-py3-none-any.whl"
    assert_output --partial "s3 cp $dir/tt_lang_light-1.0.0.dev20260705-py3-none-any.whl s3://tenstorrent-pypi/tt-lang/tt_lang_light-1.0.0.dev20260705-py3-none-any.whl"
    assert_output --partial "s3api put-object --bucket tenstorrent-pypi --key tt-lang/2026-07/"
    assert_output --partial "s3api put-object --bucket tenstorrent-pypi --key tt-lang/2026-07 --body"
    assert_output --partial "s3 ls s3://tenstorrent-pypi/tt-lang/"
    assert_output --partial "s3api put-object --bucket tenstorrent-pypi --key tt-lang/"
    assert_output --partial "s3api put-object --bucket tenstorrent-pypi --key tt-lang --body"

    run cat "$PUT_BODIES_DIR/tt-lang_2026-07_"
    assert_output --partial "https://pypi.eng.aws.tenstorrent.com/tt-lang/tt_lang-0.9.0.dev20260704-py3-none-any.whl"
    assert_output --partial "https://pypi.eng.aws.tenstorrent.com/tt-lang/tt_lang-1.0.0.dev20260705-py3-none-any.whl"
    assert_output --partial "https://pypi.eng.aws.tenstorrent.com/tt-lang/tt_lang_light-1.0.0.dev20260705-py3-none-any.whl"
    refute_output --partial "#sha256="
    refute_output --partial "README"
    refute_output --partial "tt_lang-0.0.1-py3-none-any.whl\">"

    run cat "$PUT_BODIES_DIR/tt-lang_"
    assert_output --partial '<a href="2026-07/">2026-07/</a><br>'
    assert_output --partial '<a href="releases/">releases/</a><br>'
    assert_output --partial '<a href="https://pypi.eng.aws.tenstorrent.com/tt-lang/tt_lang-0.0.1-py3-none-any.whl" style="display:none" data-ttlang-hidden-stable-wheel="true">tt_lang-0.0.1-py3-none-any.whl</a>'
    refute_output --partial "tt_lang-0.0.1.dev20260704-py3-none-any.whl"

    run cat "$PUT_BODIES_DIR/tt-lang"
    assert_output --partial '<a href="2026-07/">2026-07/</a><br>'
    assert_output --partial '<a href="releases/">releases/</a><br>'
    assert_output --partial '<a href="https://pypi.eng.aws.tenstorrent.com/tt-lang/tt_lang-0.0.1-py3-none-any.whl" style="display:none" data-ttlang-hidden-stable-wheel="true">tt_lang-0.0.1-py3-none-any.whl</a>'
    refute_output --partial "tt_lang-0.0.1.dev20260704-py3-none-any.whl"
}

@test "--prefix tt-lang/releases publishes top-level wheels and regenerates release view" {
    dir=$(make_dist_dir \
        "tt_lang-1.0.0-py3-none-any.whl" \
        "tt_lang_light-1.0.0-py3-none-any.whl")
    cat > "$LS_PARENT" <<EOF
                           PRE 2026-07/
                           PRE releases/
2026-07-04 00:00:00       1234 tt_lang-1.0.0-py3-none-any.whl
2026-07-04 00:00:00       1234 tt_lang_light-1.0.0-py3-none-any.whl
2026-07-04 00:00:00       1234 tt_lang-1.0.0.dev20260704-py3-none-any.whl
EOF
    run -0 "$SCRIPT" --prefix tt-lang/releases "$dir"

    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3 cp $dir/tt_lang-1.0.0-py3-none-any.whl s3://tenstorrent-pypi/tt-lang/tt_lang-1.0.0-py3-none-any.whl"
    assert_output --partial "s3 cp $dir/tt_lang_light-1.0.0-py3-none-any.whl s3://tenstorrent-pypi/tt-lang/tt_lang_light-1.0.0-py3-none-any.whl"
    assert_output --partial "s3api put-object --bucket tenstorrent-pypi --key tt-lang/releases/"
    assert_output --partial "s3api put-object --bucket tenstorrent-pypi --key tt-lang/releases --body"
    assert_output --partial "s3api put-object --bucket tenstorrent-pypi --key tt-lang/"

    run cat "$PUT_BODIES_DIR/tt-lang_releases_"
    assert_output --partial "https://pypi.eng.aws.tenstorrent.com/tt-lang/tt_lang-1.0.0-py3-none-any.whl"
    assert_output --partial "https://pypi.eng.aws.tenstorrent.com/tt-lang/tt_lang_light-1.0.0-py3-none-any.whl"
    refute_output --partial "dev20260704"

    run cat "$PUT_BODIES_DIR/tt-lang_"
    assert_output --partial '<a href="releases/">releases/</a><br>'
    assert_output --partial '<a href="https://pypi.eng.aws.tenstorrent.com/tt-lang/tt_lang-1.0.0-py3-none-any.whl" style="display:none" data-ttlang-hidden-stable-wheel="true">tt_lang-1.0.0-py3-none-any.whl</a>'
    refute_output --partial "tt_lang-1.0.0.dev20260704-py3-none-any.whl"
}

@test "--prefix composes with --overwrite" {
    dir=$(make_dist_dir "tt_lang-1.0-py3-none-any.whl")
    run -0 "$SCRIPT" --overwrite --prefix tt-lang/2026-07 "$dir"

    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3://tenstorrent-pypi/tt-lang/tt_lang-1.0-py3-none-any.whl"
}

@test "--overwrite-if true skips object-existence checks" {
    dir=$(make_dist_dir "tt_lang-1.0-py3-none-any.whl")
    HEAD_OBJECT_EXISTS=true run -0 "$SCRIPT" \
        --overwrite-if true \
        --prefix tt-lang/2026-07 \
        "$dir"

    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3api head-object"
    assert_output --partial "s3 cp $dir/tt_lang-1.0-py3-none-any.whl"
}

@test "--overwrite-if false preserves object-existence checks" {
    dir=$(make_dist_dir "tt_lang-1.0-py3-none-any.whl")
    HEAD_OBJECT_EXISTS=true run -1 "$SCRIPT" \
        --overwrite-if false \
        --prefix tt-lang/2026-07 \
        "$dir"

    assert_output --partial "S3 object already exists"
}

@test "--overwrite-if rejects non-boolean values" {
    dir=$(make_dist_dir "tt_lang-1.0-py3-none-any.whl")
    run -2 "$SCRIPT" \
        --overwrite-if yes \
        --prefix tt-lang/2026-07 \
        "$dir"
}

@test "--prefix without --overwrite rejects an existing wheel object" {
    dir=$(make_dist_dir "tt_lang-1.0-py3-none-any.whl")
    HEAD_OBJECT_EXISTS=true run -1 "$SCRIPT" --prefix tt-lang/2026-07 "$dir"
    assert_output --partial "S3 object already exists"

    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3api head-object --bucket tenstorrent-pypi --key tt-lang/tt_lang-1.0-py3-none-any.whl"
    refute_output --partial "s3 cp"
}

@test "--prefix preflights all destination keys before uploading" {
    dir=$(make_dist_dir \
        "tt_lang-1.0-py3-none-any.whl" \
        "tt_lang_light-1.0-py3-none-any.whl")
    HEAD_OBJECT_EXISTS_KEY="tt_lang_light-1.0-py3-none-any.whl" \
        run -1 "$SCRIPT" --prefix tt-lang/2026-07 "$dir"
    assert_output --partial "S3 object already exists"

    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3api head-object --bucket tenstorrent-pypi --key tt-lang/tt_lang-1.0-py3-none-any.whl"
    assert_output --partial "s3api head-object --bucket tenstorrent-pypi --key tt-lang/tt_lang_light-1.0-py3-none-any.whl"
    refute_output --partial "s3 cp"
}

@test "--prefix without a value -> usage error (exit 2)" {
    run -2 "$SCRIPT" --prefix
}

# Persistent fake S3: a flat key->size manifest, driven by real s3 cp/put-object
# writes and read back by s3 ls, so two sequential publishes into the same
# month actually accumulate instead of relying on a hand-authored ls fixture.
@test "publishing twice into the same month accumulates both dev wheels at top-level keys" {
    OBJSTORE="$BATS_TEST_TMPDIR/objstore.tsv"
    : > "$OBJSTORE"
    export OBJSTORE
    cat > "$BINDIR/aws" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_AWS_ARGS"
strip_key() { local u="$1"; u="${u#s3://}"; echo "${u#*/}"; }
put() {
    local key="$1" size="$2"
    grep -v -F -- "$(printf '%s\t' "$key")" "$OBJSTORE" > "$OBJSTORE.tmp" 2>/dev/null || true
    mv "$OBJSTORE.tmp" "$OBJSTORE"
    printf '%s\t%s\n' "$key" "$size" >> "$OBJSTORE"
}
case "$1 $2" in
    "s3 cp")
        src="$3"; key="$(strip_key "$4")"
        put "$key" "$(wc -c < "$src")"
        ;;
    "s3api head-object")
        shift 2; key=""
        while [[ $# -gt 0 ]]; do case "$1" in --key) key="$2"; shift 2 ;; *) shift ;; esac; done
        grep -q -F -- "$(printf '%s\t' "$key")" "$OBJSTORE" 2>/dev/null && exit 0
        echo "An error occurred (404) when calling the HeadObject operation: Not Found" >&2
        exit 1
        ;;
    "s3api put-object")
        shift 2; key="" body=""
        while [[ $# -gt 0 ]]; do
            case "$1" in
                --key) key="$2"; shift 2 ;;
                --body) body="$2"; shift 2 ;;
                *) shift ;;
            esac
        done
        put "$key" "$(wc -c < "$body")"
        cp "$body" "$PUT_BODIES_DIR/${key//\//_}"
        ;;
    "s3 ls")
        prefix="$(strip_key "$3")"
        declare -A is_dir=() printed=()
        while IFS=$'\t' read -r key size; do
            [[ "$key" == "$prefix"* ]] || continue
            rest="${key#"$prefix"}"
            [[ -n "$rest" ]] || continue
            [[ "$rest" == */* ]] && is_dir["${rest%%/*}"]=1
        done < "$OBJSTORE"
        while IFS=$'\t' read -r key size; do
            [[ "$key" == "$prefix"* ]] || continue
            rest="${key#"$prefix"}"
            [[ -n "$rest" ]] || continue
            if [[ "$rest" == */* ]]; then
                seg="${rest%%/*}"
                [[ -n "${printed[$seg]:-}" ]] && continue
                printed["$seg"]=1
                printf '                           PRE %s/\n' "$seg"
            else
                [[ -n "${is_dir[$rest]:-}" ]] && continue
                printf '2026-01-01 00:00:00 %8d %s\n' "$size" "$rest"
            fi
        done < "$OBJSTORE"
        ;;
esac
EOF
    chmod +x "$BINDIR/aws"

    dir1=$(make_dist_dir "tt_lang-1.0.0.dev20260704-py3-none-any.whl")
    run -0 "$SCRIPT" --prefix tt-lang/2026-07 "$dir1"

    dir2=$(make_dist_dir "tt_lang_light-1.0.0.dev20260705-py3-none-any.whl")
    run -0 "$SCRIPT" --prefix tt-lang/2026-07 "$dir2"

    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3 cp $dir1/tt_lang-1.0.0.dev20260704-py3-none-any.whl s3://tenstorrent-pypi/tt-lang/tt_lang-1.0.0.dev20260704-py3-none-any.whl"
    assert_output --partial "s3 cp $dir2/tt_lang_light-1.0.0.dev20260705-py3-none-any.whl s3://tenstorrent-pypi/tt-lang/tt_lang_light-1.0.0.dev20260705-py3-none-any.whl"

    run cat "$PUT_BODIES_DIR/tt-lang_2026-07_"
    assert_output --partial "https://pypi.eng.aws.tenstorrent.com/tt-lang/tt_lang-1.0.0.dev20260704-py3-none-any.whl"
    assert_output --partial "https://pypi.eng.aws.tenstorrent.com/tt-lang/tt_lang_light-1.0.0.dev20260705-py3-none-any.whl"
}
