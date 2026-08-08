#!/usr/bin/env bats
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Tests for .github/scripts/s3-pypi-ops.sh.

load test_helper

setup() {
    SCRIPT="$SCRIPTS_DIR/s3-pypi-ops.sh"
    FAKE_AWS_ARGS="$BATS_TEST_TMPDIR/aws_args"
    : > "$FAKE_AWS_ARGS"
    export FAKE_AWS_ARGS
    local bindir="$BATS_TEST_TMPDIR/bin"
    mkdir -p "$bindir"
    cat > "$bindir/aws" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >> "$FAKE_AWS_ARGS"
EOF
    chmod +x "$bindir/aws"
    export PATH="$bindir:$PATH"
}

@test "rejects a prefix outside the tt-lang allowlist" {
    run -2 "$SCRIPT" --operation delete --prefix ttnn/foo --confirm ttnn/foo --dry-run false
    assert_output --partial "not in the tt-lang allowlist"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "rejects the sibling tt-lang-light/ and tt-lang-sim/ package prefixes" {
    run -2 "$SCRIPT" --operation put-index --prefix tt-lang-light/2026-07 --dry-run false
    assert_output --partial "not in the tt-lang allowlist"
    run -2 "$SCRIPT" --operation put-index --prefix tt-lang-sim/2026-07 --dry-run false
    assert_output --partial "not in the tt-lang allowlist"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "rejects the bucket root and bare allowlist root for delete" {
    run -2 "$SCRIPT" --operation delete --prefix tt-lang/ --confirm tt-lang/ --dry-run false
    assert_output --partial "refusing destructive op"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "rejects shell metacharacters in a prefix" {
    run -2 "$SCRIPT" --operation inspect --prefix 'tt-lang/x;rm -rf /'
    assert_output --partial "invalid characters"
}

@test "rejects an absolute path" {
    run -2 "$SCRIPT" --operation inspect --prefix /etc/passwd
    assert_output --partial "prefix must be bucket-relative"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "rejects an s3:// URI" {
    run -2 "$SCRIPT" --operation inspect --prefix s3://tenstorrent-pypi/tt-lang
    assert_output --partial "prefix must be bucket-relative"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "rejects '..' path traversal" {
    run -2 "$SCRIPT" --operation inspect --prefix tt-lang/x/../ttnn
    assert_output --partial ".."
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "rejects a prefix that only shares a leading substring with an allowlist entry" {
    run -2 "$SCRIPT" --operation inspect --prefix tt-lang-evil/foo
    assert_output --partial "not in the tt-lang allowlist"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"

    run -2 "$SCRIPT" --operation inspect --prefix tt-langX/foo
    assert_output --partial "not in the tt-lang allowlist"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "delete requires a matching confirm token" {
    run -2 "$SCRIPT" --operation delete --prefix tt-lang/ttmetal/dead --confirm wrong --dry-run false
    assert_output --partial "confirm"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "rm"
}

@test "move rejects a bare allowlist root as the source" {
    run -2 "$SCRIPT" --operation move --source tt-lang --dest tt-lang/ttmetal/x --dry-run false
    assert_output --partial "refusing destructive op"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "copy rejects a bare allowlist root as the source" {
    run -2 "$SCRIPT" --operation copy --source tt-lang --dest tt-lang/x --dry-run false
    assert_output --partial "refusing destructive op"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "move rejects a dest outside the tt-lang allowlist" {
    run -2 "$SCRIPT" --operation move --source tt-lang/x --dest ttnn/x --dry-run false
    assert_output --partial "not in the tt-lang allowlist"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "copy rejects a dest outside the tt-lang allowlist" {
    run -2 "$SCRIPT" --operation copy --source tt-lang/x --dest ttnn/x --dry-run false
    assert_output --partial "not in the tt-lang allowlist"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "dry-run prints the plan and performs no writes" {
    run -0 "$SCRIPT" --operation move --source tt-lang/13adda8 --dest tt-lang/ttmetal/13adda8
    assert_output --partial "DRY-RUN"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "mv"
}

@test "move (dry-run false) runs recursive mv" {
    run -0 "$SCRIPT" --operation move --source tt-lang/13adda8 --dest tt-lang/ttmetal/13adda8 --dry-run false
    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3 mv s3://tenstorrent-pypi/tt-lang/13adda8/ s3://tenstorrent-pypi/tt-lang/ttmetal/13adda8/ --recursive --copy-props metadata-directive"
}

@test "copy (dry-run false) runs recursive cp" {
    run -0 "$SCRIPT" --operation copy --source tt-lang/a --dest tt-lang/b --dry-run false
    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3 cp s3://tenstorrent-pypi/tt-lang/a/ s3://tenstorrent-pypi/tt-lang/b/ --recursive --copy-props metadata-directive"
}

@test "inspect runs recursive ls" {
    run -0 "$SCRIPT" --operation inspect --prefix tt-lang/x
    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3 ls s3://tenstorrent-pypi/tt-lang/x/ --recursive"
}

@test "put-index dry-run prints the plan and performs no writes" {
    run -0 "$SCRIPT" --operation put-index --prefix tt-lang/ttmetal
    assert_output --partial "DRY-RUN"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "put-object"
}

@test "top-level put-index hides stable root wheels and keeps README" {
    S3_BODY="$BATS_TEST_TMPDIR/tt-lang-index.html"
    export S3_BODY
    cat > "$BATS_TEST_TMPDIR/bin/aws" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_AWS_ARGS"
if [[ "$1 $2" == "s3 ls" ]]; then
    printf '                           PRE 2026-07/\n'
    printf '                           PRE releases/\n'
    printf '                           PRE ttmetal/\n'
    printf '2026-07-04 00:00:00       1234 tt_lang-0.0.1-py3-none-any.whl\n'
    printf '2026-07-04 00:00:00       1234 tt_lang-0.0.1.dev20260704-py3-none-any.whl\n'
elif [[ "$1 $2" == "s3api get-object" ]]; then
    cp "$S3_BODY" "${@: -1}"
elif [[ "$1 $2" == "s3api put-object" ]]; then
    shift 2
    body=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --body) body="$2"; shift 2 ;;
            *) shift ;;
        esac
    done
    cp "$body" "$S3_BODY"
fi
EOF
    chmod +x "$BATS_TEST_TMPDIR/bin/aws"

    run -0 "$SCRIPT" --operation put-index --prefix tt-lang --dry-run false

    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3 ls s3://tenstorrent-pypi/tt-lang/"
    assert_output --partial "s3api put-object --bucket tenstorrent-pypi --key tt-lang/"
    assert_output --partial "s3api put-object --bucket tenstorrent-pypi --key tt-lang --body"
    assert_output --partial "s3api get-object --bucket tenstorrent-pypi --key tt-lang/"

    run cat "$S3_BODY"
    assert_output --partial 'id="ttlang-s3-readme"'
    assert_output --partial '<a href="2026-07/">2026-07/</a><br>'
    assert_output --partial '<a href="releases/">releases/</a><br>'
    assert_output --partial '<a href="ttmetal/">ttmetal/</a><br>'
    assert_output --partial '<a href="https://pypi.eng.aws.tenstorrent.com/tt-lang/tt_lang-0.0.1-py3-none-any.whl" style="display:none" data-ttlang-hidden-stable-wheel="true">tt_lang-0.0.1-py3-none-any.whl</a>'
    refute_output --partial "tt_lang-0.0.1.dev20260704-py3-none-any.whl"
}

@test "releases put-index rebuilds the stable release view" {
    S3_RELEASE_BODY="$BATS_TEST_TMPDIR/tt-lang-releases-index.html"
    export S3_RELEASE_BODY
    cat > "$BATS_TEST_TMPDIR/bin/aws" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_AWS_ARGS"
if [[ "$1 $2" == "s3 ls" ]]; then
    printf '                           PRE 2026-07/\n'
    printf '                           PRE releases/\n'
    printf '2026-07-04 00:00:00       1234 tt_lang-0.0.1-py3-none-any.whl\n'
    printf '2026-07-04 00:00:00       1234 tt_lang_light-0.0.1-py3-none-any.whl\n'
    printf '2026-07-04 00:00:00       1234 tt_lang-0.0.1.dev20260704-py3-none-any.whl\n'
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
    case "$key" in
        tt-lang/releases | tt-lang/releases/) cp "$body" "$S3_RELEASE_BODY" ;;
    esac
fi
EOF
    chmod +x "$BATS_TEST_TMPDIR/bin/aws"

    run -0 "$SCRIPT" --operation put-index --prefix tt-lang/releases --dry-run false

    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3 ls s3://tenstorrent-pypi/tt-lang/"
    assert_output --partial "s3api put-object --bucket tenstorrent-pypi --key tt-lang/releases/"
    assert_output --partial "s3api put-object --bucket tenstorrent-pypi --key tt-lang/releases --body"

    run cat "$S3_RELEASE_BODY"
    assert_output --partial "https://pypi.eng.aws.tenstorrent.com/tt-lang/tt_lang-0.0.1-py3-none-any.whl"
    assert_output --partial "https://pypi.eng.aws.tenstorrent.com/tt-lang/tt_lang_light-0.0.1-py3-none-any.whl"
    refute_output --partial "dev20260704"
}

@test "rejects an unknown operation" {
    run -2 "$SCRIPT" --operation bogus
    assert_output --partial "unknown operation"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "readonly-cmd allows ls but rejects rm" {
    run -0 "$SCRIPT" --operation readonly-cmd -- s3 ls s3://tenstorrent-pypi/tt-lang/
    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3 ls s3://tenstorrent-pypi/tt-lang/"

    run -2 "$SCRIPT" --operation readonly-cmd -- s3 rm s3://tenstorrent-pypi/tt-lang/x
    assert_output --partial "read-only"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "rm"
}

@test "readonly-cmd rejects other write verbs" {
    run -2 "$SCRIPT" --operation readonly-cmd -- s3 cp x y
    assert_output --partial "read-only"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"

    run -2 "$SCRIPT" --operation readonly-cmd -- s3api put-object --bucket tenstorrent-pypi --key tt-lang/x --body f
    assert_output --partial "read-only"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "readonly-cmd rejects an s3:// target outside tt-lang/" {
    run -2 "$SCRIPT" --operation readonly-cmd -- s3 ls s3://tenstorrent-pypi/ttnn/
    assert_output --partial "read-only target must be under s3://tenstorrent-pypi/tt-lang/"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "readonly-cmd rejects bucket-root listing forms" {
    run -2 "$SCRIPT" --operation readonly-cmd -- s3 ls
    assert_output --partial "read-only s3 ls requires"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"

    run -2 "$SCRIPT" --operation readonly-cmd -- s3api list-objects-v2 --bucket tenstorrent-pypi
    assert_output --partial "requires --prefix under tt-lang/"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "list-objects-v2"
}

@test "readonly-cmd rejects a --key outside tt-lang/" {
    run -2 "$SCRIPT" --operation readonly-cmd -- s3api head-object --bucket tenstorrent-pypi --key ttnn/x
    assert_output --partial "read-only --key/--prefix must be under tt-lang/"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "readonly-cmd rejects --arg=value targets outside tt-lang/" {
    run -2 "$SCRIPT" --operation readonly-cmd -- s3api list-objects-v2 --bucket=tenstorrent-pypi --prefix=ttnn/
    assert_output --partial "read-only --key/--prefix must be under tt-lang/"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "list-objects-v2"
}

@test "readonly-cmd allows a --key under tt-lang/" {
    run -0 "$SCRIPT" --operation readonly-cmd -- s3api head-object --bucket tenstorrent-pypi --key tt-lang/ttmetal/13adda8/README.html
    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3api head-object --bucket tenstorrent-pypi --key tt-lang/ttmetal/13adda8/README.html"
}

@test "readonly-cmd allows a --prefix under tt-lang/" {
    run -0 "$SCRIPT" --operation readonly-cmd -- s3api list-objects-v2 --bucket=tenstorrent-pypi --prefix=tt-lang/ttmetal/
    run cat "$FAKE_AWS_ARGS"
    assert_output --partial "s3api list-objects-v2 --bucket=tenstorrent-pypi --prefix=tt-lang/ttmetal/"
}

@test "readonly-cmd rejects an unlisted flag (space form)" {
    run -2 "$SCRIPT" --operation readonly-cmd -- s3 ls s3://tenstorrent-pypi/tt-lang/ --endpoint-url http://x
    assert_output --partial "flag not allowed"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "readonly-cmd rejects an unlisted flag (= form)" {
    run -2 "$SCRIPT" --operation readonly-cmd -- s3 ls s3://tenstorrent-pypi/tt-lang/ --endpoint-url=http://x
    assert_output --partial "flag not allowed"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "readonly-cmd rejects --profile" {
    run -2 "$SCRIPT" --operation readonly-cmd -- s3 ls s3://tenstorrent-pypi/tt-lang/ --profile evil
    assert_output --partial "flag not allowed"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "readonly-cmd rejects an s3:// target missing the trailing slash" {
    run -2 "$SCRIPT" --operation readonly-cmd -- s3 ls s3://tenstorrent-pypi/tt-lang
    assert_output --partial "must be under s3://tenstorrent-pypi/tt-lang/"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "readonly-cmd rejects a --prefix missing the trailing slash" {
    run -2 "$SCRIPT" --operation readonly-cmd -- s3api list-objects-v2 --bucket tenstorrent-pypi --prefix tt-lang
    assert_output --partial "must be under tt-lang/"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "readonly-cmd rejects '..' in an s3:// target" {
    run -2 "$SCRIPT" --operation readonly-cmd -- s3 ls s3://tenstorrent-pypi/tt-lang/../ttnn/
    assert_output --partial ".."
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "readonly-cmd rejects '..' in a --prefix value" {
    run -2 "$SCRIPT" --operation readonly-cmd -- s3api list-objects-v2 --bucket tenstorrent-pypi --prefix tt-lang/../ttnn
    assert_output --partial ".."
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "list-objects-v2"
}

@test "readonly-cmd rejects get-object (verb dropped)" {
    run -2 "$SCRIPT" --operation readonly-cmd -- s3api get-object --bucket tenstorrent-pypi --key tt-lang/x out
    assert_output --partial "read-only verbs only"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "dry-run rejects a non-boolean value" {
    run -2 "$SCRIPT" --operation move --source tt-lang/13adda8 --dest tt-lang/ttmetal/13adda8 --dry-run bogus
    assert_output --partial "must be true or false"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "dry-run rejects a case-mismatched boolean value" {
    run -2 "$SCRIPT" --operation move --source tt-lang/13adda8 --dest tt-lang/ttmetal/13adda8 --dry-run FALSE
    assert_output --partial "must be true or false"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}

@test "rejects a value-taking flag with no value" {
    run -2 "$SCRIPT" --operation
    assert_output --partial "--operation requires a value"
    run cat "$FAKE_AWS_ARGS"
    refute_output --partial "s3"
}
