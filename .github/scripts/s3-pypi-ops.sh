#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Vetted S3 maintenance for the tt-lang prefixes of the shared tenstorrent-pypi
# bucket. Write operations are constrained to a tt-lang prefix allowlist and are
# dry-run by default; deletes require a confirm token. A read-only escape hatch
# runs allowlisted aws read verbs with arguments passed as argv (no shell).
#
# Usage:
#   s3-pypi-ops.sh --operation <inspect|put-index|move|copy|delete|readonly-cmd>
#       [--prefix P] [--source S] [--dest D] [--confirm TOKEN]
#       [--dry-run true|false] [-- <aws read args>]

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/s3-index.sh
. "$script_dir/lib/s3-index.sh"

bucket="${TTLANG_S3_BUCKET:-tenstorrent-pypi}"
ALLOWED_PREFIXES=(tt-lang/)
READONLY_VERBS=("s3 ls" "s3api head-object" "s3api list-objects-v2")
READONLY_FLAGS=(--bucket --key --prefix --recursive --page-size --max-items --max-keys --delimiter --start-after --human-readable --summarize --no-paginate --output)

operation="" prefix="" src="" dest="" confirm="" dry_run="true"
readonly_args=()

die() { echo "$1" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --operation) [[ $# -ge 2 ]] || die "--operation requires a value"; operation="$2"; shift 2 ;;
        --prefix)    [[ $# -ge 2 ]] || die "--prefix requires a value"; prefix="${2%/}"; shift 2 ;;
        --source)    [[ $# -ge 2 ]] || die "--source requires a value"; src="${2%/}"; shift 2 ;;
        --dest)      [[ $# -ge 2 ]] || die "--dest requires a value"; dest="${2%/}"; shift 2 ;;
        --confirm)   [[ $# -ge 2 ]] || die "--confirm requires a value"; confirm="${2%/}"; shift 2 ;;
        --dry-run)   [[ $# -ge 2 ]] || die "--dry-run requires a value"; dry_run="$2"; shift 2 ;;
        --)          shift; readonly_args=("$@"); break ;;
        *)           die "Unknown argument: $1" ;;
    esac
done

[[ "$dry_run" == "true" || "$dry_run" == "false" ]] || die "--dry-run must be true or false: $dry_run"

# Reject anything that is empty, absolute, an s3:// URI, contains "..", contains
# characters outside a safe set, or does not start with an allowed prefix.
assert_allowed() {
    local path="$1"
    [[ -n "$path" ]] || die "empty prefix is not allowed"
    [[ "$path" != /* && "$path" != s3://* ]] || die "prefix must be bucket-relative"
    [[ "$path" != *..* ]] || die "prefix must not contain '..'"
    [[ "$path" =~ ^[A-Za-z0-9._/-]+$ ]] || die "prefix has invalid characters: $path"
    local allowed
    for allowed in "${ALLOWED_PREFIXES[@]}"; do
        [[ "$path/" == "$allowed"* ]] && return 0
    done
    die "prefix not in the tt-lang allowlist: $path"
}

# Destructive ops must target something strictly under an allowlist root.
assert_destructive_ok() {
    local path="$1"
    assert_allowed "$path"
    local allowed
    for allowed in "${ALLOWED_PREFIXES[@]}"; do
        # e.g. path "tt-lang/x" under "tt-lang/" -> "x" remains.
        if [[ "$path/" == "$allowed"* ]]; then
            local rest="${path#"${allowed%/}"/}"
            [[ -n "$rest" && "$rest" != "$path" ]] || die "refusing destructive op on allowlist root: $path"
            return 0
        fi
    done
    die "internal: prefix passed allowlist but matched no root: $path"
}

_readonly_flag_ok() {
    local f
    for f in "${READONLY_FLAGS[@]}"; do [[ "$1" == "$f" ]] && return 0; done
    return 1
}

# Every token in a read command must be a benign flag or a target under
# s3://<bucket>/tt-lang/. Unknown flags (e.g. --endpoint-url, --profile) are
# rejected so a read cannot redirect the signed request or leak its credential.
assert_readonly_targets() {
    local args=("$@") i=0 tok flag val verb
    local saw_s3_uri=false saw_bucket=false saw_key=false saw_prefix=false
    verb="${args[0]} ${args[1]}"
    while [[ $i -lt ${#args[@]} ]]; do
        tok="${args[$i]}"
        case "$tok" in
            s3://*)
                saw_s3_uri=true
                [[ "$tok" != *..* ]] || die "read-only target must not contain '..': $tok"
                [[ "$tok" == "s3://$bucket/tt-lang/"* ]] \
                    || die "read-only target must be under s3://$bucket/tt-lang/: $tok" ;;
            --*=*)
                flag="${tok%%=*}"; val="${tok#*=}"
                _readonly_flag_ok "$flag" || die "read-only flag not allowed: $flag"
                case "$flag" in
                    --bucket)
                        saw_bucket=true
                        [[ "$val" == "$bucket" ]] || die "read-only --bucket must be $bucket: $val" ;;
                    --key|--prefix)
                        [[ "$flag" == "--key" ]] && saw_key=true || saw_prefix=true
                        [[ "$val" != *..* ]] \
                            || die "read-only --key/--prefix must not contain '..': $val"
                        [[ "$val" == "tt-lang/"* ]] \
                            || die "read-only --key/--prefix must be under tt-lang/: $val" ;;
                esac ;;
            -*)
                _readonly_flag_ok "$tok" || die "read-only flag not allowed: $tok"
                case "$tok" in
                    --bucket)
                        val="${args[$((i+1))]:-}"
                        saw_bucket=true
                        [[ "$val" == "$bucket" ]] || die "read-only --bucket must be $bucket: $val"
                        i=$((i+1)) ;;
                    --key|--prefix)
                        val="${args[$((i+1))]:-}"
                        [[ "$tok" == "--key" ]] && saw_key=true || saw_prefix=true
                        [[ "$val" != *..* ]] \
                            || die "read-only --key/--prefix must not contain '..': $val"
                        [[ "$val" == "tt-lang/"* ]] \
                            || die "read-only --key/--prefix must be under tt-lang/: $val"
                        i=$((i+1)) ;;
                esac ;;
        esac
        i=$((i+1))
    done
    case "$verb" in
        "s3 ls")
            [[ "$saw_s3_uri" == true ]] || die "read-only s3 ls requires an s3://$bucket/tt-lang/ target" ;;
        "s3api head-object")
            [[ "$saw_bucket" == true ]] || die "read-only $verb requires --bucket $bucket"
            [[ "$saw_key" == true ]] || die "read-only $verb requires --key under tt-lang/" ;;
        "s3api list-objects-v2")
            [[ "$saw_bucket" == true ]] || die "read-only $verb requires --bucket $bucket"
            [[ "$saw_prefix" == true ]] || die "read-only $verb requires --prefix under tt-lang/" ;;
    esac
}

run_aws() {
    if [[ "$dry_run" == "false" ]]; then
        aws "$@"
    else
        echo "DRY-RUN: aws $*"
    fi
}

case "$operation" in
    inspect)
        assert_allowed "$prefix"
        aws s3 ls "s3://$bucket/$prefix/" --recursive
        ;;
    readonly-cmd)
        [[ "${#readonly_args[@]}" -ge 2 ]] || die "readonly-cmd needs an aws command after --"
        verb="${readonly_args[0]} ${readonly_args[1]}"
        allowed=false
        for v in "${READONLY_VERBS[@]}"; do [[ "$verb" == "$v" ]] && allowed=true; done
        [[ "$allowed" == true ]] || die "read-only verbs only: ${READONLY_VERBS[*]}"
        assert_readonly_targets "${readonly_args[@]}"
        aws "${readonly_args[@]}"
        ;;
    put-index)
        assert_allowed "$prefix"
        if [[ "$dry_run" == "false" ]]; then
            s3_regenerate_index "$bucket" "$prefix"
        else
            echo "DRY-RUN: regenerate slash-key index for $prefix"
        fi
        ;;
    copy)
        assert_destructive_ok "$src"; assert_allowed "$dest"
        # --copy-props metadata-directive keeps content-type but skips the tag
        # copy; the multipart path's s3:GetObjectTagging is not granted to the
        # role, so without it mv/cp of objects over the multipart threshold fails.
        run_aws s3 cp "s3://$bucket/$src/" "s3://$bucket/$dest/" --recursive --copy-props metadata-directive
        ;;
    move)
        assert_destructive_ok "$src"; assert_allowed "$dest"
        run_aws s3 mv "s3://$bucket/$src/" "s3://$bucket/$dest/" --recursive --copy-props metadata-directive
        ;;
    delete)
        assert_destructive_ok "$prefix"
        [[ "$confirm" == "$prefix" ]] || die "delete requires --confirm to equal the prefix ($prefix)"
        run_aws s3 rm "s3://$bucket/$prefix/" --recursive
        ;;
    *)
        die "unknown operation: $operation"
        ;;
esac
