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
ALLOWED_PREFIXES=(tt-lang/ tt-lang-light/ tt-lang-sim/)
READONLY_VERBS=("s3 ls" "s3api head-object" "s3api list-objects-v2" "s3api get-object")

operation="" prefix="" src="" dest="" confirm="" dry_run="true"
readonly_args=()

die() { echo "$1" >&2; exit 2; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --operation) operation="$2"; shift 2 ;;
        --prefix)    prefix="${2%/}"; shift 2 ;;
        --source)    src="${2%/}"; shift 2 ;;
        --dest)      dest="${2%/}"; shift 2 ;;
        --confirm)   confirm="${2%/}"; shift 2 ;;
        --dry-run)   dry_run="$2"; shift 2 ;;
        --)          shift; readonly_args=("$@"); break ;;
        *)           die "Unknown argument: $1" ;;
    esac
done

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

run_aws() {
    if [[ "$dry_run" == "true" ]]; then
        echo "DRY-RUN: aws $*"
    else
        aws "$@"
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
        aws "${readonly_args[@]}"
        ;;
    put-index)
        assert_allowed "$prefix"
        if [[ "$dry_run" == "true" ]]; then
            echo "DRY-RUN: regenerate slash-key index for $prefix"
        else
            s3_regenerate_index "$bucket" "$prefix"
        fi
        ;;
    copy)
        assert_allowed "$src"; assert_allowed "$dest"
        run_aws s3 cp "s3://$bucket/$src/" "s3://$bucket/$dest/" --recursive
        ;;
    move)
        assert_destructive_ok "$src"; assert_allowed "$dest"
        run_aws s3 mv "s3://$bucket/$src/" "s3://$bucket/$dest/" --recursive
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
