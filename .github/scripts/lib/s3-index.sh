# shellcheck shell=bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Sourceable helpers for the tt-lang S3 "directory" listings. Listings are
# written to a key ending in "/" (slash-key) because the pypi.eng CDN serves
# only the exact key requested; an object at <prefix>/index.html does not answer
# a request for <prefix>/.

# Absolute base URL for month-view anchors (top-level objects, not ../).
S3_INDEX_BASE_URL="${S3_INDEX_BASE_URL:-https://pypi.eng.aws.tenstorrent.com}"

S3_INDEX_README_NAME="${S3_INDEX_README_NAME:-README.html}"

_s3_is_readme_name() {
    [[ "$1" == "$S3_INDEX_README_NAME" ||
        "$1" == "README.html" ]]
}

# Root compatibility is limited to final X.Y.Z releases; dev wheels resolve
# from their year-month directories.
_s3_is_stable_release_wheel_name() {
    [[ "$1" =~ ^[A-Za-z0-9_]+-[0-9]+\.[0-9]+\.[0-9]+(\+[^-]+)?-[^-]+-[^-]+-[^-]+\.whl$ ]]
}

# Escape text for HTML element content and double-quoted attributes.
_html_escape() {
    local s="$1"
    # Escape & as \& in each replacement: bash 5.2's patsub_replacement treats a
    # bare & in a ${var//pat/repl} replacement as the matched text.
    s="${s//&/\&amp;}"; s="${s//</\&lt;}"; s="${s//>/\&gt;}"; s="${s//\"/\&quot;}"; s="${s//\'/\&#39;}"
    printf '%s' "$s"
}

# Wrap anchor lines (read from stdin) in the Package-Index HTML skeleton.
s3_render_index() {
    local title; title="$(_html_escape "$1")"
    printf '%s\n' \
        '<!DOCTYPE html>' \
        '<html>' \
        '  <head>' \
        '    <meta charset="UTF-8">' \
        "    <title>${title}</title>" \
        '  </head>' \
        '  <body>'
    cat
    printf '%s\n' \
        '  </body>' \
        '</html>'
}

# Anchors for the immediate children of s3://<bucket>/<prefix>/, listed from S3.
# Sub-prefixes and wheels plus the per-directory README; no sha256 (the hashed
# per-SHA listing is generated at publish time from local files by
# s3_local_wheel_anchors).
s3_child_anchors() {
    local dirs_only=false
    local hidden_stable_wheels=false
    while [[ "${1:-}" == --* ]]; do
        case "$1" in
            --directories-only) dirs_only=true ;;
            --hidden-stable-wheels) hidden_stable_wheels=true ;;
            *)
                echo "s3_child_anchors: unknown option: $1" >&2
                return 2
                ;;
        esac
        shift
    done
    local bucket="$1" prefix="$2" listing col1 col2 col3 name esc
    listing="$(aws s3 ls "s3://${bucket}/${prefix}/")" || return 1
    # shellcheck disable=SC2034
    while read -r col1 col2 col3 name; do
        if [[ "$col1" == "PRE" ]]; then
            name="$col2"
        else
            if [[ "$dirs_only" == true &&
                "$hidden_stable_wheels" == true &&
                "$name" == *.whl ]] &&
                _s3_is_stable_release_wheel_name "$name"; then
                esc="$(_html_escape "$name")"
                # Absolute href: relative would 404 from the no-slash root alias.
                printf '<a href="%s/%s/%s" style="display:none" data-ttlang-hidden-stable-wheel="true">%s</a>\n' "$S3_INDEX_BASE_URL" "$prefix" "$esc" "$esc"
                continue
            fi
            [[ "$dirs_only" == false ]] || continue
            # A find-links directory holds only wheels and the README; skip the
            # slash-key object itself, index.html, attempt.json markers, etc.
            [[ "$name" == *.whl ]] || _s3_is_readme_name "$name" || continue
        fi
        esc="$(_html_escape "$name")"
        if [[ "$col1" == "PRE" ]]; then
            printf '<a href="%s">%s</a><br>\n' "$esc" "$esc"
        else
            printf '<a href="%s/%s/%s">%s</a><br>\n' "$S3_INDEX_BASE_URL" "$prefix" "$esc" "$esc"
        fi
    done <<< "$listing"
}

# Anchors for a local wheel dist: optional README plus each *.whl with a
# #sha256 fragment computed from the local file (no download). Absolute hrefs
# under <prefix>: relative would 404 from the no-slash root alias.
s3_local_wheel_anchors() {
    local include_readme=true
    if [[ "${1:-}" == "--no-readme" ]]; then
        include_readme=false
        shift
    fi
    local prefix="$1" dist_dir="$2" f name esc digest
    if [[ "$include_readme" == true ]]; then
        esc="$(_html_escape "$S3_INDEX_README_NAME")"
        printf '<a href="%s/%s/%s">%s</a><br>\n' "$S3_INDEX_BASE_URL" "$prefix" "$esc" "$esc"
    fi
    for f in "$dist_dir"/*.whl; do
        [[ -e "$f" ]] || continue
        name="$(basename "$f")"; esc="$(_html_escape "$name")"
        digest="$(sha256sum "$f" | awk '{print $1}')"
        printf '<a href="%s/%s/%s#sha256=%s">%s</a><br>\n' "$S3_INDEX_BASE_URL" "$prefix" "$esc" "$digest" "$esc"
    done
}

# Upload an HTML file to both the slash-key s3://<bucket>/<prefix>/ and the
# no-slash alias s3://<bucket>/<prefix>. The CDN serves exact object keys, while
# pip users commonly omit the trailing slash in --find-links URLs.
s3_put_index() {
    local bucket="$1" prefix="$2" html_file="$3"
    aws s3api put-object \
        --bucket "$bucket" \
        --key "${prefix}/" \
        --body "$html_file" \
        --content-type "text/html; charset=utf-8" >/dev/null
    aws s3api put-object \
        --bucket "$bucket" \
        --key "$prefix" \
        --body "$html_file" \
        --content-type "text/html; charset=utf-8" >/dev/null
}

# Regenerate the slash-key listing for <prefix> from its current S3 children.
# Refuses to write if the listing fails or comes back empty, so a transient
# `aws s3 ls` failure can't overwrite a live index with a blank page.
s3_regenerate_index() {
    local dirs_only=false
    local hidden_stable_wheels=false
    while [[ "${1:-}" == --* ]]; do
        case "$1" in
            --directories-only) dirs_only=true ;;
            --hidden-stable-wheels) hidden_stable_wheels=true ;;
            *)
                echo "s3_regenerate_index: unknown option: $1" >&2
                return 2
                ;;
        esac
        shift
    done
    local bucket="$1" prefix="$2" anchors tmp
    local anchor_args=()
    [[ "$dirs_only" == true ]] && anchor_args+=(--directories-only)
    [[ "$hidden_stable_wheels" == true ]] && anchor_args+=(--hidden-stable-wheels)
    anchors="$(s3_child_anchors "${anchor_args[@]}" "$bucket" "$prefix")" || {
        echo "s3_regenerate_index: failed to list s3://${bucket}/${prefix}/" >&2
        return 1
    }
    if [[ -z "$anchors" ]]; then
        echo "s3_regenerate_index: refusing to write an empty index for ${prefix}" >&2
        return 1
    fi
    tmp="$(mktemp)"
    printf '%s\n' "$anchors" | s3_render_index "tt-lang: ${prefix}" > "$tmp"
    s3_put_index "$bucket" "$prefix" "$tmp"
    rm -f "$tmp"
}

_s3_top_level_wheel_view_anchors() {
    local bucket="$1" view_kind="$2" selector="${3:-}"
    local year_month="${selector/-/}"
    local listing col1 col2 col3 name esc
    listing="$(aws s3 ls "s3://${bucket}/tt-lang/")" || return 1
    # shellcheck disable=SC2034
    while read -r col1 col2 col3 name; do
        [[ "$col1" == "PRE" ]] && continue
        [[ "$name" == *.whl ]] || continue
        case "$view_kind" in
            month)
                [[ "$name" == *"dev${year_month}"* ]] || continue
                ;;
            releases)
                _s3_is_stable_release_wheel_name "$name" || continue
                ;;
            *)
                echo "_s3_top_level_wheel_view_anchors: unknown view kind: $view_kind" >&2
                return 2
                ;;
        esac
        esc="$(_html_escape "$name")"
        printf '<a href="%s/tt-lang/%s">%s</a><br>\n' "$S3_INDEX_BASE_URL" "$esc" "$esc"
    done <<< "$listing"
}

# Absolute href, not ../: s3_put_index's no-slash alias would resolve a
# relative ../ one level too high.
s3_month_view_anchors() {
    local bucket="$1" month="$2"
    _s3_top_level_wheel_view_anchors "$bucket" month "$month"
}

s3_release_view_anchors() {
    local bucket="$1"
    _s3_top_level_wheel_view_anchors "$bucket" releases
}

_s3_regenerate_top_level_wheel_view() {
    local bucket="$1" prefix="$2" view_kind="$3" selector="${4:-}"
    local anchors tmp
    anchors="$(_s3_top_level_wheel_view_anchors "$bucket" "$view_kind" "$selector")" || {
        echo "s3_regenerate_${view_kind}_view: failed to list top-level wheels for ${prefix}" >&2
        return 1
    }
    if [[ -z "$anchors" ]]; then
        echo "s3_regenerate_${view_kind}_view: refusing to write an empty view for ${prefix}" >&2
        return 1
    fi
    tmp="$(mktemp)"
    printf '%s\n' "$anchors" | s3_render_index "tt-lang: ${prefix}" > "$tmp"
    s3_put_index "$bucket" "$prefix" "$tmp"
    rm -f "$tmp"
}

# Refuses to write an empty view: a transient ls failure or a wheel-less
# selector must not blank a live page.
s3_regenerate_month_view() {
    local bucket="$1" prefix="$2" month
    month="${prefix#tt-lang/}"
    _s3_regenerate_top_level_wheel_view "$bucket" "$prefix" month "$month"
}

s3_regenerate_release_view() {
    local bucket="$1"
    _s3_regenerate_top_level_wheel_view "$bucket" "tt-lang/releases" releases
}
