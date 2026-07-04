# shellcheck shell=bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Sourceable helpers for the tt-lang S3 "directory" listings. Listings are
# written to a key ending in "/" (slash-key) because the pypi.eng CDN serves
# only the exact key requested; an object at <prefix>/index.html does not answer
# a request for <prefix>/.

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
# Sub-prefixes and (wheels + README.txt); no sha256 (the hashed per-SHA listing
# is generated at publish time from local files by s3_local_wheel_anchors).
s3_child_anchors() {
    local bucket="$1" prefix="$2" listing col1 col2 col3 name esc
    listing="$(aws s3 ls "s3://${bucket}/${prefix}/")" || return 1
    # shellcheck disable=SC2034
    while read -r col1 col2 col3 name; do
        if [[ "$col1" == "PRE" ]]; then
            name="$col2"
        else
            # A find-links directory holds only wheels and the README; skip the
            # slash-key object itself, index.html, attempt.json markers, etc.
            [[ "$name" == *.whl || "$name" == "README.txt" ]] || continue
        fi
        esc="$(_html_escape "$name")"
        printf '<a href="%s">%s</a><br>\n' "$esc" "$esc"
    done <<< "$listing"
}

# Anchors for a local wheel dist: README.txt plus each *.whl with a #sha256
# fragment computed from the local file (no download).
s3_local_wheel_anchors() {
    local dist_dir="$1" f name esc digest
    esc="$(_html_escape "README.txt")"
    printf '<a href="%s">%s</a><br>\n' "$esc" "$esc"
    for f in "$dist_dir"/*.whl; do
        [[ -e "$f" ]] || continue
        name="$(basename "$f")"; esc="$(_html_escape "$name")"
        digest="$(sha256sum "$f" | awk '{print $1}')"
        printf '<a href="%s#sha256=%s">%s</a><br>\n' "$esc" "$digest" "$esc"
    done
}

# Upload an HTML file to the slash-key s3://<bucket>/<prefix>/ (trailing slash is
# the object key, so the directory URL resolves).
s3_put_index() {
    local bucket="$1" prefix="$2" html_file="$3"
    aws s3api put-object \
        --bucket "$bucket" \
        --key "${prefix}/" \
        --body "$html_file" \
        --content-type "text/html; charset=utf-8" >/dev/null
}

# Regenerate the slash-key listing for <prefix> from its current S3 children.
# Refuses to write if the listing fails or comes back empty, so a transient
# `aws s3 ls` failure can't overwrite a live index with a blank page.
s3_regenerate_index() {
    local bucket="$1" prefix="$2" anchors tmp
    anchors="$(s3_child_anchors "$bucket" "$prefix")" || {
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
