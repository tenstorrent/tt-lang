# shellcheck shell=bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Sourceable helpers for the tt-lang S3 "directory" listings. Listings are
# written to a key ending in "/" (slash-key) because the pypi.eng CDN serves
# only the exact key requested; an object at <prefix>/index.html does not answer
# a request for <prefix>/.

# Wrap anchor lines (read from stdin) in the Package-Index HTML skeleton.
s3_render_index() {
    local title="$1"
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

# Emit one anchor line per immediate child of s3://<bucket>/<prefix>/.
# `aws s3 ls` prints "PRE name/" for sub-prefixes and "<date> <time> <size> name"
# for objects. Skip the slash-key object itself (empty name) and legacy index.html.
s3_child_anchors() {
    local bucket="$1" prefix="$2" listing col1 col2 col3 name
    listing="$(aws s3 ls "s3://${bucket}/${prefix}/")" || return 1
    # col3 (object size) is consumed only for field alignment, never read.
    # shellcheck disable=SC2034
    while read -r col1 col2 col3 name; do
        if [[ "$col1" == "PRE" ]]; then
            name="$col2"
        fi
        [[ -z "$name" || "$name" == "index.html" ]] && continue
        printf '<a href="%s">%s</a><br>\n' "$name" "$name"
    done <<< "$listing"
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
