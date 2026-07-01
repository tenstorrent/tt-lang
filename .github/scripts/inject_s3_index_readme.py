#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Inject the S3-index README into an s3pypi-generated root index.html.

s3pypi rewrites the root index.html on every `upload --put-root-index`, so this
runs after each upload to restore the human-facing README above the package
anchor links. PEP 503 clients read only the <a> links, so the injected block is
inert for pip resolution.

The injected block is delimited by sentinel comments and replaced in place on
re-run, so repeated injection never duplicates it.

Usage: inject_s3_index_readme.py <readme.md> <index.html>
"""

import html
import re
import sys

START = "<!-- ttlang-s3-readme:start -->"
END = "<!-- ttlang-s3-readme:end -->"


def render_markdown(text: str) -> str:
    """Render Markdown to an HTML fragment, falling back to escaped text.

    The fallback keeps the injection working when the `markdown` package is
    absent rather than failing the publish.
    """
    try:
        import markdown
    except ImportError:
        return "<pre>\n" + html.escape(text) + "\n</pre>"
    return markdown.markdown(text, extensions=["fenced_code"])


def build_block(readme_md: str) -> str:
    fragment = render_markdown(readme_md)
    return f'{START}\n<section id="ttlang-s3-readme">\n{fragment}\n</section>\n{END}\n'


def inject(index_html: str, block: str) -> str:
    """Return index_html with block placed above the anchor links.

    Any previously injected block is removed first (idempotent). The block is
    inserted immediately after the opening <body> tag so it precedes every <a>.
    """
    existing = re.compile(re.escape(START) + ".*?" + re.escape(END) + r"\n?", re.DOTALL)
    index_html = existing.sub("", index_html)

    body = re.search(r"<body[^>]*>", index_html, re.IGNORECASE)
    if body:
        cut = body.end()
        return index_html[:cut] + "\n" + block + index_html[cut:]
    return block + index_html


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(
            "Usage: inject_s3_index_readme.py <readme.md> <index.html>", file=sys.stderr
        )
        return 2
    readme_path, index_path = argv[1], argv[2]

    with open(readme_path, encoding="utf-8") as handle:
        readme_md = handle.read()
    with open(index_path, encoding="utf-8") as handle:
        index_html = handle.read()

    result = inject(index_html, build_block(readme_md))

    with open(index_path, "w", encoding="utf-8") as handle:
        handle.write(result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
