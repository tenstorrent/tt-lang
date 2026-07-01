# tt-lang internal S3 PyPI index

This is Tenstorrent's internal PyPI index for pre-release `tt-lang` wheels. Public
releases are on [PyPI](https://pypi.org/project/tt-lang/); this index hosts the
more frequently updated development builds.

## Install

Pick a published version from the workflow summary or the package list below, then
select it explicitly (public PyPI also hosts `tt-lang`):

Nightly development wheels are grouped by year-month:

```bash
pip install \
  --extra-index-url https://pypi.eng.aws.tenstorrent.com/<YYYY-MM>/ \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "tt-lang==<version>"
```

Light wheels built and device-tested against a specific tt-metal commit are grouped
by that commit's 7-character prefix. Use these when the environment already provides
`ttnn` from an external tt-metal at that commit:

```bash
pip install \
  --extra-index-url https://pypi.eng.aws.tenstorrent.com/<ttmetal7>/ \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "tt-lang-light==<version>"
```

## More

- [Getting started](https://github.com/tenstorrent/tt-lang/blob/main/docs/sphinx/getting-started.md)
  (internal S3 wheels section)
- [Documentation](https://docs.tenstorrent.com/tt-lang/)
