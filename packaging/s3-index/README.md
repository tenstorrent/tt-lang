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
  --extra-index-url https://pypi.eng.aws.tenstorrent.com/tt-lang/<YYYY-MM>/ \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "tt-lang==<version>"
```

Light wheels built and device-tested against a specific tt-metal commit are
published as browsable directories under `tt-lang/ttmetal/<ttmetal7>/`, where
`<ttmetal7>` is that commit's 7-character prefix. Browse the set at
`https://pypi.eng.aws.tenstorrent.com/tt-lang/ttmetal/` (trailing slash
required). Install with `--find-links` pointed at the per-SHA directory:

```bash
pip install \
  --find-links https://pypi.eng.aws.tenstorrent.com/tt-lang/ttmetal/<ttmetal7>/ \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "tt-lang-light==<version>"
```

## Verify and run

Confirm the version and the source revisions the wheel was built from. Include
`build_info()` when filing issues -- it reports the tt-lang commit and the
tt-metal tag and commit the wheel was built against, which pins down the exact
tt-metal a per-commit light wheel targets:

```bash
python -c 'import ttnn, ttl; print(ttnn.__file__, ttl.__version__)'
python -c 'import ttl; print(ttl.build_info())'
```

`tt-lang-setup` copies the tutorials into the environment; run one to check the
install end to end:

```bash
tt-lang-setup
python tutorials/elementwise/step_4_multinode_grid_full.py
```

For the full walkthrough, including configuring an external tt-metal for light
wheels, see the [tt-lang documentation](https://docs.tenstorrent.com/tt-lang) and
the getting-started guide linked below.

## More

- [Getting started](https://github.com/tenstorrent/tt-lang/blob/main/docs/sphinx/getting-started.md)
  (internal S3 wheels section)
- [Documentation](https://docs.tenstorrent.com/tt-lang/)
