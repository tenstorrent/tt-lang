# Contributing guidelines for TT-Lang

Thank you for your interest in contributing to tt-lang! We appreciate your support.

## Before You Submit a PR

### Pre-commit Hooks (Required)

All contributors **must** run pre-commit hooks before submitting a pull request. This ensures code quality and consistency across the project.

#### Installation

Install pre-commit using pip:

```bash
pip install pre-commit
```

Or using your system package manager:

```bash
# macOS
brew install pre-commit

# Ubuntu/Debian
sudo apt install pre-commit
```

#### Setup

After cloning the repository, install the git hook scripts:

```bash
cd /path/to/tt-lang
pre-commit install
```

This configures git to automatically run pre-commit checks before each commit.

#### What Gets Checked

The pre-commit hooks automatically run the following checks:

- **Black** - Python code formatting (PEP 8 compliant)
- **clang-format** - C++ code formatting (LLVM style)
- **Trailing whitespace** - Removes trailing whitespace
- **End of file fixer** - Ensures files end with a newline
- **Large files check** - Prevents accidentally committing large files
- **YAML/TOML syntax** - Validates configuration files
- **Copyright headers** - Verifies proper SPDX copyright notices

#### Running Pre-commit

Once installed, pre-commit runs automatically on `git commit`. You can also run it manually:

```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files path/to/file1.py path/to/file2.cpp
```

If pre-commit makes changes, review them, stage the changes, and commit again:

```bash
git add -u
git commit -m "Your commit message"
```

For more information, visit the [pre-commit website](https://pre-commit.com/).

## PR Guidelines

### Community contributions

For all PRs, we have an internal policy listed below which your PR will go through after an initial review has been done.

The initial review will encompass the following:
* Reviewing the PR for CI/CD readiness, making sure that the code and PR at a high level make sense for the project.
* Once approved for CI/CD readiness, a Tenstorrent developer will kick off our CI/CD pipeline on your behalf.

**Note:** Your PR must pass all pre-commit checks. CI will reject PRs that do not meet these standards.

### Internal contributions
For internal contributions we have the following guidelines:

* A 24 hour merge rule exists. The rule is to wait at least 24 hours since the PR was initially opened for review. This gives members of our teams that span the globe opportunity to provide feedback to PRs.

In addition to the 24 hour rule, the following prerequisites for landing a PR exist:
* At least 1 reviewer signs off on the change
* Component owners sign-offs (GitHub will tell you if this hasn't been met)
* Green CI
* Rebasing or further changes to the PR do not reset the 24 hour counter.
