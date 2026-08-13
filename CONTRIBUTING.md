# Contributing to ptx-anneal

We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process

`ptx-anneal` is developed internally at Meta and mirrored to GitHub. Changes
flow from our internal repository to the public one; pull requests opened on
GitHub are imported internally, reviewed, and synced back out.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style

`ptx-anneal` is a Python project (targeting Python 3.10+). We use
[`ruff`](https://docs.astral.sh/ruff/) for linting, configured under `[tool.ruff]`
in `pyproject.toml`.

* Line length is 120 characters (enforced: `E501`, plus `I`/`UP`/`B`).
* CI gates on `ruff check` only. Formatting is **not** normalized in this tree, so
  don't run `ruff format` as a drive-by — it would reformat unrelated files.
* Every source file must carry the Meta copyright header found in the existing
  files.
* Run the tests with `pytest` before sending a pull request. They are CPU-only:
  no GPU, torch, triton or cuda-python required.

```bash
pip3 install -e ".[dev]"
ruff check .
pytest
```

## License

By contributing to `ptx-anneal`, you agree that your contributions will be
licensed under the LICENSE file in the root directory of this source tree.
