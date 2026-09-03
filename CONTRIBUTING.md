# Contributing to OTT
We'd love to accept your contributions to this project.

There are many ways to contribute to OTT, with the most common ones being contribution of code, documentation
to the project, participating in discussions or raising issues.

## Contributing code or documentation
1. fork the repository using the [Fork](https://github.com/ott-jax/ott/fork) button on GitHub.
2. ```shell
   git clone https://github.com/<YOUR_USERNAME>/ott.git ott && \
   cd ott && \
   pip install -e '.[dev]' && \
   pre-commit install
   ```

When committing changes, sometimes you might want or need to bypass the pre-commit checks. This can be
done via the `--no-verify` flag as:
```shell
git commit --no-verify -m "<SOME_COMMIT_MESSAGE>"
```

## Running tests
Tests are run with [pytest](https://docs.pytest.org/), which requires the `test`
and `neural` extras: `pip install -e '.[test,neural]'`.
```shell
python -m pytest -n auto  # run all tests, distributed over the available cores
python -m pytest -m fast  # run only the fast tests
python -m pytest -k "test_euclidean_point_cloud"  # run tests matching the expression
python -m pytest --memray  # also check memory
python -m pytest --strict-rng  # fail if a PRNG key is consumed twice
```
The linter is run separately, via [pre-commit](https://pre-commit.com/):
```shell
pre-commit run --all-files
```

## Documentation
Building the documentation requires the `docs` and `neural` extras:
`pip install -e '.[docs,neural]'`. From the root of the repository, run:
```shell
make -C docs clean  # remove the built documentation
make -C docs html  # build the documentation
make -C docs linkcheck spelling  # lint the documentation
```
Installing `PyEnchant` is required to run spellchecker, please refer to the
[installation instructions](https://pyenchant.github.io/pyenchant/install.html). On macOS Silicon, it may be necessary
to also set `PYENCHANT_LIBRARY_PATH` environment variable, as, e.g.,
`export PYENCHANT_LIBRARY_PATH=/opt/homebrew/lib/libenchant-2.2.dylib`. False positives and correctly spelled words can
be added to one of the files in [docs/spelling](https://github.com/ott-jax/ott/tree/main/docs/spelling).

## Building the package
The package can be built using [build](https://build.pypa.io/):
```shell
python -m build --sdist --wheel --outdir dist/
twine check dist/*
```
Afterwards, the built package will be located under `dist/`.

## Code reviews
All submissions, including submissions by project members, require review. We use GitHub
[pull requests](https://github.com/ott-jax/ott/pulls) for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

# Community guidelines
We abide by the principles of openness, respect, and consideration of others of the Python Software Foundation's
[code of conduct](https://www.python.org/psf/codeofconduct/).
