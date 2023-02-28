# Contributing to OTT
We'd love to accept your contributions to this project.

There are many ways to contribute to OTT, with the most common ones being contribution of code, documentation
to the project, participating in discussions or raising issues.

## Contributing code or documentation
1. fork the repository using the **Fork** button on GitHub or the following
   [link](https://github.com/ott-jax/ott/fork)
2. ```shell
   git clone https://github.com/<YOUR_USERNAME>/ott
   cd ott
   pip install -e .'[dev]'
   pre-commit install
   ```

When committing changes, sometimes you might want or need to bypass the pre-commit checks. This can be
done via the ``--no-verify`` flag as:
```shell
git commit --no-verify -m "<SOME_COMMIT_MESSAGE>"
```

## Running tests
In order to run tests, we utilize [tox](https://tox.wiki/):
```shell
tox run  # run linter and all tests on all available Python versions
tox run -- -n auto -m fast  # run linter and fast tests in parallel
tox -e py3.8  # run all tests on Python3.8
tox -e py3.9 -- -k "test_euclidean_point_cloud"  # run tests matching the expression on Python3.9
tox -e py3.10 -- --memray  # test also memory on Python3.10
```
Alternatively, tests can be also run using the [pytest](https://docs.pytest.org/):
```shell
python -m pytest
```
This requires the ``'[test]'`` extra requirements to be installed as ``pip install -e.'[test]'``.

## Documentation
From the root of the repository, run:
```shell
tox -e build-docs  # build documentation
tox -e clean-docs  # remove documentation
tox -e lint-docs  # run spellchecker and linkchecker
```
Installing ``pyEnchant`` is required to run spellchecker, please refer to the
[installation instructions](https://pyenchant.github.io/pyenchant/install.html). On macOS Silicon, it may be necessary
to also set ``PYENCHANT_LIBRARY_PATH`` environment variable.

## Building the package
The package can be built using:
```shell
tox -e build-package
```
Afterwards, the built package will be located under ``dist/``.

## Code reviews
All submissions, including submissions by project members, require review. We use GitHub
[pull requests](https://github.com/ott-jax/ott/pulls) for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

# Community Guidelines
We abide by the principles of openness, respect, and consideration of others of the Python Software Foundation's
[code of conduct](https://www.python.org/psf/codeofconduct/).
