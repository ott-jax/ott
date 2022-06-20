# Contributing to OTT

We'd love to accept your contributions to this project.

There are many ways to contribute to OTT, with the most common ones being contribution of code, documentation
to the project, participating in discussions or raising issues.

## Contributing code or documentation
1. fork the repository using the **Fork** button on GitHub or the following
   [link](https://github.com/ott-jax/ott/fork)
2. ```bash
   git clone https://github.com/YOUR_USERNAME/ott-jax
   cd ott-jax
   pip install -e .'[dev,test]'
   pre-commit install
   ```

When committing changes, sometimes you might want or need to bypass the pre-commit checks. This can be
done via the ``--no-verify`` flag as:
```bash
git commit --no-verify -m "The commit message"
```

## Running tests
In order to run tests, you can:
```bash
pytest -n auto  # automatic number of jobs
pytest tests/sinkhorn_test.py  # only test within a specific file
pytest -k "test_euclidean_point_cloud"  # only tests which contain the expression
```

## Building documentation
From the root of the repository, run:
```bash
pip install -e.'[docs]'
cd docs
make html  # use `-j 4` to run using 4 jobs
<your favorite browser> _build/html/index.html
# run `make clean` to remove generated files
```

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more information on using pull requests.

# Community Guidelines
We abide by the principles of openness, respect, and consideration of others of the Python Software Foundation's
[code of conduct](https://www.python.org/psf/codeofconduct/).
