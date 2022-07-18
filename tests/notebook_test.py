from pathlib import Path

import pytest
from testbook import testbook

ROOT = Path("docs/notebooks")


@pytest.mark.notebook
class TestNotebook:

  @pytest.mark.parametrize(
      "notebook", [
          "point_clouds", "Hessians", "gromov_wasserstein", "LRSinkhorn",
          "GWLRSinkhorn", "wasserstein_barycenters_gmms"
      ]
  )
  def test_notebook_regression(self, notebook: str, request):
    kernel_name = request.config.getoption("--kernel-name")
    timeout = request.config.getoption("--notebook-cell-timeout")

    if not notebook.endswith(".ipynb"):
      notebook += ".ipynb"

    with testbook(
        ROOT / notebook, execute=True, timeout=timeout, kernel_name=kernel_name
    ) as _:
      pass
