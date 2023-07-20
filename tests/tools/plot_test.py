# Copyright OTT-JAX
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import matplotlib.pyplot as plt
import ott
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
from ott.tools import plot


class TestSoftSort:

  def test_plot(self, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    n, m, d = 12, 7, 3
    rngs = jax.random.split(jax.random.PRNGKey(0), 3)
    xs = [
        jax.random.normal(rngs[0], (n, d)) + 1,
        jax.random.normal(rngs[1], (n, d)) + 1
    ]
    y = jax.random.uniform(rngs[2], (m, d))

    solver = sinkhorn.Sinkhorn()
    ots = [
        solver(linear_problem.LinearProblem(pointcloud.PointCloud(x, y)))
        for x in xs
    ]

    plott = plot.Plot()
    _ = plott(ots[0])
    fig = plt.figure(figsize=(8, 5))
    plott = ott.tools.plot.Plot(fig=fig, title="test")
    plott.animate(ots, frame_rate=2, titles=["test1", "test2"])
