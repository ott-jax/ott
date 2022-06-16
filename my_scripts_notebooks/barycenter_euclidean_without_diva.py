import sys
sys.path.insert(1, '/Users/ersi/Documents/ott/')

import jax
import jax.numpy as jnp
import sklearn
from sklearn import datasets

from ott.core import bar_problems, continuous_barycenter
from ott.geometry.costs import Euclidean, Bures

# Set problem size
n_max = 20
dim = 2
N = 3
n = [jax.random.randint(jax.random.PRNGKey(x),
                        shape=(1,), minval=20, maxval=n_max) for x in range(N)]
n = jnp.concatenate(n)

seed = 42 # added seed for reproducible data
data = sklearn.datasets.make_blobs(n_samples=n, n_features=dim, 
                                centers=None, shuffle=False, random_state=seed)
y = jnp.array(data[0])
labels = jnp.array(data[1])

bar_p = bar_problems.BarycenterProblem(
      y, num_per_segment=n, num_segments=N, max_measure_size=n_max,
      # cost_fn= Bures(dimension=1), # for 1-dimensional Gaussians
      cost_fn= Euclidean(),
      epsilon=0.01)

solver = continuous_barycenter.WassersteinBarycenter()

bar_size=10
out = solver(bar_p, bar_size=bar_size)

print(y)
