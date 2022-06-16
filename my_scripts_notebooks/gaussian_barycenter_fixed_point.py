import sys
sys.path.insert(1, '/Users/ersi/Documents/ott/')
import jax
import jax.numpy as jnp
from sklearn.datasets import make_spd_matrix

from ott.geometry.costs import Euclidean, Bures

d = 2  # dimensionality
seed = 42 # seed for reproducible results
N = 3 # number of Gaussians to be averaged by the barycenter

weights = jax.random.uniform(key=jax.random.PRNGKey(seed), shape=(N,))
weights = weights / jnp.sum(weights)

for i in range(N):
    mean = jax.random.uniform(key=jax.random.PRNGKey(seed), shape=(1, d))
    cov = jnp.asarray(make_spd_matrix(n_dim=d, random_state=seed))
    pointcloud = jnp.concatenate((mean, jnp.reshape(cov, (-1, d*d))), axis=1)
    # print(pointcloud.shape)
    if i == 0:
        pointclouds = pointcloud
    else:
        pointclouds = jnp.vstack((pointclouds, pointcloud))
    print(pointclouds.shape)

my_bures = Bures(dimension=d)

bary = my_bures.barycenter(weights=weights, xs=pointclouds)

bary_mean = bary[:d]
bary_cov = jnp.reshape(bary[d:], (d, d))
print(bary_mean)

print(bary_cov)
