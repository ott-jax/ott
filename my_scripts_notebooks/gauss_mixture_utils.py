import numpy as np
import jax.numpy as jnp

def gauss_mixture_mean_and_cov_to_pointcloud(means, covariances, dimension, num_components):
    """Helper function to obtain pointcloud representation from means and covariances of a GMM.
       Args:
            means: jnp.ndarray (num_points x dimension)
            covariances: jnp.ndarray (num_points x dimension x dimension)
            dimension: the dimension of the Gaussians
            num_components: the number of components in the GMM
        Returns:
            pointcloud: jnp.nndarray (num_points x d + d^2)
    """
    pointcloud = jnp.asarray([jnp.concatenate((means[i], jnp.reshape(covariances[i], (dimension * dimension, )))) for i in range(num_components)])
    return pointcloud

def gauss_mixture_pointcloud_to_mean_and_cov(pointcloud, dimension, num_components):
    """Helper function to obtain the means and covariances from a pointcloud of a GMM.
        Args:
            pointcloud: jnp.nndarray (num_points x d + d^2)
            dimension: the dimension of the Gaussians
            num_components: the number of components in the GMM
        Returns:
            means: jnp.ndarray (num_points x dimension)
            covariances: jnp.ndarray (num_points x dimension x dimension)
    """
    means = jnp.asarray([pointcloud[i][0:dimension] for i in range(num_components)])
    covariances = jnp.asarray([jnp.reshape(pointcloud[i][dimension:dimension+dimension**2], (dimension, dimension)) for i in range(num_components)])
    return means, covariances
