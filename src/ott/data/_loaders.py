from typing import Any, Optional

import jax
import jax.numpy as jnp
import jax.random as rand
import jax.scipy as jsp
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


__all__ = ["get_cifar10"]

def to_fp_array(arr: Any) -> jnp.array:
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    return jnp.asarray(arr, dtype=dtype)

def get_cifar10(
    *,
    sigma: float = 0.5,
    gaussian_blur_kernel_size: int = 2,
    num_points: Optional[int] = None,
    root: str = "data/cifar10",
    batch_size: int = 1000,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    def psd_gaussian_blur(x: jnp.ndarray) -> jnp.ndarray:
        assert x.ndim == 4, x.shape
        *_, num_chan = x.shape

        ys = jnp.arange(-gaussian_blur_kernel_size, gaussian_blur_kernel_size + 1)
        window = jsp.stats.norm.pdf(ys, scale=sigma) * jsp.stats.norm.pdf(
            ys[:, None], scale=sigma
        )
        window = jnp.repeat(window[..., None], num_chan, axis=-1)
        conv = jax.vmap(lambda x: jsp.signal.convolve(x, window, mode="same"))
        return conv(x)

    # Download and load datasets
    ds= torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform
    )
 
    # Process in batches instead of all at once
    subset_ds = Subset(ds, range(num_points)) if num_points is not None else ds
    loader = DataLoader(subset_ds, batch_size=batch_size, shuffle=False)
    
    src_batches = []
    target_batches = []
    label_batches = []
    
    for batch_src, batch_label in loader:
        batch_src = batch_src.numpy().squeeze()
        batch_src = np.expand_dims(batch_src, axis=-1)
        batch_src = jnp.array(batch_src)
        
        batch_target = psd_gaussian_blur(batch_src)
        
        src_batches.append(batch_src.reshape(batch_src.shape[0], -1))
        target_batches.append(batch_target.reshape(batch_target.shape[0], -1))
        label_batches.append(jnp.array(batch_label.numpy()))
    
    src = jnp.concatenate(src_batches, axis=0)
    target = jnp.concatenate(target_batches, axis=0)
    labels = jnp.concatenate(label_batches, axis=0)
    
    return to_fp_array(src), to_fp_array(target), labels