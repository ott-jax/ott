from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

__all__ = ["get_cifar10"]


def to_fp_array(arr: Any) -> jnp.array:
  dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
  return jnp.asarray(arr, dtype=dtype)


def get_cifar10(
    *,
    sigma: float = 0.5,
    gaussian_blur_kernel_size: int = 2,
    root: str = "data/cifar10",
    batch_size: int = 1000,
    use_flip: bool = True
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5,), (0.5,)),
  ])

  # Download and load datasets
  ds = torchvision.datasets.CIFAR10(
      root=root, train=True, download=True, transform=transform
  )
  loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

  if use_flip:
    transform_flip = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomHorizontalFlip(p=1.0)
    ])

    ds_flip = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_flip
    )
    loader_flip = DataLoader(ds_flip, batch_size=batch_size, shuffle=False)
  else:
    loader_flip = []

  src_batches = []
  target_batches = []
  label_batches = []

  def psd_gaussian_blur(x: jnp.ndarray) -> jnp.ndarray:
    assert x.ndim == 4, x.shape
    n, c, h, w = x.shape
    assert h == w
    # 1D Gaussian kernel
    ys = jnp.arange(-gaussian_blur_kernel_size, gaussian_blur_kernel_size + 1)
    kernel_1d = jsp.stats.norm.pdf(ys, scale=sigma)
    kernel_1d = kernel_1d / jnp.sum(kernel_1d)  # Normalize

    # Flatten input to apply convolution
    x = x.reshape(n * c, h, w)

    def apply_separable_conv(img):
      # Convolve along rows
      img_h = jsp.signal.convolve(img, kernel_1d[None, :], mode="same")
      # Convolve along columns
      return jsp.signal.convolve(img_h, kernel_1d[:, None], mode="same")

    conv = jax.vmap(apply_separable_conv)
    blurred = conv(x)

    return blurred.reshape(n, c, h, w)

  for ld in [loader, loader_flip] if use_flip else [loader]:
    for batch_src, batch_label in ld:
      batch_src = jnp.array(batch_src.numpy())

      batch_target = psd_gaussian_blur(batch_src)

      src_batches.append(batch_src.reshape(batch_src.shape[0], -1))
      target_batches.append(batch_target.reshape(batch_target.shape[0], -1))
      label_batches.append(jnp.array(batch_label.numpy()))

  src = jnp.concatenate(src_batches, axis=0)
  target = jnp.concatenate(target_batches, axis=0)
  labels = jnp.concatenate(label_batches, axis=0)

  return to_fp_array(src), to_fp_array(target), labels
