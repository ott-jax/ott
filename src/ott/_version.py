from importlib.metadata import PackageNotFoundError, version

try:
  __version__ = version("ott-jax")
except PackageNotFoundError:
  __version__ = ""

del version, PackageNotFoundError
