"""positronixlab: A complete framework to explore modularity in neural networks."""

from importlib.metadata import PackageNotFoundError, version as _package_version

try:
    __version__ = _package_version("parallelazy")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
