"""positronixlab: A complete framework to explore modularity in neural networks."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

try:
    __version__ = _package_version("positronixlab")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"
