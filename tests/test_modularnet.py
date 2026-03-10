"""Basic smoke tests for the modularnet package."""

import modularnet


def test_package_importable():
    """Test that the modularnet package can be imported."""
    assert modularnet is not None


def test_version_defined():
    """Test that the package version is defined."""
    assert hasattr(modularnet, "__version__")
    assert isinstance(modularnet.__version__, str)
    assert len(modularnet.__version__) > 0
