"""Setup script for image sharpness enhancement system."""

from setuptools import setup, find_packages

setup(
    name="image-sharpness-enhancement",
    version="0.1.0",
    description="Image sharpening using coefficient bounds from complex analysis",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0.0",
            "hypothesis>=6.0.0",
            "pytest-cov>=3.0.0",
        ],
    },
    python_requires=">=3.8",
)
