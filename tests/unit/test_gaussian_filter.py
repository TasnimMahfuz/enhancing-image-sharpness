"""Unit tests for Gaussian filter edge cases."""

import numpy as np
import pytest

from src.gaussian_filter import GaussianFilter


def test_very_small_sigma():
    """Test Gaussian filter with very small sigma (0.1)."""
    sigma = 0.1
    gaussian_filter = GaussianFilter(sigma)
    
    # Should create valid kernel
    assert gaussian_filter.kernel is not None
    assert gaussian_filter.kernel.sum() == pytest.approx(1.0, abs=0.0001)
    
    # Kernel should be small for small sigma
    assert gaussian_filter.kernel.shape[0] <= 5


def test_large_sigma():
    """Test Gaussian filter with large sigma (5.0)."""
    sigma = 5.0
    gaussian_filter = GaussianFilter(sigma)
    
    # Should create valid kernel
    assert gaussian_filter.kernel is not None
    assert gaussian_filter.kernel.sum() == pytest.approx(1.0, abs=0.0001)
    
    # Kernel should be large for large sigma
    assert gaussian_filter.kernel.shape[0] >= 31  # 2*ceil(3*5)+1 = 31


def test_grayscale_image():
    """Test filtering on grayscale image."""
    sigma = 1.0
    gaussian_filter = GaussianFilter(sigma)
    
    # Create simple grayscale image
    image = np.array([[100, 150, 200],
                      [150, 200, 150],
                      [200, 150, 100]], dtype=np.float32)
    
    smoothed = gaussian_filter.apply(image)
    
    # Should preserve shape
    assert smoothed.shape == image.shape
    
    # Should be smoothed (center value closer to neighbors)
    assert smoothed[1, 1] != image[1, 1]



def test_color_image():
    """Test filtering on color image."""
    sigma = 1.0
    gaussian_filter = GaussianFilter(sigma)
    
    # Create simple color image (10x10x3)
    image = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8).astype(np.float32)
    
    smoothed = gaussian_filter.apply(image)
    
    # Should preserve shape
    assert smoothed.shape == image.shape
    
    # Each channel should be smoothed
    for c in range(3):
        assert not np.array_equal(smoothed[:, :, c], image[:, :, c])


def test_kernel_normalization_accuracy():
    """Test that kernel normalization is accurate for various sigma values."""
    sigmas = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    for sigma in sigmas:
        gaussian_filter = GaussianFilter(sigma)
        kernel_sum = gaussian_filter.kernel.sum()
        
        # Should be very close to 1.0
        assert abs(kernel_sum - 1.0) < 0.0001, \
            f"Kernel sum {kernel_sum} not normalized for sigma={sigma}"


def test_invalid_sigma():
    """Test that invalid sigma values raise errors."""
    with pytest.raises(ValueError, match="sigma must be positive"):
        GaussianFilter(sigma=0)
    
    with pytest.raises(ValueError, match="sigma must be positive"):
        GaussianFilter(sigma=-1.0)


def test_invalid_image_dimensions():
    """Test that invalid image dimensions raise errors."""
    gaussian_filter = GaussianFilter(sigma=1.0)
    
    # 1D array should fail
    with pytest.raises(ValueError, match="Image must be 2D or 3D"):
        gaussian_filter.apply(np.array([1, 2, 3]))
    
    # 4D array should fail
    with pytest.raises(ValueError, match="Image must be 2D or 3D"):
        gaussian_filter.apply(np.random.rand(10, 10, 3, 3))
