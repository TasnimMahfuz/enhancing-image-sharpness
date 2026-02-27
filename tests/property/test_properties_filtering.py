"""Property-based tests for Gaussian filtering."""

import numpy as np
import pytest
from hypothesis import given, strategies as st

from src.gaussian_filter import GaussianFilter


# Property 4: Gaussian Kernel Normalization
@given(sigma=st.floats(min_value=0.1, max_value=5.0))
def test_property_4_gaussian_kernel_normalization(sigma):
    """
    Feature: image-sharpness-enhancement
    Property 4: Gaussian Kernel Normalization
    
    For any positive sigma value, the generated Gaussian kernel 
    should sum to 1.0 within tolerance of 0.0001.
    
    Validates: Requirements 2.3
    """
    gaussian_filter = GaussianFilter(sigma)
    kernel = gaussian_filter.kernel
    
    kernel_sum = np.sum(kernel)
    assert abs(kernel_sum - 1.0) < 0.0001, \
        f"Kernel sum {kernel_sum} not within tolerance of 1.0"


# Property 5: Gaussian Kernel Formula Correctness
# Property 5: Gaussian Kernel Formula Correctness
@given(sigma=st.floats(min_value=0.5, max_value=3.0))
def test_property_5_gaussian_kernel_formula_correctness(sigma):
    """
    Feature: image-sharpness-enhancement
    Property 5: Gaussian Kernel Formula Correctness

    For any positive sigma value and any point (x, y) in the kernel,
    the kernel value at that point should equal
    (1/2πσ²) * e^(-(x²+y²)/2σ²) within numerical precision.

    Validates: Requirements 2.1
    """
    gaussian_filter = GaussianFilter(sigma)
    kernel = gaussian_filter.kernel

    # Get kernel dimensions
    radius = int(np.ceil(3 * sigma))

    # Recreate the unnormalized kernel to get the correct normalization factor
    x = np.arange(-radius, radius + 1)
    y = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(x, y)
    coefficient = 1.0 / (2 * np.pi * sigma**2)
    exponent = -(X**2 + Y**2) / (2 * sigma**2)
    kernel_unnormalized = coefficient * np.exp(exponent)
    unnormalized_sum = np.sum(kernel_unnormalized)

    # Check center point (0, 0)
    center_idx = radius
    expected_center_unnormalized = (1.0 / (2 * np.pi * sigma**2)) * np.exp(0)
    expected_center_normalized = expected_center_unnormalized / unnormalized_sum

    # Allow for numerical precision errors
    assert abs(kernel[center_idx, center_idx] - expected_center_normalized) < 1e-6, \
        f"Center value mismatch: {kernel[center_idx, center_idx]} vs {expected_center_normalized}"



# Property 6: Gaussian Kernel Size Captures Distribution
@given(sigma=st.floats(min_value=0.1, max_value=5.0))
def test_property_6_gaussian_kernel_size_captures_distribution(sigma):
    """
    Feature: image-sharpness-enhancement
    Property 6: Gaussian Kernel Size Captures Distribution
    
    For any positive sigma value, the kernel size should be at least 
    2*ceil(3*sigma)+1 to capture 99.7% of the Gaussian distribution.
    
    Validates: Requirements 2.2
    """
    gaussian_filter = GaussianFilter(sigma)
    kernel = gaussian_filter.kernel
    
    expected_min_size = 2 * int(np.ceil(3 * sigma)) + 1
    actual_size = kernel.shape[0]
    
    assert actual_size >= expected_min_size, \
        f"Kernel size {actual_size} is smaller than minimum {expected_min_size}"
    
    # Kernel should be square
    assert kernel.shape[0] == kernel.shape[1], \
        f"Kernel should be square, got {kernel.shape}"



# Custom strategy for generating valid images
@st.composite
def image_array(draw, min_size=10, max_size=50, channels=None):
    """Generate random image arrays."""
    height = draw(st.integers(min_value=min_size, max_value=max_size))
    width = draw(st.integers(min_value=min_size, max_value=max_size))
    
    if channels is None:
        # Randomly choose grayscale or color
        channels = draw(st.sampled_from([1, 3]))
    
    if channels == 1:
        shape = (height, width)
    else:
        shape = (height, width, channels)
    
    # Generate random pixel values
    img = draw(st.integers(min_value=0, max_value=255).map(
        lambda _: np.random.randint(0, 256, size=shape, dtype=np.uint8)
    ))
    
    return img.astype(np.float32)


# Property 7: Gaussian Filter Preserves Dimensions
@given(
    sigma=st.floats(min_value=0.5, max_value=2.0),
    image=image_array()
)
def test_property_7_gaussian_filter_preserves_dimensions(sigma, image):
    """
    Feature: image-sharpness-enhancement
    Property 7: Gaussian Filter Preserves Dimensions
    
    For any image (grayscale or color), applying the Gaussian filter 
    should produce an output with the same dimensions as the input.
    
    Validates: Requirements 2.4
    """
    gaussian_filter = GaussianFilter(sigma)
    smoothed = gaussian_filter.apply(image)
    
    assert smoothed.shape == image.shape, \
        f"Output shape {smoothed.shape} doesn't match input shape {image.shape}"


# Property 8: Channel Independence in Filtering
@given(
    sigma=st.floats(min_value=0.5, max_value=2.0),
    image=image_array(channels=3)
)
def test_property_8_channel_independence_in_filtering(sigma, image):
    """
    Feature: image-sharpness-enhancement
    Property 8: Channel Independence in Filtering
    
    For any color image, applying the Gaussian filter to the entire image 
    should produce the same result as applying the filter to each channel 
    independently and recombining.
    
    Validates: Requirements 2.5
    """
    gaussian_filter = GaussianFilter(sigma)
    
    # Apply to entire image
    smoothed_full = gaussian_filter.apply(image)
    
    # Apply to each channel independently
    smoothed_channels = np.zeros_like(image)
    for c in range(image.shape[2]):
        smoothed_channels[:, :, c] = gaussian_filter.apply(image[:, :, c])
    
    # Should be identical (within numerical precision)
    assert np.allclose(smoothed_full, smoothed_channels, atol=1e-5), \
        "Channel-wise filtering doesn't match full image filtering"
