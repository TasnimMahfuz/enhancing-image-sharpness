"""Property-based tests for quality metrics calculation.

Feature: image-sharpness-enhancement
Tests Properties 22-24 from the design document.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings

from src.metrics_calculator import MetricsCalculator


# Strategy for generating valid images
@st.composite
def image_array(draw, min_size=10, max_size=100, channels=None):
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
    img = np.random.randint(0, 256, shape, dtype=np.uint8)
    
    return img


@given(
    image1=image_array(),
    image2=image_array()
)
@settings(max_examples=100, deadline=None)
def test_property_22_plcc_range_validity(image1, image2):
    """
    Feature: image-sharpness-enhancement
    Property 22: PLCC Range Validity
    
    For any two images, the computed PLCC value should be in the range [-1, 1].
    
    Validates: Requirements 8.1, 8.3
    """
    # Ensure images have the same shape
    if image1.shape != image2.shape:
        # Resize image2 to match image1
        if image1.ndim == 2:
            image2 = np.resize(image2, image1.shape)
        else:
            # For color images, ensure 3 channels
            if image2.ndim == 2:
                image2 = np.stack([image2] * 3, axis=-1)
            image2 = np.resize(image2, image1.shape)
    
    calculator = MetricsCalculator()
    plcc = calculator.calculate_plcc(image1, image2)
    
    assert -1.0 <= plcc <= 1.0, \
        f"PLCC value {plcc} is outside valid range [-1, 1]"


@given(
    image1=image_array(),
    image2=image_array()
)
@settings(max_examples=100, deadline=None)
def test_property_23_srocc_range_validity(image1, image2):
    """
    Feature: image-sharpness-enhancement
    Property 23: SROCC Range Validity
    
    For any two images, the computed SROCC value should be in the range [-1, 1].
    
    Validates: Requirements 8.2, 8.4
    """
    # Ensure images have the same shape
    if image1.shape != image2.shape:
        # Resize image2 to match image1
        if image1.ndim == 2:
            image2 = np.resize(image2, image1.shape)
        else:
            # For color images, ensure 3 channels
            if image2.ndim == 2:
                image2 = np.stack([image2] * 3, axis=-1)
            image2 = np.resize(image2, image1.shape)
    
    calculator = MetricsCalculator()
    srocc = calculator.calculate_srocc(image1, image2)
    
    assert -1.0 <= srocc <= 1.0, \
        f"SROCC value {srocc} is outside valid range [-1, 1]"


@given(
    color_image=image_array(channels=3)
)
@settings(max_examples=100, deadline=None)
def test_property_24_metrics_use_grayscale(color_image):
    """
    Feature: image-sharpness-enhancement
    Property 24: Metrics Use Grayscale
    
    For any color image pair, computing metrics on the color images should
    produce the same result as computing metrics on their grayscale conversions.
    
    Validates: Requirements 8.3, 8.4, 8.5
    """
    # Create a second color image
    color_image2 = np.random.randint(0, 256, color_image.shape, dtype=np.uint8)
    
    calculator = MetricsCalculator()
    
    # Convert to grayscale manually
    gray1 = calculator._to_grayscale(color_image)
    gray2 = calculator._to_grayscale(color_image2)
    
    # Calculate metrics on color images
    plcc_color = calculator.calculate_plcc(color_image, color_image2)
    srocc_color = calculator.calculate_srocc(color_image, color_image2)
    
    # Calculate metrics on grayscale images
    plcc_gray = calculator.calculate_plcc(gray1, gray2)
    srocc_gray = calculator.calculate_srocc(gray1, gray2)
    
    # They should be equal (within numerical precision)
    assert abs(plcc_color - plcc_gray) < 1e-10, \
        f"PLCC differs: color={plcc_color}, gray={plcc_gray}"
    assert abs(srocc_color - srocc_gray) < 1e-10, \
        f"SROCC differs: color={srocc_color}, gray={srocc_gray}"
