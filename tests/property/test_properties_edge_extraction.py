"""Property-based tests for edge extraction."""

import numpy as np
import pytest
from hypothesis import given, strategies as st

from src.edge_extractor import EdgeExtractor


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


# Property 9: Edge Extraction is Subtraction
@given(
    reference=image_array(),
)
def test_property_9_edge_extraction_is_subtraction(reference):
    """
    Feature: image-sharpness-enhancement
    Property 9: Edge Extraction is Subtraction
    
    For any original image and smoothed image of the same dimensions, 
    the edge image should equal the original minus the smoothed image 
    at every pixel.
    
    Validates: Requirements 3.1
    """
    # Create a slightly smoothed version (just add small random noise)
    smoothed = reference + np.random.randn(*reference.shape).astype(np.float32) * 5
    
    extractor = EdgeExtractor()
    edge = extractor.extract(reference, smoothed)
    
    # Should be exactly reference - smoothed
    expected_edge = reference - smoothed
    assert np.allclose(edge, expected_edge, atol=1e-5), \
        "Edge extraction doesn't match simple subtraction"



# Property 10: Edge Extraction Preserves Data Type
@given(
    reference=image_array(),
)
def test_property_10_edge_extraction_preserves_data_type(reference):
    """
    Feature: image-sharpness-enhancement
    Property 10: Edge Extraction Preserves Data Type
    
    For any two images with the same data type, the edge extraction 
    should produce an output with the same data type.
    
    Validates: Requirements 3.2
    """
    # Create smoothed version
    smoothed = reference * 0.9  # Slightly darker
    
    extractor = EdgeExtractor()
    edge = extractor.extract(reference, smoothed)
    
    # Should be float32 (as per implementation)
    assert edge.dtype == np.float32, \
        f"Expected float32, got {edge.dtype}"


# Property 11: Edge Extraction Preserves Negative Values
@given(
    reference=image_array(),
)
def test_property_11_edge_extraction_preserves_negative_values(reference):
    """
    Feature: image-sharpness-enhancement
    Property 11: Edge Extraction Preserves Negative Values
    
    For any original and smoothed images where smoothed has larger values 
    at some pixels, the edge image should contain negative values at those 
    pixels without clipping.
    
    Validates: Requirements 3.3
    """
    # Create smoothed version that's brighter in some areas
    smoothed = reference + 50.0  # Add constant to make it brighter
    
    extractor = EdgeExtractor()
    edge = extractor.extract(reference, smoothed)
    
    # Should have negative values (reference - smoothed will be negative)
    assert np.any(edge < 0), \
        "Edge image should contain negative values when smoothed > reference"
    
    # Check that negative values are preserved (not clipped to 0)
    expected_negative = reference - smoothed
    assert np.allclose(edge, expected_negative, atol=1e-5), \
        "Negative values not preserved correctly"
