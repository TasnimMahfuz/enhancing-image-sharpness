"""Unit tests for edge extraction edge cases."""

import numpy as np
import pytest

from src.edge_extractor import EdgeExtractor


def test_edge_extraction_with_negative_values():
    """Test that edge extraction preserves negative values."""
    extractor = EdgeExtractor()
    
    # Create images where smoothed > original
    reference = np.array([[100, 150, 200],
                          [150, 200, 150],
                          [200, 150, 100]], dtype=np.float32)
    
    smoothed = np.array([[150, 200, 250],
                         [200, 250, 200],
                         [250, 200, 150]], dtype=np.float32)
    
    edge = extractor.extract(reference, smoothed)
    
    # Should have negative values
    assert np.any(edge < 0), "Should have negative values"
    
    # Check specific values
    expected = reference - smoothed
    assert np.allclose(edge, expected), "Edge values don't match expected"


def test_data_type_preservation():
    """Test that edge extraction preserves float32 data type."""
    extractor = EdgeExtractor()
    
    # Test with uint8 input
    reference = np.array([[100, 150], [200, 250]], dtype=np.uint8)
    smoothed = np.array([[90, 140], [190, 240]], dtype=np.uint8)
    
    edge = extractor.extract(reference, smoothed)
    
    # Should be converted to float32
    assert edge.dtype == np.float32, f"Expected float32, got {edge.dtype}"


def test_mismatched_shapes_raise_error():
    """Test that mismatched image shapes raise ValueError."""
    extractor = EdgeExtractor()
    
    reference = np.random.rand(10, 10).astype(np.float32)
    smoothed = np.random.rand(10, 15).astype(np.float32)  # Different width
    
    with pytest.raises(ValueError, match="Image shapes must match"):
        extractor.extract(reference, smoothed)


def test_grayscale_image():
    """Test edge extraction on grayscale images."""
    extractor = EdgeExtractor()
    
    reference = np.array([[100, 150, 200],
                          [150, 200, 150],
                          [200, 150, 100]], dtype=np.float32)
    
    smoothed = reference * 0.9  # Slightly darker
    
    edge = extractor.extract(reference, smoothed)
    
    # Should preserve shape
    assert edge.shape == reference.shape
    
    # Should be positive (reference > smoothed)
    assert np.all(edge >= 0)



def test_color_image():
    """Test edge extraction on color images."""
    extractor = EdgeExtractor()
    
    # Create 3-channel color image
    reference = np.random.randint(100, 200, (10, 10, 3), dtype=np.uint8).astype(np.float32)
    smoothed = reference * 0.8  # Darker
    
    edge = extractor.extract(reference, smoothed)
    
    # Should preserve shape including channels
    assert edge.shape == reference.shape
    assert edge.shape[2] == 3, "Should have 3 channels"


def test_zero_edge():
    """Test edge extraction when images are identical."""
    extractor = EdgeExtractor()
    
    reference = np.array([[100, 150, 200],
                          [150, 200, 150],
                          [200, 150, 100]], dtype=np.float32)
    
    smoothed = reference.copy()  # Identical
    
    edge = extractor.extract(reference, smoothed)
    
    # Should be all zeros
    assert np.allclose(edge, 0), "Edge should be zero for identical images"
