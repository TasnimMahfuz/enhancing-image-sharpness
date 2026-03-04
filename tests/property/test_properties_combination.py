"""
Property-based tests for ImageCombiner class.

Tests universal properties that should hold across all valid inputs,
using hypothesis for automated test case generation.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from src.image_combiner import ImageCombiner


class TestImageCombinationProperties:
    """Property tests for image combination."""
    
    @given(
        height=st.integers(min_value=10, max_value=100),
        width=st.integers(min_value=10, max_value=100),
        channels=st.sampled_from([None, 3])  # None for grayscale, 3 for color
    )
    @settings(max_examples=100)
    def test_property_17_combination_is_addition(self, height, width, channels):
        """
        Feature: image-sharpness-enhancement
        Property 17: Image Combination is Addition
        
        For any blurred image and enhanced edge image of the same dimensions,
        the combined image (before clipping) should equal their pixel-wise sum.
        Validates: Requirements 6.1
        """
        combiner = ImageCombiner()
        
        # Create test images
        if channels is None:
            # Grayscale
            blurred = np.random.randint(0, 256, (height, width), dtype=np.uint8)
            enhanced_edge = np.random.randn(height, width).astype(np.float32) * 10
        else:
            # Color
            blurred = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
            enhanced_edge = np.random.randn(height, width, channels).astype(np.float32) * 10
        
        # Compute expected result (before clipping)
        expected_before_clip = blurred.astype(np.float32) + enhanced_edge.astype(np.float32)
        
        # Get actual result
        sharpened = combiner.combine(blurred, enhanced_edge)
        
        # Verify that the combination equals addition (accounting for clipping)
        # For pixels where expected is in [0, 255], result should match
        # For pixels outside range, result should be clipped
        expected_clipped = np.clip(expected_before_clip, 0, 255).astype(np.uint8)
        
        np.testing.assert_array_equal(sharpened, expected_clipped,
            err_msg="Combined image should equal clipped pixel-wise sum")
    
    @given(
        height=st.integers(min_value=10, max_value=100),
        width=st.integers(min_value=10, max_value=100),
        channels=st.sampled_from([None, 3])
    )
    @settings(max_examples=100)
    def test_property_18_combination_clips_to_valid_range(self, height, width, channels):
        """
        Feature: image-sharpness-enhancement
        Property 18: Image Combination Clips to Valid Range
        
        For any combined image, all pixel values should be in the range [0, 255]
        after combination.
        Validates: Requirements 6.2
        """
        combiner = ImageCombiner()
        
        # Create test images with potential for overflow/underflow
        if channels is None:
            # Grayscale
            blurred = np.random.randint(0, 256, (height, width), dtype=np.uint8)
            # Create enhanced_edge that will cause some values to go outside [0, 255]
            enhanced_edge = np.random.randn(height, width).astype(np.float32) * 100
        else:
            # Color
            blurred = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
            enhanced_edge = np.random.randn(height, width, channels).astype(np.float32) * 100
        
        # Combine images
        sharpened = combiner.combine(blurred, enhanced_edge)
        
        # Verify all pixels are in valid range
        assert np.all(sharpened >= 0), "All pixels should be >= 0"
        assert np.all(sharpened <= 255), "All pixels should be <= 255"
        
        # Verify dtype is uint8
        assert sharpened.dtype == np.uint8, f"Output should be uint8, got {sharpened.dtype}"
    
    @given(
        height=st.integers(min_value=10, max_value=100),
        width=st.integers(min_value=10, max_value=100),
        channels=st.sampled_from([None, 3])
    )
    @settings(max_examples=100)
    def test_property_19_combination_preserves_dimensions(self, height, width, channels):
        """
        Feature: image-sharpness-enhancement
        Property 19: Image Combination Preserves Dimensions
        
        For any two images of the same dimensions, the combined image should 
        have the same dimensions.
        Validates: Requirements 6.3
        """
        combiner = ImageCombiner()
        
        # Create test images
        if channels is None:
            # Grayscale
            blurred = np.random.randint(0, 256, (height, width), dtype=np.uint8)
            enhanced_edge = np.random.randn(height, width).astype(np.float32) * 10
            expected_shape = (height, width)
        else:
            # Color
            blurred = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
            enhanced_edge = np.random.randn(height, width, channels).astype(np.float32) * 10
            expected_shape = (height, width, channels)
        
        # Combine images
        sharpened = combiner.combine(blurred, enhanced_edge)
        
        # Verify dimensions are preserved
        assert sharpened.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {sharpened.shape}"
        assert sharpened.shape == blurred.shape, \
            "Output shape should match input shape"
    
    @given(
        height=st.integers(min_value=10, max_value=50),
        width=st.integers(min_value=10, max_value=50)
    )
    @settings(max_examples=50)
    def test_mismatched_dimensions_raises_error(self, height, width):
        """
        Property: Combining images with mismatched dimensions should raise ValueError.
        
        This ensures proper validation of inputs.
        """
        combiner = ImageCombiner()
        
        # Create images with different dimensions
        blurred = np.random.randint(0, 256, (height, width), dtype=np.uint8)
        enhanced_edge = np.random.randn(height + 5, width + 5).astype(np.float32)
        
        # Should raise ValueError
        with pytest.raises(ValueError, match="Image dimensions must match"):
            combiner.combine(blurred, enhanced_edge)
    
    @given(
        height=st.integers(min_value=10, max_value=50),
        width=st.integers(min_value=10, max_value=50),
        channels=st.sampled_from([None, 3])
    )
    @settings(max_examples=50)
    def test_handles_all_channels_independently(self, height, width, channels):
        """
        Property: Image combination should handle all color channels independently.
        
        For color images, each channel should be processed independently.
        Validates: Requirements 6.4
        """
        if channels is None:
            # Skip for grayscale
            return
        
        combiner = ImageCombiner()
        
        # Create color image
        blurred = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
        enhanced_edge = np.random.randn(height, width, channels).astype(np.float32) * 10
        
        # Combine entire image
        sharpened_full = combiner.combine(blurred, enhanced_edge)
        
        # Combine each channel independently
        sharpened_channels = np.zeros_like(sharpened_full)
        for c in range(channels):
            sharpened_channels[:, :, c] = combiner.combine(
                blurred[:, :, c],
                enhanced_edge[:, :, c]
            )
        
        # Results should be identical
        np.testing.assert_array_equal(sharpened_full, sharpened_channels,
            err_msg="Full image combination should equal channel-wise combination")
