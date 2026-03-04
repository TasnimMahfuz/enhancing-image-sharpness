"""Unit tests for MetricsCalculator class.

Feature: image-sharpness-enhancement
Tests edge cases and specific examples for quality metrics calculation.
"""

import pytest
import numpy as np

from src.metrics_calculator import MetricsCalculator


class TestMetricsCalculator:
    """Test suite for MetricsCalculator class."""
    
    def test_identical_images_plcc(self):
        """
        Test PLCC with identical images.
        
        When two images are identical, PLCC should be 1.0 (perfect correlation).
        
        Validates: Requirements 8.1
        """
        calculator = MetricsCalculator()
        
        # Create identical images
        image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        plcc = calculator.calculate_plcc(image, image)
        
        assert abs(plcc - 1.0) < 1e-10, \
            f"PLCC for identical images should be 1.0, got {plcc}"
    
    def test_identical_images_srocc(self):
        """
        Test SROCC with identical images.
        
        When two images are identical, SROCC should be 1.0 (perfect rank correlation).
        
        Validates: Requirements 8.2
        """
        calculator = MetricsCalculator()
        
        # Create identical images
        image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        srocc = calculator.calculate_srocc(image, image)
        
        assert abs(srocc - 1.0) < 1e-10, \
            f"SROCC for identical images should be 1.0, got {srocc}"
    
    def test_constant_images_plcc(self):
        """
        Test PLCC with constant images.
        
        When both images are constant (all pixels same value), PLCC should be 0.0
        because standard deviation is zero.
        
        Validates: Requirements 8.1
        """
        calculator = MetricsCalculator()
        
        # Create constant images
        image1 = np.full((50, 50), 100, dtype=np.uint8)
        image2 = np.full((50, 50), 150, dtype=np.uint8)
        
        plcc = calculator.calculate_plcc(image1, image2)
        
        assert abs(plcc - 0.0) < 1e-10, \
            f"PLCC for constant images should be 0.0, got {plcc}"
    
    def test_constant_images_srocc(self):
        """
        Test SROCC with constant images.
        
        When both images are constant, all ranks are identical (all zeros),
        so SROCC should be 1.0 (perfect rank correlation).
        
        Validates: Requirements 8.2
        """
        calculator = MetricsCalculator()
        
        # Create constant images
        image1 = np.full((50, 50), 100, dtype=np.uint8)
        image2 = np.full((50, 50), 150, dtype=np.uint8)
        
        srocc = calculator.calculate_srocc(image1, image2)
        
        # When all values are the same, all ranks are 0, 1, 2, ... n-1 in order
        # So both rank arrays are identical, giving SROCC = 1.0
        assert abs(srocc - 1.0) < 1e-10, \
            f"SROCC for constant images should be 1.0, got {srocc}"
    
    def test_grayscale_conversion_accuracy(self):
        """
        Test grayscale conversion formula accuracy.
        
        Verify that the grayscale conversion uses the correct formula:
        gray = 0.299*R + 0.587*G + 0.114*B
        
        Validates: Requirements 8.5
        """
        calculator = MetricsCalculator()
        
        # Create a simple color image with known values
        color_image = np.array([
            [[100, 150, 200], [50, 100, 150]],
            [[200, 100, 50], [0, 0, 0]]
        ], dtype=np.uint8)
        
        gray = calculator._to_grayscale(color_image)
        
        # Calculate expected values manually
        expected = np.array([
            [0.299*100 + 0.587*150 + 0.114*200, 0.299*50 + 0.587*100 + 0.114*150],
            [0.299*200 + 0.587*100 + 0.114*50, 0.299*0 + 0.587*0 + 0.114*0]
        ])
        
        np.testing.assert_allclose(gray, expected, rtol=1e-10, atol=1e-10,
                                   err_msg="Grayscale conversion formula incorrect")
    
    def test_grayscale_image_unchanged(self):
        """
        Test that grayscale images are returned unchanged.
        
        When input is already grayscale, _to_grayscale should return it as-is.
        
        Validates: Requirements 8.5
        """
        calculator = MetricsCalculator()
        
        # Create grayscale image
        gray_image = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        
        result = calculator._to_grayscale(gray_image)
        
        np.testing.assert_array_equal(result, gray_image,
                                      err_msg="Grayscale image should be unchanged")
    
    def test_color_image_metrics(self):
        """
        Test metrics calculation with color images.
        
        Verify that metrics can be calculated on color images and produce
        valid results.
        
        Validates: Requirements 8.1, 8.2, 8.5
        """
        calculator = MetricsCalculator()
        
        # Create color images
        image1 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        image2 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        plcc = calculator.calculate_plcc(image1, image2)
        srocc = calculator.calculate_srocc(image1, image2)
        
        # Verify results are in valid range
        assert -1.0 <= plcc <= 1.0, f"PLCC {plcc} outside valid range"
        assert -1.0 <= srocc <= 1.0, f"SROCC {srocc} outside valid range"
    
    def test_perfectly_negatively_correlated_images(self):
        """
        Test PLCC with perfectly negatively correlated images.
        
        When one image is the inverse of another, PLCC should be close to -1.0.
        
        Validates: Requirements 8.1
        """
        calculator = MetricsCalculator()
        
        # Create image and its inverse
        image1 = np.arange(0, 256, dtype=np.uint8).reshape(16, 16)
        image2 = 255 - image1
        
        plcc = calculator.calculate_plcc(image1, image2)
        
        # Should be very close to -1.0
        assert plcc < -0.99, \
            f"PLCC for negatively correlated images should be close to -1.0, got {plcc}"
    
    def test_invalid_image_shape(self):
        """
        Test that invalid image shapes raise appropriate errors.
        
        Images with wrong number of dimensions should raise ValueError.
        
        Validates: Requirements 8.5
        """
        calculator = MetricsCalculator()
        
        # Create 1D array (invalid)
        invalid_image = np.array([1, 2, 3, 4, 5])
        
        with pytest.raises(ValueError, match="Image must be 2D.*or 3D"):
            calculator._to_grayscale(invalid_image)
    
    def test_metrics_with_different_value_ranges(self):
        """
        Test metrics with images having different value ranges.
        
        Verify that metrics work correctly even when images have very different
        pixel value distributions.
        
        Validates: Requirements 8.1, 8.2
        """
        calculator = MetricsCalculator()
        
        # Create images with different value ranges
        image1 = np.random.randint(0, 50, (50, 50), dtype=np.uint8)  # Dark image
        image2 = np.random.randint(200, 256, (50, 50), dtype=np.uint8)  # Bright image
        
        plcc = calculator.calculate_plcc(image1, image2)
        srocc = calculator.calculate_srocc(image1, image2)
        
        # Should produce valid results
        assert -1.0 <= plcc <= 1.0, f"PLCC {plcc} outside valid range"
        assert -1.0 <= srocc <= 1.0, f"SROCC {srocc} outside valid range"
