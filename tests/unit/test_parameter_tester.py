"""Unit tests for parameter testing functionality.

Feature: image-sharpness-enhancement
Tests Example 4 and additional unit tests for ParameterTester.
"""

import pytest
import numpy as np
from PIL import Image

from src.parameter_tester import ParameterTester
from src.data_models import TestResult


def test_example_4_papers_recommended_parameters():
    """
    Feature: image-sharpness-enhancement
    Example 4: Paper's Recommended Parameters
    
    Test t=0.6, λ=0 produces valid output with real CSIQ images.
    
    Validates: Requirements 7.5
    """
    # Load real CSIQ images
    reference_path = "datasets/CSIQ/src_imgs/1600.png"
    blurred_path = "datasets/CSIQ/dst_imgs/blur/1600.BLUR.3.png"
    
    reference = np.array(Image.open(reference_path))
    blurred = np.array(Image.open(blurred_path))
    
    # Test with paper's recommended parameters
    tester = ParameterTester()
    results = tester.test_parameters(
        blurred=blurred,
        reference=reference,
        t_values=[0.6],
        lambda_values=[0.0],
        sigma_values=[1.0]
    )
    
    # Should produce exactly one result
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    
    result = results[0]
    
    # Verify parameters match
    assert result.t == 0.6, f"Expected t=0.6, got {result.t}"
    assert result.lambda_param == 0.0, f"Expected λ=0.0, got {result.lambda_param}"
    assert result.sigma == 1.0, f"Expected σ=1.0, got {result.sigma}"
    
    # Verify output is valid
    assert result.sharpened_image is not None, "Sharpened image should not be None"
    assert result.sharpened_image.shape == blurred.shape, \
        f"Sharpened shape {result.sharpened_image.shape} != blurred shape {blurred.shape}"
    
    # Verify metrics are in valid range
    assert -1.0 <= result.plcc <= 1.0, f"PLCC {result.plcc} outside valid range"
    assert -1.0 <= result.srocc <= 1.0, f"SROCC {result.srocc} outside valid range"
    
    # Verify pixel values are in valid range
    assert np.all(result.sharpened_image >= 0), "Sharpened image has negative pixels"
    assert np.all(result.sharpened_image <= 255), "Sharpened image has pixels > 255"


def test_parameter_tester_with_multiple_parameters():
    """
    Test parameter tester with multiple parameter values.
    
    Verifies that all combinations are tested and results are returned.
    """
    # Load real CSIQ images (use smaller image for faster testing)
    reference_path = "datasets/CSIQ/src_imgs/1600.png"
    blurred_path = "datasets/CSIQ/dst_imgs/blur/1600.BLUR.1.png"
    
    reference = np.array(Image.open(reference_path))
    blurred = np.array(Image.open(blurred_path))
    
    # Test with multiple parameter values
    tester = ParameterTester()
    results = tester.test_parameters(
        blurred=blurred,
        reference=reference,
        t_values=[0.5, 0.6, 0.7],
        lambda_values=[0.0, 0.5],
        sigma_values=[1.0, 2.0]
    )
    
    # Should produce 3 * 2 * 2 = 12 results
    assert len(results) == 12, f"Expected 12 results, got {len(results)}"
    
    # Verify all combinations are present
    t_values_found = set(r.t for r in results)
    lambda_values_found = set(r.lambda_param for r in results)
    sigma_values_found = set(r.sigma for r in results)
    
    assert t_values_found == {0.5, 0.6, 0.7}, f"Missing t values: {t_values_found}"
    assert lambda_values_found == {0.0, 0.5}, f"Missing lambda values: {lambda_values_found}"
    assert sigma_values_found == {1.0, 2.0}, f"Missing sigma values: {sigma_values_found}"


def test_find_optimal_by_plcc():
    """
    Test finding optimal parameters by PLCC metric.
    """
    # Create mock test results
    results = [
        TestResult(
            t=0.5, lambda_param=0.0, sigma=1.0,
            sharpened_image=np.zeros((10, 10), dtype=np.uint8),
            plcc=0.85, srocc=0.80
        ),
        TestResult(
            t=0.6, lambda_param=0.0, sigma=1.0,
            sharpened_image=np.zeros((10, 10), dtype=np.uint8),
            plcc=0.92, srocc=0.88
        ),
        TestResult(
            t=0.7, lambda_param=0.0, sigma=1.0,
            sharpened_image=np.zeros((10, 10), dtype=np.uint8),
            plcc=0.88, srocc=0.90
        ),
    ]
    
    tester = ParameterTester()
    optimal = tester.find_optimal(results, metric='plcc')
    
    # Should return the result with highest PLCC (0.92)
    assert optimal.t == 0.6, f"Expected t=0.6, got {optimal.t}"
    assert optimal.plcc == 0.92, f"Expected PLCC=0.92, got {optimal.plcc}"


def test_find_optimal_by_srocc():
    """
    Test finding optimal parameters by SROCC metric.
    """
    # Create mock test results
    results = [
        TestResult(
            t=0.5, lambda_param=0.0, sigma=1.0,
            sharpened_image=np.zeros((10, 10), dtype=np.uint8),
            plcc=0.85, srocc=0.80
        ),
        TestResult(
            t=0.6, lambda_param=0.0, sigma=1.0,
            sharpened_image=np.zeros((10, 10), dtype=np.uint8),
            plcc=0.92, srocc=0.88
        ),
        TestResult(
            t=0.7, lambda_param=0.0, sigma=1.0,
            sharpened_image=np.zeros((10, 10), dtype=np.uint8),
            plcc=0.88, srocc=0.90
        ),
    ]
    
    tester = ParameterTester()
    optimal = tester.find_optimal(results, metric='srocc')
    
    # Should return the result with highest SROCC (0.90)
    assert optimal.t == 0.7, f"Expected t=0.7, got {optimal.t}"
    assert optimal.srocc == 0.90, f"Expected SROCC=0.90, got {optimal.srocc}"


def test_find_optimal_with_empty_results():
    """
    Test that find_optimal raises ValueError with empty results list.
    """
    tester = ParameterTester()
    
    with pytest.raises(ValueError, match="results list is empty"):
        tester.find_optimal([], metric='plcc')


def test_find_optimal_with_invalid_metric():
    """
    Test that find_optimal raises ValueError with invalid metric.
    """
    results = [
        TestResult(
            t=0.5, lambda_param=0.0, sigma=1.0,
            sharpened_image=np.zeros((10, 10), dtype=np.uint8),
            plcc=0.85, srocc=0.80
        ),
    ]
    
    tester = ParameterTester()
    
    with pytest.raises(ValueError, match="Invalid metric"):
        tester.find_optimal(results, metric='invalid')


def test_parameter_tester_handles_failures_gracefully():
    """
    Test that parameter tester continues when some combinations fail.
    
    This test uses invalid parameters (t = 1.0) to trigger failures.
    """
    # Create small test images
    reference = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    blurred = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    
    # Include some invalid t values (exactly 1.0) that will fail
    tester = ParameterTester()
    results = tester.test_parameters(
        blurred=blurred,
        reference=reference,
        t_values=[0.5, 1.0],  # 1.0 will fail due to division by zero
        lambda_values=[0.0],
        sigma_values=[1.0]
    )
    
    # Should only get 1 result (t=0.5 succeeds, t=1.0 fails)
    assert len(results) == 1, f"Expected 1 result, got {len(results)}"
    assert results[0].t == 0.5, f"Expected t=0.5, got {results[0].t}"
