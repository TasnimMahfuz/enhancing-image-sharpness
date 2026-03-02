"""Property-based tests for parameter testing functionality.

Feature: image-sharpness-enhancement
Tests Properties 20-21 from the design document.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from src.parameter_tester import ParameterTester


# Strategy for generating valid images
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
    img = np.random.randint(0, 256, shape, dtype=np.uint8)
    
    return img


# Strategy for generating valid parameter lists
@st.composite
def parameter_lists(draw):
    """Generate lists of valid parameters for testing."""
    # Generate small lists to keep test time reasonable
    # Avoid t values very close to 1.0 (causes division by zero in u_n calculation)
    t_values = draw(st.lists(
        st.floats(min_value=0.0, max_value=0.95),
        min_size=1,
        max_size=3,
        unique=True
    ))
    
    lambda_values = draw(st.lists(
        st.floats(min_value=-2.0, max_value=2.0),
        min_size=1,
        max_size=3,
        unique=True
    ))
    
    sigma_values = draw(st.lists(
        st.floats(min_value=0.5, max_value=3.0),
        min_size=1,
        max_size=3,
        unique=True
    ))
    
    return t_values, lambda_values, sigma_values


@given(
    blurred=image_array(min_size=20, max_size=40),
    params=parameter_lists()
)
@settings(max_examples=50, deadline=None)
def test_property_20_parameter_testing_generates_all_combinations(blurred, params):
    """
    Feature: image-sharpness-enhancement
    Property 20: Parameter Testing Generates All Combinations
    
    For any lists of t, lambda, and sigma parameters, the number of test results
    should equal len(t_list) × len(lambda_list) × len(sigma_list).
    
    Validates: Requirements 7.3
    """
    t_values, lambda_values, sigma_values = params
    
    # Create reference image (same as blurred for simplicity)
    reference = blurred.copy()
    
    # Test parameters
    tester = ParameterTester()
    results = tester.test_parameters(
        blurred=blurred,
        reference=reference,
        t_values=t_values,
        lambda_values=lambda_values,
        sigma_values=sigma_values
    )
    
    # Calculate expected number of combinations
    expected_count = len(t_values) * len(lambda_values) * len(sigma_values)
    
    # All combinations should succeed (no invalid parameters)
    assert len(results) == expected_count, \
        f"Expected {expected_count} results, got {len(results)}"


@given(
    blurred=image_array(min_size=20, max_size=40),
    params=parameter_lists()
)
@settings(max_examples=50, deadline=None)
def test_property_21_test_results_store_parameters(blurred, params):
    """
    Feature: image-sharpness-enhancement
    Property 21: Test Results Store Parameters
    
    For any test result, the stored parameters should match the parameters
    used to generate that result.
    
    Validates: Requirements 7.4
    """
    t_values, lambda_values, sigma_values = params
    
    # Create reference image (same as blurred for simplicity)
    reference = blurred.copy()
    
    # Test parameters
    tester = ParameterTester()
    results = tester.test_parameters(
        blurred=blurred,
        reference=reference,
        t_values=t_values,
        lambda_values=lambda_values,
        sigma_values=sigma_values
    )
    
    # Verify each result stores the correct parameters
    for result in results:
        # Check that stored parameters are from the input lists
        assert result.t in t_values, \
            f"Result t={result.t} not in input t_values={t_values}"
        assert result.lambda_param in lambda_values, \
            f"Result lambda={result.lambda_param} not in input lambda_values={lambda_values}"
        assert result.sigma in sigma_values, \
            f"Result sigma={result.sigma} not in input sigma_values={sigma_values}"
        
        # Check that result has valid metrics
        assert -1.0 <= result.plcc <= 1.0, \
            f"PLCC {result.plcc} outside valid range"
        assert -1.0 <= result.srocc <= 1.0, \
            f"SROCC {result.srocc} outside valid range"
        
        # Check that result has a sharpened image with correct shape
        assert result.sharpened_image.shape == blurred.shape, \
            f"Sharpened image shape {result.sharpened_image.shape} != input shape {blurred.shape}"


@given(
    blurred=image_array(min_size=20, max_size=40),
    params=parameter_lists()
)
@settings(max_examples=30, deadline=None)
def test_find_optimal_returns_best_result(blurred, params):
    """
    Test that find_optimal returns the result with the highest metric value.
    
    This is an additional property test to ensure the find_optimal method
    correctly identifies the best result.
    """
    t_values, lambda_values, sigma_values = params
    
    # Create reference image
    reference = blurred.copy()
    
    # Test parameters
    tester = ParameterTester()
    results = tester.test_parameters(
        blurred=blurred,
        reference=reference,
        t_values=t_values,
        lambda_values=lambda_values,
        sigma_values=sigma_values
    )
    
    # Skip if no results
    assume(len(results) > 0)
    
    # Find optimal by PLCC
    optimal_plcc = tester.find_optimal(results, metric='plcc')
    
    # Verify it has the highest PLCC
    max_plcc = max(r.plcc for r in results)
    assert optimal_plcc.plcc == max_plcc, \
        f"Optimal PLCC {optimal_plcc.plcc} != max PLCC {max_plcc}"
    
    # Find optimal by SROCC
    optimal_srocc = tester.find_optimal(results, metric='srocc')
    
    # Verify it has the highest SROCC
    max_srocc = max(r.srocc for r in results)
    assert optimal_srocc.srocc == max_srocc, \
        f"Optimal SROCC {optimal_srocc.srocc} != max SROCC {max_srocc}"
