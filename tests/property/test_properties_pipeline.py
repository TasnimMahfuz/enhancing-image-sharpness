"""
Property-based tests for complete image processing pipeline.

Feature: image-sharpness-enhancement
Tests Properties 25-26 from the design document.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings

from src.image_processor import ImageProcessor
from src.data_models import ProcessingConfig


# Custom strategies for generating test data
@st.composite
def valid_image_array(draw, min_size=10, max_size=100):
    """Generate valid image arrays (grayscale or color)."""
    height = draw(st.integers(min_value=min_size, max_value=max_size))
    width = draw(st.integers(min_value=min_size, max_value=max_size))
    
    # Randomly choose grayscale or color
    is_color = draw(st.booleans())
    
    if is_color:
        shape = (height, width, 3)
    else:
        shape = (height, width)
    
    # Generate random pixel values using numpy
    img = draw(st.integers(min_value=0, max_value=255).map(
        lambda _: np.random.randint(0, 256, size=shape, dtype=np.uint8)
    ))
    
    return img


@st.composite
def valid_processing_config(draw):
    """Generate valid processing configurations."""
    sigma = draw(st.floats(min_value=0.1, max_value=5.0))
    t = draw(st.floats(min_value=0.0, max_value=0.99))  # Avoid t=1
    lambda_param = draw(st.floats(min_value=-10.0, max_value=10.0))
    omega = draw(st.floats(min_value=0.1, max_value=2.0))
    
    return ProcessingConfig(
        sigma=sigma,
        t=t,
        lambda_param=lambda_param,
        omega=omega
    )


@given(
    config=valid_processing_config(),
    image=valid_image_array(min_size=20, max_size=50)
)
@settings(max_examples=50, deadline=5000)
def test_property_25_complete_pipeline_produces_output(config, image):
    """
    Feature: image-sharpness-enhancement
    Property 25: Complete Pipeline Produces Output
    
    **Validates: Requirements 10.1, 10.3**
    
    For any valid blurred image, reference image, and configuration parameters,
    the image processor should successfully produce a sharpened image without errors.
    """
    processor = ImageProcessor()
    
    # Use the same image as both blurred and reference for simplicity
    blurred = image
    reference = image
    
    # Process should complete without raising exceptions
    result = processor.process(blurred, reference, config)
    
    # Verify result has expected structure
    assert result.sharpened_image is not None
    assert result.sharpened_image.shape == image.shape
    assert result.sharpened_image.dtype == np.uint8
    
    # Verify metrics are in valid range
    assert -1 <= result.plcc <= 1
    assert -1 <= result.srocc <= 1
    
    # Verify processing time is recorded
    assert result.processing_time >= 0
    
    # Verify parameters are stored
    assert result.parameters == config


@given(
    sigma=st.floats(min_value=-10.0, max_value=0.0),
    image=valid_image_array(min_size=20, max_size=50)
)
@settings(max_examples=20)
def test_property_26_pipeline_error_messages_include_step_invalid_sigma(sigma, image):
    """
    Feature: image-sharpness-enhancement
    Property 26: Pipeline Error Messages Include Step (Invalid Sigma)
    
    **Validates: Requirements 10.5**
    
    For any processing failure (invalid input, numerical error, etc.),
    the error message should indicate which processing step failed and why.
    
    This test checks that invalid sigma values produce descriptive errors.
    """
    processor = ImageProcessor()
    
    config = ProcessingConfig(sigma=sigma, t=0.5, lambda_param=0.0, omega=1.0)
    
    with pytest.raises(ValueError) as exc_info:
        processor.process(image, image, config)
    
    error_message = str(exc_info.value)
    
    # Error message should mention sigma and valid range
    assert 'sigma' in error_message.lower() or 'σ' in error_message
    assert 'positive' in error_message.lower() or '> 0' in error_message


@given(
    t=st.floats(min_value=1.01, max_value=2.0),
    image=valid_image_array(min_size=20, max_size=50)
)
@settings(max_examples=20)
def test_property_26_pipeline_error_messages_include_step_invalid_t(t, image):
    """
    Feature: image-sharpness-enhancement
    Property 26: Pipeline Error Messages Include Step (Invalid t)
    
    **Validates: Requirements 10.5**
    
    For any processing failure with invalid t parameter,
    the error message should indicate the parameter and valid range.
    """
    processor = ImageProcessor()
    
    config = ProcessingConfig(sigma=1.0, t=t, lambda_param=0.0, omega=1.0)
    
    with pytest.raises(ValueError) as exc_info:
        processor.process(image, image, config)
    
    error_message = str(exc_info.value)
    
    # Error message should mention t and valid range [0, 1]
    assert 't' in error_message.lower()
    assert '[0, 1]' in error_message or '0' in error_message and '1' in error_message


@given(
    omega=st.floats(min_value=-10.0, max_value=0.0),
    image=valid_image_array(min_size=20, max_size=50)
)
@settings(max_examples=20)
def test_property_26_pipeline_error_messages_include_step_invalid_omega(omega, image):
    """
    Feature: image-sharpness-enhancement
    Property 26: Pipeline Error Messages Include Step (Invalid Omega)
    
    **Validates: Requirements 10.5**
    
    For any processing failure with invalid omega parameter,
    the error message should indicate the parameter and valid range.
    """
    processor = ImageProcessor()
    
    config = ProcessingConfig(sigma=1.0, t=0.5, lambda_param=0.0, omega=omega)
    
    with pytest.raises(ValueError) as exc_info:
        processor.process(image, image, config)
    
    error_message = str(exc_info.value)
    
    # Error message should mention omega and valid range
    assert 'omega' in error_message.lower() or 'ω' in error_message
    assert 'positive' in error_message.lower() or '> 0' in error_message


def test_property_26_pipeline_error_messages_include_step_mismatched_dimensions():
    """
    Feature: image-sharpness-enhancement
    Property 26: Pipeline Error Messages Include Step (Mismatched Dimensions)
    
    **Validates: Requirements 10.5**
    
    When images have mismatched dimensions, the error message should
    indicate the dimension mismatch clearly.
    """
    processor = ImageProcessor()
    
    blurred = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    reference = np.random.randint(0, 256, (60, 60, 3), dtype=np.uint8)
    config = ProcessingConfig(sigma=1.0, t=0.5, lambda_param=0.0, omega=1.0)
    
    with pytest.raises(ValueError) as exc_info:
        processor.process(blurred, reference, config)
    
    error_message = str(exc_info.value)
    
    # Error message should mention dimension mismatch
    assert 'dimension' in error_message.lower() or 'shape' in error_message.lower()
    assert 'match' in error_message.lower()
