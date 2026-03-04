"""
Property-based tests for configuration parsing and formatting.

Feature: image-sharpness-enhancement
Tests Properties 27-30 from the design document.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings

from src.image_processor import ImageProcessor
from src.data_models import ProcessingConfig, ProcessingResult


# Custom strategies for generating test data
@st.composite
def valid_config_dict(draw):
    """Generate valid configuration dictionaries."""
    sigma = draw(st.floats(min_value=0.1, max_value=5.0))
    t = draw(st.floats(min_value=0.0, max_value=0.99))  # Avoid t=1
    lambda_param = draw(st.floats(min_value=-10.0, max_value=10.0))
    omega = draw(st.floats(min_value=0.1, max_value=2.0))
    
    # Randomly use 'lambda' or 'lambda_param' key
    lambda_key = draw(st.sampled_from(['lambda', 'lambda_param']))
    
    config_dict = {
        'sigma': sigma,
        't': t,
        lambda_key: lambda_param,
        'omega': omega
    }
    
    return config_dict


@st.composite
def valid_processing_config(draw):
    """Generate valid ProcessingConfig objects."""
    sigma = draw(st.floats(min_value=0.1, max_value=5.0))
    t = draw(st.floats(min_value=0.0, max_value=0.99))
    lambda_param = draw(st.floats(min_value=-10.0, max_value=10.0))
    omega = draw(st.floats(min_value=0.1, max_value=2.0))
    
    return ProcessingConfig(
        sigma=sigma,
        t=t,
        lambda_param=lambda_param,
        omega=omega
    )


@given(config_dict=valid_config_dict())
@settings(max_examples=100)
def test_property_27_configuration_parsing_extracts_parameters(config_dict):
    """
    Feature: image-sharpness-enhancement
    Property 27: Configuration Parsing Extracts Parameters
    
    **Validates: Requirements 12.1**
    
    For any valid configuration dictionary containing sigma, t, and lambda parameters,
    parsing should correctly extract all three parameter values.
    """
    processor = ImageProcessor()
    
    # Parse the configuration
    config = processor.parse_config(config_dict)
    
    # Verify all parameters are correctly extracted
    assert abs(config.sigma - config_dict['sigma']) < 1e-9
    assert abs(config.t - config_dict['t']) < 1e-9
    assert abs(config.omega - config_dict['omega']) < 1e-9
    
    # Lambda can be under 'lambda' or 'lambda_param' key
    if 'lambda' in config_dict:
        assert abs(config.lambda_param - config_dict['lambda']) < 1e-9
    else:
        assert abs(config.lambda_param - config_dict['lambda_param']) < 1e-9


@given(
    sigma=st.floats(min_value=-10.0, max_value=0.0)
)
@settings(max_examples=50)
def test_property_28_invalid_sigma_produces_descriptive_error(sigma):
    """
    Feature: image-sharpness-enhancement
    Property 28: Invalid Parameters Produce Descriptive Errors (Sigma)
    
    **Validates: Requirements 12.2**
    
    For any invalid parameter value (sigma ≤ 0), the error message should
    indicate which parameter is invalid and what the valid range is.
    """
    processor = ImageProcessor()
    
    config_dict = {
        'sigma': sigma,
        't': 0.5,
        'lambda': 0.0,
        'omega': 1.0
    }
    
    with pytest.raises(ValueError) as exc_info:
        processor.parse_config(config_dict)
    
    error_message = str(exc_info.value)
    
    # Error should mention sigma and valid range
    assert 'sigma' in error_message.lower() or 'σ' in error_message
    assert 'positive' in error_message.lower() or '> 0' in error_message


@given(
    t=st.one_of(
        st.floats(min_value=-10.0, max_value=-0.01),
        st.floats(min_value=1.01, max_value=10.0)
    )
)
@settings(max_examples=50)
def test_property_28_invalid_t_produces_descriptive_error(t):
    """
    Feature: image-sharpness-enhancement
    Property 28: Invalid Parameters Produce Descriptive Errors (t)
    
    **Validates: Requirements 12.2**
    
    For any invalid t value (t ∉ [0,1]), the error message should
    indicate which parameter is invalid and what the valid range is.
    """
    processor = ImageProcessor()
    
    config_dict = {
        'sigma': 1.0,
        't': t,
        'lambda': 0.0,
        'omega': 1.0
    }
    
    with pytest.raises(ValueError) as exc_info:
        processor.parse_config(config_dict)
    
    error_message = str(exc_info.value)
    
    # Error should mention t and valid range [0, 1]
    assert 't' in error_message.lower()
    assert '[0, 1]' in error_message or ('0' in error_message and '1' in error_message)


@given(
    omega=st.floats(min_value=-10.0, max_value=0.0)
)
@settings(max_examples=50)
def test_property_28_invalid_omega_produces_descriptive_error(omega):
    """
    Feature: image-sharpness-enhancement
    Property 28: Invalid Parameters Produce Descriptive Errors (Omega)
    
    **Validates: Requirements 12.2**
    
    For any invalid omega value (omega ≤ 0), the error message should
    indicate which parameter is invalid and what the valid range is.
    """
    processor = ImageProcessor()
    
    config_dict = {
        'sigma': 1.0,
        't': 0.5,
        'lambda': 0.0,
        'omega': omega
    }
    
    with pytest.raises(ValueError) as exc_info:
        processor.parse_config(config_dict)
    
    error_message = str(exc_info.value)
    
    # Error should mention omega and valid range
    assert 'omega' in error_message.lower() or 'ω' in error_message
    assert 'positive' in error_message.lower() or '> 0' in error_message


def test_property_28_missing_sigma_produces_descriptive_error():
    """
    Feature: image-sharpness-enhancement
    Property 28: Invalid Parameters Produce Descriptive Errors (Missing Sigma)
    
    **Validates: Requirements 12.2**
    
    When required parameter sigma is missing, error should indicate this clearly.
    """
    processor = ImageProcessor()
    
    config_dict = {
        't': 0.5,
        'lambda': 0.0,
        'omega': 1.0
    }
    
    with pytest.raises(ValueError) as exc_info:
        processor.parse_config(config_dict)
    
    error_message = str(exc_info.value)
    
    # Error should mention missing sigma
    assert 'sigma' in error_message.lower() or 'σ' in error_message
    assert 'missing' in error_message.lower() or 'required' in error_message.lower()


def test_property_28_missing_t_produces_descriptive_error():
    """
    Feature: image-sharpness-enhancement
    Property 28: Invalid Parameters Produce Descriptive Errors (Missing t)
    
    **Validates: Requirements 12.2**
    
    When required parameter t is missing, error should indicate this clearly.
    """
    processor = ImageProcessor()
    
    config_dict = {
        'sigma': 1.0,
        'lambda': 0.0,
        'omega': 1.0
    }
    
    with pytest.raises(ValueError) as exc_info:
        processor.parse_config(config_dict)
    
    error_message = str(exc_info.value)
    
    # Error should mention missing t
    assert 't' in error_message.lower()
    assert 'missing' in error_message.lower() or 'required' in error_message.lower()


def test_property_28_missing_lambda_produces_descriptive_error():
    """
    Feature: image-sharpness-enhancement
    Property 28: Invalid Parameters Produce Descriptive Errors (Missing Lambda)
    
    **Validates: Requirements 12.2**
    
    When required parameter lambda is missing, error should indicate this clearly.
    """
    processor = ImageProcessor()
    
    config_dict = {
        'sigma': 1.0,
        't': 0.5,
        'omega': 1.0
    }
    
    with pytest.raises(ValueError) as exc_info:
        processor.parse_config(config_dict)
    
    error_message = str(exc_info.value)
    
    # Error should mention missing lambda
    assert 'lambda' in error_message.lower() or 'λ' in error_message


@given(config=valid_processing_config())
@settings(max_examples=100)
def test_property_29_output_format_contains_required_fields(config):
    """
    Feature: image-sharpness-enhancement
    Property 29: Output Format Contains Required Fields
    
    **Validates: Requirements 12.3**
    
    For any processing result, the output dictionary should contain fields for
    sharpened_image, plcc, srocc, and parameters.
    """
    processor = ImageProcessor()
    
    # Create a mock ProcessingResult
    sharpened_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    result = ProcessingResult(
        sharpened_image=sharpened_image,
        plcc=0.95,
        srocc=0.93,
        parameters=config,
        processing_time=1.5
    )
    
    # Format the result
    output = processor.format_result(result)
    
    # Verify all required fields are present
    assert 'sharpened_image' in output
    assert 'plcc' in output
    assert 'srocc' in output
    assert 'parameters' in output
    assert 'processing_time' in output
    
    # Verify parameters dict has all required fields
    assert 'sigma' in output['parameters']
    assert 't' in output['parameters']
    assert 'lambda_param' in output['parameters']
    assert 'omega' in output['parameters']
    
    # Verify values match
    assert np.array_equal(output['sharpened_image'], sharpened_image)
    assert output['plcc'] == 0.95
    assert output['srocc'] == 0.93
    assert output['processing_time'] == 1.5
    assert abs(output['parameters']['sigma'] - config.sigma) < 1e-9
    assert abs(output['parameters']['t'] - config.t) < 1e-9
    assert abs(output['parameters']['lambda_param'] - config.lambda_param) < 1e-9
    assert abs(output['parameters']['omega'] - config.omega) < 1e-9


@given(config_dict=valid_config_dict())
@settings(max_examples=100)
def test_property_30_configuration_round_trip(config_dict):
    """
    Feature: image-sharpness-enhancement
    Property 30: Configuration Round-Trip Property
    
    **Validates: Requirements 12.4**
    
    For any valid parameter configuration, serializing to dictionary format,
    then parsing, then serializing again should produce an equivalent dictionary.
    
    Round-trip: dict → parse → ProcessingConfig → format → dict
    """
    processor = ImageProcessor()
    
    # Parse the configuration
    config = processor.parse_config(config_dict)
    
    # Create a mock result to format
    sharpened_image = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    result = ProcessingResult(
        sharpened_image=sharpened_image,
        plcc=0.95,
        srocc=0.93,
        parameters=config,
        processing_time=1.5
    )
    
    # Format the result
    formatted = processor.format_result(result)
    
    # Extract parameters from formatted output
    params_dict = formatted['parameters']
    
    # Parse again (need to convert lambda_param back to lambda for consistency)
    params_dict_for_parse = {
        'sigma': params_dict['sigma'],
        't': params_dict['t'],
        'lambda': params_dict['lambda_param'],
        'omega': params_dict['omega']
    }
    
    config2 = processor.parse_config(params_dict_for_parse)
    
    # Verify round-trip produces equivalent values
    assert abs(config.sigma - config2.sigma) < 1e-9
    assert abs(config.t - config2.t) < 1e-9
    assert abs(config.lambda_param - config2.lambda_param) < 1e-9
    assert abs(config.omega - config2.omega) < 1e-9
