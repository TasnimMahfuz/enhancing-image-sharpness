"""Shared test fixtures and configuration."""

import pytest
import numpy as np
from hypothesis import strategies as st


@pytest.fixture
def sample_grayscale_image():
    """Generate a small grayscale test image."""
    return np.random.randint(0, 256, (50, 50), dtype=np.uint8)


@pytest.fixture
def sample_color_image():
    """Generate a small color test image."""
    return np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_pair(sample_color_image):
    """Generate a pair of test images."""
    reference = sample_color_image
    # Create distorted version by adding noise
    noise = np.random.randint(-10, 10, reference.shape, dtype=np.int16)
    distorted = np.clip(reference.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return reference, distorted


# Hypothesis strategies for property-based testing
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


@st.composite
def processing_config_strategy(draw):
    """Generate random valid processing configurations."""
    sigma = draw(st.floats(min_value=0.1, max_value=5.0))
    t = draw(st.floats(min_value=0.0, max_value=1.0))
    lambda_param = draw(st.floats(min_value=-10.0, max_value=10.0))
    omega = draw(st.floats(min_value=0.1, max_value=2.0))
    
    from src.data_models import ProcessingConfig
    return ProcessingConfig(
        sigma=sigma,
        t=t,
        lambda_param=lambda_param,
        omega=omega
    )
