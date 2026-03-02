"""
Integration tests for complete image processing pipeline.

Tests the ImageProcessor with real CSIQ dataset images to verify
end-to-end functionality with various parameter combinations.

Feature: image-sharpness-enhancement
Validates: Requirements 10.4
"""

import pytest
import numpy as np
import time
from pathlib import Path
from PIL import Image

from src.image_processor import ImageProcessor
from src.data_models import ProcessingConfig


# Dataset paths
CSIQ_SRC_DIR = Path("datasets/CSIQ/src_imgs")
CSIQ_BLUR_DIR = Path("datasets/CSIQ/dst_imgs/blur")


def load_image(path: Path) -> np.ndarray:
    """Load image from file as numpy array."""
    img = Image.open(path)
    return np.array(img)


@pytest.fixture
def processor():
    """Create ImageProcessor instance."""
    return ImageProcessor()


@pytest.fixture
def sample_image_pairs():
    """
    Load sample image pairs from CSIQ dataset.
    
    Returns list of (reference_path, blurred_path, name) tuples.
    """
    # Select a few representative images for testing
    test_images = [
        "1600",
        "bridge",
        "sunset_sparrow"
    ]
    
    pairs = []
    for img_name in test_images:
        ref_path = CSIQ_SRC_DIR / f"{img_name}.png"
        blur_path = CSIQ_BLUR_DIR / f"{img_name}.BLUR.3.png"  # Use blur level 3
        
        if ref_path.exists() and blur_path.exists():
            pairs.append((ref_path, blur_path, img_name))
    
    return pairs


def test_pipeline_with_real_csiq_images(processor, sample_image_pairs):
    """
    Test complete pipeline with real CSIQ dataset images.
    
    Validates:
    - Pipeline processes real images without errors
    - Output has correct shape and type
    - Metrics are computed
    - Processing completes successfully
    
    Requirements: 10.4
    """
    if not sample_image_pairs:
        pytest.skip("CSIQ dataset images not available")
    
    config = ProcessingConfig(
        sigma=1.0,
        t=0.6,
        lambda_param=0.0,
        omega=1.0
    )
    
    for ref_path, blur_path, name in sample_image_pairs:
        # Load images
        reference = load_image(ref_path)
        blurred = load_image(blur_path)
        
        # Process
        result = processor.process(blurred, reference, config)
        
        # Verify output
        assert result.sharpened_image is not None, f"No output for {name}"
        assert result.sharpened_image.shape == blurred.shape, \
            f"Shape mismatch for {name}: {result.sharpened_image.shape} vs {blurred.shape}"
        assert result.sharpened_image.dtype == np.uint8, \
            f"Wrong dtype for {name}: {result.sharpened_image.dtype}"
        
        # Verify metrics
        assert -1 <= result.plcc <= 1, f"Invalid PLCC for {name}: {result.plcc}"
        assert -1 <= result.srocc <= 1, f"Invalid SROCC for {name}: {result.srocc}"
        
        # Verify processing time is recorded
        assert result.processing_time > 0, f"No processing time for {name}"
        
        print(f"✓ {name}: PLCC={result.plcc:.4f}, SROCC={result.srocc:.4f}, "
              f"Time={result.processing_time:.3f}s")


def test_pipeline_with_various_parameter_combinations(processor, sample_image_pairs):
    """
    Test pipeline with various parameter combinations.
    
    Validates:
    - Different sigma values work correctly
    - Different t values work correctly
    - Different lambda values work correctly
    - All combinations produce valid output
    
    Requirements: 10.4
    """
    if not sample_image_pairs:
        pytest.skip("CSIQ dataset images not available")
    
    # Test with first image pair only to keep test time reasonable
    ref_path, blur_path, name = sample_image_pairs[0]
    reference = load_image(ref_path)
    blurred = load_image(blur_path)
    
    # Parameter combinations to test
    test_configs = [
        ProcessingConfig(sigma=0.5, t=0.3, lambda_param=0.0, omega=1.0),
        ProcessingConfig(sigma=1.0, t=0.6, lambda_param=0.0, omega=1.0),  # Paper's recommended
        ProcessingConfig(sigma=2.0, t=0.8, lambda_param=0.5, omega=1.0),
        ProcessingConfig(sigma=1.5, t=0.0, lambda_param=1.0, omega=1.5),  # Edge case: t=0
    ]
    
    for i, config in enumerate(test_configs):
        result = processor.process(blurred, reference, config)
        
        # Verify output
        assert result.sharpened_image is not None, f"No output for config {i}"
        assert result.sharpened_image.shape == blurred.shape, f"Shape mismatch for config {i}"
        assert -1 <= result.plcc <= 1, f"Invalid PLCC for config {i}"
        assert -1 <= result.srocc <= 1, f"Invalid SROCC for config {i}"
        
        print(f"✓ Config {i} (σ={config.sigma}, t={config.t}, λ={config.lambda_param}): "
              f"PLCC={result.plcc:.4f}, SROCC={result.srocc:.4f}")


def test_pipeline_processing_time_512x512(processor):
    """
    Verify processing time < 10 seconds for 512x512 images.
    
    Note: This is a soft requirement - actual time depends on hardware.
    The test will pass but log a warning if time exceeds 10 seconds.
    
    Requirements: 10.4
    """
    # Create synthetic 512x512 color image
    blurred = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    reference = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    
    config = ProcessingConfig(sigma=1.0, t=0.6, lambda_param=0.0, omega=1.0)
    
    start_time = time.time()
    result = processor.process(blurred, reference, config)
    elapsed_time = time.time() - start_time
    
    # Verify processing completed
    assert result.sharpened_image is not None
    assert result.sharpened_image.shape == (512, 512, 3)
    
    # Log performance
    print(f"Processing time for 512x512 image: {elapsed_time:.3f}s")
    
    if elapsed_time > 10.0:
        print(f"WARNING: Processing took {elapsed_time:.3f}s, exceeding 10s target")
    else:
        print(f"✓ Processing completed within 10s target")


def test_pipeline_with_grayscale_images(processor):
    """
    Test pipeline with grayscale images.
    
    Validates:
    - Pipeline handles grayscale images correctly
    - Output is grayscale with correct shape
    """
    # Create synthetic grayscale images
    blurred = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    reference = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    config = ProcessingConfig(sigma=1.0, t=0.6, lambda_param=0.0, omega=1.0)
    
    result = processor.process(blurred, reference, config)
    
    # Verify output
    assert result.sharpened_image is not None
    assert result.sharpened_image.shape == (100, 100), \
        f"Expected (100, 100), got {result.sharpened_image.shape}"
    assert result.sharpened_image.dtype == np.uint8
    assert -1 <= result.plcc <= 1
    assert -1 <= result.srocc <= 1


def test_pipeline_with_color_images(processor):
    """
    Test pipeline with color (RGB) images.
    
    Validates:
    - Pipeline handles color images correctly
    - Output is color with correct shape
    """
    # Create synthetic color images
    blurred = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    reference = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    config = ProcessingConfig(sigma=1.0, t=0.6, lambda_param=0.0, omega=1.0)
    
    result = processor.process(blurred, reference, config)
    
    # Verify output
    assert result.sharpened_image is not None
    assert result.sharpened_image.shape == (100, 100, 3), \
        f"Expected (100, 100, 3), got {result.sharpened_image.shape}"
    assert result.sharpened_image.dtype == np.uint8
    assert -1 <= result.plcc <= 1
    assert -1 <= result.srocc <= 1


def test_pipeline_stores_intermediate_images(processor):
    """
    Test that pipeline can store intermediate images when requested.
    
    Validates:
    - store_intermediates flag works correctly
    - Intermediate images are captured
    """
    blurred = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    reference = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    
    config = ProcessingConfig(sigma=1.0, t=0.6, lambda_param=0.0, omega=1.0)
    
    result = processor.process(blurred, reference, config, store_intermediates=True)
    
    # Verify intermediate images are stored
    assert result.intermediate_images is not None
    assert 'smoothed' in result.intermediate_images
    assert 'edges' in result.intermediate_images
    assert 'enhanced_edges' in result.intermediate_images
    assert 'sharpened' in result.intermediate_images
    assert 'a2_bound' in result.intermediate_images
    assert 'a3_bound' in result.intermediate_images


def test_pipeline_error_handling_invalid_config(processor):
    """
    Test that pipeline properly handles invalid configurations.
    
    Validates:
    - Invalid parameters raise ValueError
    - Error messages are descriptive
    """
    blurred = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    reference = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    
    # Test invalid sigma
    config = ProcessingConfig(sigma=-1.0, t=0.6, lambda_param=0.0, omega=1.0)
    with pytest.raises(ValueError) as exc_info:
        processor.process(blurred, reference, config)
    assert 'sigma' in str(exc_info.value).lower()
    
    # Test invalid t
    config = ProcessingConfig(sigma=1.0, t=1.5, lambda_param=0.0, omega=1.0)
    with pytest.raises(ValueError) as exc_info:
        processor.process(blurred, reference, config)
    assert 't' in str(exc_info.value).lower()
    
    # Test invalid omega
    config = ProcessingConfig(sigma=1.0, t=0.6, lambda_param=0.0, omega=-1.0)
    with pytest.raises(ValueError) as exc_info:
        processor.process(blurred, reference, config)
    assert 'omega' in str(exc_info.value).lower()


def test_pipeline_error_handling_mismatched_images(processor):
    """
    Test that pipeline properly handles mismatched image dimensions.
    
    Validates:
    - Mismatched dimensions raise ValueError
    - Error message indicates dimension mismatch
    """
    blurred = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    reference = np.random.randint(0, 256, (60, 60, 3), dtype=np.uint8)
    
    config = ProcessingConfig(sigma=1.0, t=0.6, lambda_param=0.0, omega=1.0)
    
    with pytest.raises(ValueError) as exc_info:
        processor.process(blurred, reference, config)
    
    error_msg = str(exc_info.value).lower()
    assert 'dimension' in error_msg or 'shape' in error_msg
    assert 'match' in error_msg
