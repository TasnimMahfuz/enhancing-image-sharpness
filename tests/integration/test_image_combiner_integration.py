"""
Integration test demonstrating the complete workflow with ImageCombiner.

This test shows how all components work together:
GaussianFilter -> EdgeExtractor -> CoefficientEnhancer -> ImageCombiner
"""

import numpy as np
from src.gaussian_filter import GaussianFilter
from src.edge_extractor import EdgeExtractor
from src.coefficient_enhancer import CoefficientEnhancer
from src.image_combiner import ImageCombiner


def test_complete_sharpening_pipeline():
    """
    Test the complete image sharpening pipeline.
    
    This integration test verifies that all components work together
    to produce a valid sharpened image.
    """
    # Create test images (simulating blurred and reference images)
    height, width = 50, 50
    blurred = np.random.randint(50, 200, (height, width), dtype=np.uint8)
    reference = np.random.randint(50, 200, (height, width), dtype=np.uint8)
    
    # Step 1: Apply Gaussian filter to reference
    gaussian_filter = GaussianFilter(sigma=1.0)
    smoothed = gaussian_filter.apply(reference.astype(np.float32))
    
    # Step 2: Extract edges
    edge_extractor = EdgeExtractor()
    edge = edge_extractor.extract(reference.astype(np.float32), smoothed)
    
    # Step 3: Enhance edges using coefficient bounds
    enhancer = CoefficientEnhancer(t=0.6, lambda_param=1.0, omega=1.0)
    enhanced_edge = enhancer.enhance(edge)
    
    # Step 4: Combine enhanced edges with blurred image
    combiner = ImageCombiner()
    sharpened = combiner.combine(blurred, enhanced_edge)
    
    # Verify output properties
    assert sharpened.shape == blurred.shape, "Output shape should match input"
    assert sharpened.dtype == np.uint8, "Output should be uint8"
    assert np.all(sharpened >= 0) and np.all(sharpened <= 255), \
        "All pixels should be in valid range [0, 255]"
    
    print(f"Pipeline completed successfully!")
    print(f"Input shape: {blurred.shape}")
    print(f"Output shape: {sharpened.shape}")
    print(f"Output range: [{sharpened.min()}, {sharpened.max()}]")


def test_complete_pipeline_with_color_image():
    """
    Test the complete pipeline with a color image.
    """
    # Create color test images
    height, width, channels = 50, 50, 3
    blurred = np.random.randint(50, 200, (height, width, channels), dtype=np.uint8)
    reference = np.random.randint(50, 200, (height, width, channels), dtype=np.uint8)
    
    # Step 1: Apply Gaussian filter
    gaussian_filter = GaussianFilter(sigma=1.0)
    smoothed = gaussian_filter.apply(reference.astype(np.float32))
    
    # Step 2: Extract edges
    edge_extractor = EdgeExtractor()
    edge = edge_extractor.extract(reference.astype(np.float32), smoothed)
    
    # Step 3: Enhance edges
    enhancer = CoefficientEnhancer(t=0.6, lambda_param=1.0, omega=1.0)
    enhanced_edge = enhancer.enhance(edge)
    
    # Step 4: Combine
    combiner = ImageCombiner()
    sharpened = combiner.combine(blurred, enhanced_edge)
    
    # Verify output properties
    assert sharpened.shape == blurred.shape, "Output shape should match input"
    assert sharpened.dtype == np.uint8, "Output should be uint8"
    assert np.all(sharpened >= 0) and np.all(sharpened <= 255), \
        "All pixels should be in valid range [0, 255]"
    
    print(f"Color pipeline completed successfully!")
    print(f"Input shape: {blurred.shape}")
    print(f"Output shape: {sharpened.shape}")


def test_pipeline_with_paper_recommended_parameters():
    """
    Test pipeline with the paper's recommended parameters: t=0.6, λ=0.
    """
    # Create test images
    height, width = 50, 50
    blurred = np.random.randint(50, 200, (height, width), dtype=np.uint8)
    reference = np.random.randint(50, 200, (height, width), dtype=np.uint8)
    
    # Use paper's recommended parameters
    gaussian_filter = GaussianFilter(sigma=1.0)
    smoothed = gaussian_filter.apply(reference.astype(np.float32))
    
    edge_extractor = EdgeExtractor()
    edge = edge_extractor.extract(reference.astype(np.float32), smoothed)
    
    # Paper recommends t=0.6, λ=0
    enhancer = CoefficientEnhancer(t=0.6, lambda_param=0.0, omega=1.0)
    enhanced_edge = enhancer.enhance(edge)
    
    combiner = ImageCombiner()
    sharpened = combiner.combine(blurred, enhanced_edge)
    
    # With λ=0, the enhancement factor is 0, so sharpened should equal blurred
    # (since enhanced_edge will be all zeros)
    np.testing.assert_array_equal(sharpened, blurred,
        err_msg="With λ=0, sharpened should equal blurred (no enhancement)")
    
    print(f"Paper parameters test passed!")
