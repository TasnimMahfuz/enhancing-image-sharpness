"""Property-based tests for dataset loading."""

import os
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
from hypothesis import given, strategies as st, settings
import pytest

from src.dataset_loader import DatasetLoader


# Custom strategies for generating test data
@st.composite
def image_array(draw, min_size=10, max_size=100):
    """Generate random image arrays."""
    height = draw(st.integers(min_value=min_size, max_value=max_size))
    width = draw(st.integers(min_value=min_size, max_value=max_size))
    channels = draw(st.sampled_from([1, 3]))
    
    if channels == 1:
        shape = (height, width)
    else:
        shape = (height, width, channels)
    
    # Generate random pixel values
    img = np.random.randint(0, 256, shape, dtype=np.uint8)
    
    return img


@given(img_array=image_array())
@settings(max_examples=50, deadline=None)
def test_property_1_image_loading_preserves_dimensions(img_array):
    """
    Feature: image-sharpness-enhancement
    Property 1: Image Loading Preserves Dimensions
    
    **Validates: Requirements 1.5, 1.6, 1.7**
    
    For any valid image file, loading the image and checking its dimensions 
    should return the same dimensions as the original file metadata.
    """
    loader = DatasetLoader()
    
    # Create temporary image files
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = os.path.join(tmpdir, 'reference.png')
        dist_path = os.path.join(tmpdir, 'distorted.png')
        
        # Save images
        if img_array.ndim == 2:
            ref_img = Image.fromarray(img_array, mode='L')
            dist_img = Image.fromarray(img_array, mode='L')
        else:
            ref_img = Image.fromarray(img_array, mode='RGB')
            dist_img = Image.fromarray(img_array, mode='RGB')
        
        ref_img.save(ref_path)
        dist_img.save(dist_path)
        
        # Load the image pair
        pair = loader.load_image_pair(ref_path, dist_path)
        
        # Verify dimensions are preserved
        assert pair.reference.shape == img_array.shape, \
            f"Reference dimensions {pair.reference.shape} don't match original {img_array.shape}"
        assert pair.distorted.shape == img_array.shape, \
            f"Distorted dimensions {pair.distorted.shape} don't match original {img_array.shape}"


@given(
    height=st.integers(min_value=10, max_value=100),
    width=st.integers(min_value=10, max_value=100),
    channels=st.sampled_from([1, 3])
)
@settings(max_examples=50, deadline=None)
def test_property_2_dataset_loader_returns_paired_images(height, width, channels):
    """
    Feature: image-sharpness-enhancement
    Property 2: Dataset Loader Returns Paired Images
    
    **Validates: Requirements 1.5**
    
    For any valid dataset path with reference and distorted images, 
    the loader should return image pairs where both images have the same dimensions.
    """
    loader = DatasetLoader()
    
    # Create temporary dataset structure
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate random images
        if channels == 1:
            shape = (height, width)
            mode = 'L'
        else:
            shape = (height, width, channels)
            mode = 'RGB'
        
        ref_array = np.random.randint(0, 256, shape, dtype=np.uint8)
        dist_array = np.random.randint(0, 256, shape, dtype=np.uint8)
        
        ref_path = os.path.join(tmpdir, 'reference.png')
        dist_path = os.path.join(tmpdir, 'distorted.png')
        
        # Save images
        ref_img = Image.fromarray(ref_array, mode=mode)
        dist_img = Image.fromarray(dist_array, mode=mode)
        
        ref_img.save(ref_path)
        dist_img.save(dist_path)
        
        # Load the image pair
        pair = loader.load_image_pair(ref_path, dist_path)
        
        # Verify both images have the same dimensions
        assert pair.reference.shape == pair.distorted.shape, \
            f"Reference shape {pair.reference.shape} doesn't match distorted shape {pair.distorted.shape}"


@given(
    invalid_path=st.text(min_size=1, max_size=50).filter(lambda x: not os.path.exists(x))
)
@settings(max_examples=20, deadline=None)
def test_property_3_error_messages_include_context(invalid_path):
    """
    Feature: image-sharpness-enhancement
    Property 3: Error Messages Include Context
    
    **Validates: Requirements 1.6**
    
    For any invalid file path or unreadable image, the error message 
    should contain both the file path and a description of the failure reason.
    """
    loader = DatasetLoader()
    
    # Create a valid temporary file for one path
    with tempfile.TemporaryDirectory() as tmpdir:
        valid_path = os.path.join(tmpdir, 'valid.png')
        img = Image.fromarray(np.random.randint(0, 256, (10, 10), dtype=np.uint8), mode='L')
        img.save(valid_path)
        
        # Test with invalid reference path
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_image_pair(invalid_path, valid_path)
        
        error_message = str(exc_info.value)
        # Verify error message contains the file path
        assert invalid_path in error_message or 'not found' in error_message.lower(), \
            f"Error message '{error_message}' doesn't contain file path or 'not found'"
        
        # Test with invalid distorted path
        with pytest.raises(FileNotFoundError) as exc_info:
            loader.load_image_pair(valid_path, invalid_path)
        
        error_message = str(exc_info.value)
        # Verify error message contains the file path
        assert invalid_path in error_message or 'not found' in error_message.lower(), \
            f"Error message '{error_message}' doesn't contain file path or 'not found'"


def test_property_3_corrupted_file_error_context():
    """
    Feature: image-sharpness-enhancement
    Property 3: Error Messages Include Context (corrupted files)
    
    **Validates: Requirements 1.6**
    
    For corrupted image files, the error message should contain 
    the file path and indicate the file is corrupted/unreadable.
    """
    loader = DatasetLoader()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a corrupted image file (just write random bytes)
        corrupted_path = os.path.join(tmpdir, 'corrupted.png')
        with open(corrupted_path, 'wb') as f:
            f.write(b'not a valid image file')
        
        # Create a valid image
        valid_path = os.path.join(tmpdir, 'valid.png')
        img = Image.fromarray(np.random.randint(0, 256, (10, 10), dtype=np.uint8), mode='L')
        img.save(valid_path)
        
        # Test with corrupted reference
        with pytest.raises(IOError) as exc_info:
            loader.load_image_pair(corrupted_path, valid_path)
        
        error_message = str(exc_info.value)
        # Verify error message contains the file path
        assert corrupted_path in error_message, \
            f"Error message '{error_message}' doesn't contain file path '{corrupted_path}'"
        
        # Test with corrupted distorted
        with pytest.raises(IOError) as exc_info:
            loader.load_image_pair(valid_path, corrupted_path)
        
        error_message = str(exc_info.value)
        # Verify error message contains the file path
        assert corrupted_path in error_message, \
            f"Error message '{error_message}' doesn't contain file path '{corrupted_path}'"
