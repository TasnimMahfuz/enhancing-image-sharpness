"""Unit tests for dataset loader."""

import os
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.dataset_loader import DatasetLoader
from src.exceptions import FileLoadError, ValidationError


class TestDatasetLoader:
    """Unit tests for DatasetLoader class."""
    
    def test_load_image_pair_basic(self):
        """Test basic image pair loading."""
        loader = DatasetLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            ref_array = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
            dist_array = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
            
            ref_path = os.path.join(tmpdir, 'reference.png')
            dist_path = os.path.join(tmpdir, 'distorted.png')
            
            Image.fromarray(ref_array).save(ref_path)
            Image.fromarray(dist_array).save(dist_path)
            
            # Load pair
            pair = loader.load_image_pair(ref_path, dist_path)
            
            assert pair.reference.shape == (50, 50, 3)
            assert pair.distorted.shape == (50, 50, 3)
            assert pair.reference_path == ref_path
            assert pair.distorted_path == dist_path
    
    def test_load_image_pair_grayscale(self):
        """Test loading grayscale images."""
        loader = DatasetLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create grayscale test images
            ref_array = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
            dist_array = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
            
            ref_path = os.path.join(tmpdir, 'reference.png')
            dist_path = os.path.join(tmpdir, 'distorted.png')
            
            Image.fromarray(ref_array, mode='L').save(ref_path)
            Image.fromarray(dist_array, mode='L').save(dist_path)
            
            # Load pair
            pair = loader.load_image_pair(ref_path, dist_path)
            
            assert pair.reference.shape == (50, 50)
            assert pair.distorted.shape == (50, 50)
    
    def test_load_image_pair_missing_reference(self):
        """Test error handling for missing reference image."""
        loader = DatasetLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            dist_path = os.path.join(tmpdir, 'distorted.png')
            Image.fromarray(np.zeros((10, 10), dtype=np.uint8), mode='L').save(dist_path)
            
            with pytest.raises(FileLoadError) as exc_info:
                loader.load_image_pair('/nonexistent/reference.png', dist_path)
            
            assert 'reference' in str(exc_info.value).lower()
            assert '/nonexistent/reference.png' in str(exc_info.value)
    
    def test_load_image_pair_missing_distorted(self):
        """Test error handling for missing distorted image."""
        loader = DatasetLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_path = os.path.join(tmpdir, 'reference.png')
            Image.fromarray(np.zeros((10, 10), dtype=np.uint8), mode='L').save(ref_path)
            
            with pytest.raises(FileLoadError) as exc_info:
                loader.load_image_pair(ref_path, '/nonexistent/distorted.png')
            
            assert 'distorted' in str(exc_info.value).lower()
            assert '/nonexistent/distorted.png' in str(exc_info.value)
    
    def test_load_image_pair_corrupted_file(self):
        """Test error handling for corrupted image files."""
        loader = DatasetLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create corrupted file
            corrupted_path = os.path.join(tmpdir, 'corrupted.png')
            with open(corrupted_path, 'wb') as f:
                f.write(b'not an image')
            
            valid_path = os.path.join(tmpdir, 'valid.png')
            Image.fromarray(np.zeros((10, 10), dtype=np.uint8), mode='L').save(valid_path)
            
            with pytest.raises(IOError):
                loader.load_image_pair(corrupted_path, valid_path)
    
    def test_load_dataset_unsupported_format(self):
        """Test error handling for unsupported dataset format."""
        loader = DatasetLoader()
        
        with pytest.raises(ValueError) as exc_info:
            loader.load_dataset('UNKNOWN_DATASET', '/some/path')
        
        assert 'Unsupported dataset' in str(exc_info.value)
        assert 'UNKNOWN_DATASET' in str(exc_info.value)
    
    def test_load_dataset_missing_path(self):
        """Test error handling for missing dataset path."""
        loader = DatasetLoader()
        
        with pytest.raises(FileLoadError) as exc_info:
            loader.load_dataset('CSIQ', '/nonexistent/path')
        
        assert 'not found' in str(exc_info.value).lower()
        assert '/nonexistent/path' in str(exc_info.value)


class TestCSIQDataset:
    """
    Feature: image-sharpness-enhancement
    Example 5: CSIQ Dataset Support
    
    **Validates: Requirements 1.1**
    """
    
    def test_load_csiq_dataset_structure(self):
        """Test loading CSIQ dataset with proper structure."""
        loader = DatasetLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create CSIQ-like structure
            src_dir = Path(tmpdir) / 'src_imgs'
            dst_dir = Path(tmpdir) / 'dst_imgs'
            src_dir.mkdir()
            dst_dir.mkdir()
            
            # Create reference image
            ref_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            ref_path = src_dir / 'image01.png'
            Image.fromarray(ref_array).save(ref_path)
            
            # Create distorted images
            dist_array1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            dist_path1 = dst_dir / 'image01.blur.1.png'
            Image.fromarray(dist_array1).save(dist_path1)
            
            dist_array2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            dist_path2 = dst_dir / 'image01.noise.2.png'
            Image.fromarray(dist_array2).save(dist_path2)
            
            # Load dataset
            pairs = loader.load_csiq_dataset(tmpdir)
            
            assert len(pairs) == 2
            assert all(pair.reference.shape == (100, 100, 3) for pair in pairs)
            assert all(pair.distorted.shape == (100, 100, 3) for pair in pairs)
    
    def test_load_csiq_dataset_missing_src_dir(self):
        """Test error handling when CSIQ src_imgs directory is missing."""
        loader = DatasetLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only dst_dir
            dst_dir = Path(tmpdir) / 'dst_imgs'
            dst_dir.mkdir()
            
            with pytest.raises(FileLoadError) as exc_info:
                loader.load_csiq_dataset(tmpdir)
            
            assert 'src_imgs' in str(exc_info.value) or 'source' in str(exc_info.value).lower()
    
    def test_load_csiq_dataset_missing_dst_dir(self):
        """Test error handling when CSIQ dst_imgs directory is missing."""
        loader = DatasetLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only src_dir
            src_dir = Path(tmpdir) / 'src_imgs'
            src_dir.mkdir()
            
            with pytest.raises(FileLoadError) as exc_info:
                loader.load_csiq_dataset(tmpdir)
            
            assert 'dst_imgs' in str(exc_info.value) or 'distorted' in str(exc_info.value).lower()


class TestLIVEDataset:
    """
    Feature: image-sharpness-enhancement
    Example 6: LIVE Dataset Support
    
    **Validates: Requirements 1.2**
    """
    
    def test_load_live_dataset_structure(self):
        """Test loading LIVE dataset with proper structure."""
        loader = DatasetLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create LIVE-like structure
            ref_dir = Path(tmpdir) / 'refimgs'
            ref_dir.mkdir()
            
            # Create reference image
            ref_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            ref_path = ref_dir / 'building.bmp'
            Image.fromarray(ref_array).save(ref_path)
            
            # Create distortion directory
            gblur_dir = Path(tmpdir) / 'gblur'
            gblur_dir.mkdir()
            
            # Create distorted images
            dist_array1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            dist_path1 = gblur_dir / 'building_1.bmp'
            Image.fromarray(dist_array1).save(dist_path1)
            
            dist_array2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            dist_path2 = gblur_dir / 'building_2.bmp'
            Image.fromarray(dist_array2).save(dist_path2)
            
            # Load dataset
            pairs = loader.load_live_dataset(tmpdir)
            
            assert len(pairs) == 2
            assert all(pair.reference.shape == (100, 100, 3) for pair in pairs)
            assert all(pair.distorted.shape == (100, 100, 3) for pair in pairs)
    
    def test_load_live_dataset_missing_refimgs(self):
        """Test error handling when LIVE refimgs directory is missing."""
        loader = DatasetLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileLoadError) as exc_info:
                loader.load_live_dataset(tmpdir)
            
            assert 'refimgs' in str(exc_info.value) or 'reference' in str(exc_info.value).lower()
    
    def test_load_live_dataset_empty(self):
        """Test loading LIVE dataset with no distortion directories."""
        loader = DatasetLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only refimgs directory
            ref_dir = Path(tmpdir) / 'refimgs'
            ref_dir.mkdir()
            
            # Create reference image
            ref_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            ref_path = ref_dir / 'building.bmp'
            Image.fromarray(ref_array).save(ref_path)
            
            # Load dataset (should return empty list)
            pairs = loader.load_live_dataset(tmpdir)
            
            assert len(pairs) == 0


class TestTID2013Dataset:
    """
    Feature: image-sharpness-enhancement
    Example 7: TID2013 Dataset Support
    
    **Validates: Requirements 1.3**
    """
    
    def test_load_tid2013_dataset_structure(self):
        """Test loading TID2013 dataset with proper structure."""
        loader = DatasetLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create TID2013-like structure
            ref_dir = Path(tmpdir) / 'reference_images'
            dist_dir = Path(tmpdir) / 'distorted_images'
            ref_dir.mkdir()
            dist_dir.mkdir()
            
            # Create reference image
            ref_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            ref_path = ref_dir / 'I01.BMP'
            Image.fromarray(ref_array).save(ref_path)
            
            # Create distorted images
            dist_array1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            dist_path1 = dist_dir / 'I01_01_1.bmp'
            Image.fromarray(dist_array1).save(dist_path1)
            
            dist_array2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            dist_path2 = dist_dir / 'I01_02_3.bmp'
            Image.fromarray(dist_array2).save(dist_path2)
            
            # Load dataset
            pairs = loader.load_tid2013_dataset(tmpdir)
            
            assert len(pairs) == 2
            assert all(pair.reference.shape == (100, 100, 3) for pair in pairs)
            assert all(pair.distorted.shape == (100, 100, 3) for pair in pairs)
    
    def test_load_tid2013_dataset_missing_reference_dir(self):
        """Test error handling when TID2013 reference_images directory is missing."""
        loader = DatasetLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only distorted_images directory
            dist_dir = Path(tmpdir) / 'distorted_images'
            dist_dir.mkdir()
            
            with pytest.raises(FileLoadError) as exc_info:
                loader.load_tid2013_dataset(tmpdir)
            
            assert 'reference_images' in str(exc_info.value) or 'reference' in str(exc_info.value).lower()
    
    def test_load_tid2013_dataset_missing_distorted_dir(self):
        """Test error handling when TID2013 distorted_images directory is missing."""
        loader = DatasetLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only reference_images directory
            ref_dir = Path(tmpdir) / 'reference_images'
            ref_dir.mkdir()
            
            with pytest.raises(FileLoadError) as exc_info:
                loader.load_tid2013_dataset(tmpdir)
            
            assert 'distorted_images' in str(exc_info.value) or 'distorted' in str(exc_info.value).lower()


class TestKADID10kDataset:
    """
    Feature: image-sharpness-enhancement
    Example 8: KADID10k Dataset Support
    
    **Validates: Requirements 1.4**
    """
    
    def test_load_kadid10k_dataset_structure(self):
        """Test loading KADID-10k dataset with proper structure."""
        loader = DatasetLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create KADID-10k-like structure
            images_dir = Path(tmpdir) / 'images'
            images_dir.mkdir()
            
            # Create reference image
            ref_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            ref_path = images_dir / 'I01.png'
            Image.fromarray(ref_array).save(ref_path)
            
            # Create distorted images
            dist_array1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            dist_path1 = images_dir / 'I01_01_01.png'
            Image.fromarray(dist_array1).save(dist_path1)
            
            dist_array2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            dist_path2 = images_dir / 'I01_02_03.png'
            Image.fromarray(dist_array2).save(dist_path2)
            
            # Load dataset
            pairs = loader.load_kadid10k_dataset(tmpdir)
            
            assert len(pairs) == 2
            assert all(pair.reference.shape == (100, 100, 3) for pair in pairs)
            assert all(pair.distorted.shape == (100, 100, 3) for pair in pairs)
    
    def test_load_kadid10k_dataset_missing_images_dir(self):
        """Test error handling when KADID-10k images directory is missing."""
        loader = DatasetLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileLoadError) as exc_info:
                loader.load_kadid10k_dataset(tmpdir)
            
            assert 'images' in str(exc_info.value).lower()
    
    def test_load_kadid10k_dataset_multiple_references(self):
        """Test loading KADID-10k dataset with multiple reference images."""
        loader = DatasetLoader()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create KADID-10k-like structure
            images_dir = Path(tmpdir) / 'images'
            images_dir.mkdir()
            
            # Create two reference images
            for ref_num in ['01', '02']:
                ref_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
                ref_path = images_dir / f'I{ref_num}.png'
                Image.fromarray(ref_array).save(ref_path)
                
                # Create distorted images for each reference
                for dist_num in ['01', '02']:
                    dist_array = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
                    dist_path = images_dir / f'I{ref_num}_{dist_num}_01.png'
                    Image.fromarray(dist_array).save(dist_path)
            
            # Load dataset
            pairs = loader.load_kadid10k_dataset(tmpdir)
            
            # Should have 4 pairs total (2 references × 2 distortions each)
            assert len(pairs) == 4
            assert all(pair.reference.shape == (100, 100, 3) for pair in pairs)
            assert all(pair.distorted.shape == (100, 100, 3) for pair in pairs)
