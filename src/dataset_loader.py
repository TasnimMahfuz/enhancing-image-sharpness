"""Dataset loader for image quality assessment datasets."""

import os
from pathlib import Path
from typing import List
import numpy as np
from PIL import Image

from src.data_models import ImagePair


class DatasetLoader:
    """Loader for standard image quality assessment datasets."""
    
    SUPPORTED_DATASETS = ['CSIQ', 'LIVE', 'TID2013', 'KADID10k']
    
    def load_image_pair(self, reference_path: str, distorted_path: str) -> ImagePair:
        """
        Load a single image pair.
        
        Args:
            reference_path: Path to reference image
            distorted_path: Path to distorted (blurred) image
            
        Returns:
            ImagePair object
            
        Raises:
            FileNotFoundError: If either image file doesn't exist
            IOError: If image cannot be decoded
        """
        # Check if files exist
        if not os.path.exists(reference_path):
            raise FileNotFoundError(
                f"Reference image file not found: {reference_path}"
            )
        
        if not os.path.exists(distorted_path):
            raise FileNotFoundError(
                f"Distorted image file not found: {distorted_path}"
            )
        
        try:
            # Load reference image
            ref_img = Image.open(reference_path)
            reference = np.array(ref_img)
            
            if reference.size == 0:
                raise IOError(
                    f"Loaded reference image is empty: {reference_path}"
                )
            
        except Exception as e:
            if isinstance(e, (FileNotFoundError, IOError)):
                raise
            raise IOError(
                f"Failed to load reference image {reference_path}: {str(e)}"
            )
        
        try:
            # Load distorted image
            dist_img = Image.open(distorted_path)
            distorted = np.array(dist_img)
            
            if distorted.size == 0:
                raise IOError(
                    f"Loaded distorted image is empty: {distorted_path}"
                )
            
        except Exception as e:
            if isinstance(e, (FileNotFoundError, IOError)):
                raise
            raise IOError(
                f"Failed to load distorted image {distorted_path}: {str(e)}"
            )
        
        return ImagePair(
            reference=reference,
            distorted=distorted,
            reference_path=reference_path,
            distorted_path=distorted_path
        )
    
    def load_dataset(self, dataset_name: str, dataset_path: str) -> List[ImagePair]:
        """
        Load image pairs from specified dataset.
        
        Args:
            dataset_name: One of ['CSIQ', 'LIVE', 'TID2013', 'KADID10k']
            dataset_path: Path to dataset root directory
            
        Returns:
            List of ImagePair objects containing reference and distorted images
            
        Raises:
            FileNotFoundError: If dataset path doesn't exist
            ValueError: If dataset_name is not supported
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset_name}. "
                f"Supported datasets: {self.SUPPORTED_DATASETS}"
            )
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset path not found: {dataset_path}"
            )
        
        # Call dataset-specific loader
        try:
            if dataset_name == 'CSIQ':
                return self.load_csiq_dataset(dataset_path)
            elif dataset_name == 'LIVE':
                return self.load_live_dataset(dataset_path)
            elif dataset_name == 'TID2013':
                return self.load_tid2013_dataset(dataset_path)
            elif dataset_name == 'KADID10k':
                return self.load_kadid10k_dataset(dataset_path)
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            raise IOError(
                f"Failed to load {dataset_name} dataset from {dataset_path}: {str(e)}"
            )
    
    def load_csiq_dataset(self, dataset_path: str) -> List[ImagePair]:
        """
        Load CSIQ dataset.
        
        CSIQ dataset structure:
        - src_imgs/: Reference images
        - dst_imgs/: Distorted images (organized by distortion type)
        
        Args:
            dataset_path: Path to CSIQ dataset root
            
        Returns:
            List of ImagePair objects
        """
        pairs = []
        dataset_path = Path(dataset_path)
        
        # CSIQ typical structure
        src_dir = dataset_path / 'src_imgs'
        dst_dir = dataset_path / 'dst_imgs'
        
        if not src_dir.exists():
            raise FileNotFoundError(
                f"CSIQ source directory not found: {src_dir}"
            )
        
        if not dst_dir.exists():
            raise FileNotFoundError(
                f"CSIQ distorted directory not found: {dst_dir}"
            )
        
        # Find all reference images
        ref_images = sorted(src_dir.glob('*.png')) + sorted(src_dir.glob('*.jpg'))
        
        for ref_path in ref_images:
            # Find corresponding distorted images
            # CSIQ naming: reference.png -> reference.distortion_type.level.png
            ref_name = ref_path.stem
            
            # Look for distorted versions in subdirectories
            for dist_path in dst_dir.rglob(f'{ref_name}.*'):
                if dist_path.is_file() and dist_path.suffix in ['.png', '.jpg']:
                    try:
                        pair = self.load_image_pair(str(ref_path), str(dist_path))
                        pairs.append(pair)
                    except (FileNotFoundError, IOError) as e:
                        # Log warning but continue
                        print(f"Warning: Skipping pair due to error: {e}")
                        continue
        
        return pairs
    
    def load_live_dataset(self, dataset_path: str) -> List[ImagePair]:
        """
        Load LIVE dataset.
        
        LIVE dataset structure:
        - refimgs/: Reference images
        - Various distortion folders (jp2k, jpeg, wn, gblur, fastfading)
        
        Args:
            dataset_path: Path to LIVE dataset root
            
        Returns:
            List of ImagePair objects
        """
        pairs = []
        dataset_path = Path(dataset_path)
        
        ref_dir = dataset_path / 'refimgs'
        
        if not ref_dir.exists():
            raise FileNotFoundError(
                f"LIVE reference directory not found: {ref_dir}"
            )
        
        # LIVE distortion types
        distortion_dirs = ['jp2k', 'jpeg', 'wn', 'gblur', 'fastfading']
        
        for dist_type in distortion_dirs:
            dist_dir = dataset_path / dist_type
            if not dist_dir.exists():
                continue
            
            # Find all distorted images in this directory
            dist_images = sorted(dist_dir.glob('*.bmp')) + sorted(dist_dir.glob('*.png'))
            
            for dist_path in dist_images:
                # Try to find corresponding reference image
                # LIVE naming varies, so we'll try common patterns
                dist_name = dist_path.stem
                
                # Try to extract reference name (often first part before underscore or number)
                ref_candidates = list(ref_dir.glob('*.bmp')) + list(ref_dir.glob('*.png'))
                
                for ref_path in ref_candidates:
                    # Simple heuristic: if distorted name starts with reference name
                    if dist_name.startswith(ref_path.stem):
                        try:
                            pair = self.load_image_pair(str(ref_path), str(dist_path))
                            pairs.append(pair)
                            break
                        except (FileNotFoundError, IOError) as e:
                            print(f"Warning: Skipping pair due to error: {e}")
                            continue
        
        return pairs
    
    def load_tid2013_dataset(self, dataset_path: str) -> List[ImagePair]:
        """
        Load TID2013 dataset.
        
        TID2013 dataset structure:
        - reference_images/: Reference images
        - distorted_images/: Distorted images
        
        Args:
            dataset_path: Path to TID2013 dataset root
            
        Returns:
            List of ImagePair objects
        """
        pairs = []
        dataset_path = Path(dataset_path)
        
        ref_dir = dataset_path / 'reference_images'
        dist_dir = dataset_path / 'distorted_images'
        
        if not ref_dir.exists():
            raise FileNotFoundError(
                f"TID2013 reference directory not found: {ref_dir}"
            )
        
        if not dist_dir.exists():
            raise FileNotFoundError(
                f"TID2013 distorted directory not found: {dist_dir}"
            )
        
        # Find all reference images (case-insensitive)
        ref_images = (sorted(ref_dir.glob('*.bmp')) + sorted(ref_dir.glob('*.BMP')) + 
                     sorted(ref_dir.glob('*.png')) + sorted(ref_dir.glob('*.PNG')))
        
        for ref_path in ref_images:
            ref_name = ref_path.stem
            
            # TID2013 naming: I01.BMP (reference) -> I01_01_1.bmp (distorted)
            # Pattern: {ref_name}_{distortion_type}_{level}.bmp
            # Use set to avoid duplicates from case variations
            dist_paths = set()
            for pattern in [f'{ref_name}_*', f'{ref_name.lower()}_*', f'{ref_name.upper()}_*']:
                dist_paths.update(dist_dir.glob(pattern))
            
            for dist_path in dist_paths:
                if dist_path.is_file() and dist_path.suffix.lower() in ['.bmp', '.png']:
                    try:
                        pair = self.load_image_pair(str(ref_path), str(dist_path))
                        pairs.append(pair)
                    except (FileNotFoundError, IOError) as e:
                        print(f"Warning: Skipping pair due to error: {e}")
                        continue
        
        return pairs
    
    def load_kadid10k_dataset(self, dataset_path: str) -> List[ImagePair]:
        """
        Load KADID-10k dataset.
        
        KADID-10k dataset structure:
        - images/: All images (reference and distorted)
        - Reference images: I01.png, I02.png, etc.
        - Distorted images: I01_01_01.png, I01_01_02.png, etc.
        
        Args:
            dataset_path: Path to KADID-10k dataset root
            
        Returns:
            List of ImagePair objects
        """
        pairs = []
        dataset_path = Path(dataset_path)
        
        images_dir = dataset_path / 'images'
        
        if not images_dir.exists():
            raise FileNotFoundError(
                f"KADID-10k images directory not found: {images_dir}"
            )
        
        # Find all reference images (pattern: I##.png without underscores)
        all_images = sorted(images_dir.glob('*.png'))
        
        ref_images = [img for img in all_images if '_' not in img.stem]
        
        for ref_path in ref_images:
            ref_name = ref_path.stem
            
            # KADID-10k naming: I01.png (reference) -> I01_01_01.png (distorted)
            # Pattern: {ref_name}_{distortion_type}_{level}.png
            for dist_path in images_dir.glob(f'{ref_name}_*'):
                if dist_path.is_file() and dist_path.suffix == '.png':
                    try:
                        pair = self.load_image_pair(str(ref_path), str(dist_path))
                        pairs.append(pair)
                    except (FileNotFoundError, IOError) as e:
                        print(f"Warning: Skipping pair due to error: {e}")
                        continue
        
        return pairs
