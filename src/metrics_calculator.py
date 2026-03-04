"""Quality metrics calculator for image sharpening evaluation.

This module implements PLCC (Pearson Linear Correlation Coefficient) and
SROCC (Spearman's Rank Ordered Correlation Coefficient) metrics to evaluate
the quality of sharpened images against reference images.
"""

import numpy as np
from src.exceptions import ValidationError, NumericalError


class MetricsCalculator:
    """Calculate quality metrics for image comparison.
    
    This class provides methods to compute PLCC and SROCC metrics between
    sharpened and reference images. Both metrics operate on grayscale images,
    so color images are automatically converted.
    """
    
    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Convert color image to grayscale if needed.
        
        Uses the standard RGB to grayscale conversion formula:
        gray = 0.299*R + 0.587*G + 0.114*B
        
        Args:
            image: Input image, shape (H, W) for grayscale or (H, W, 3) for color
            
        Returns:
            Grayscale image with shape (H, W)
            
        Raises:
            ValidationError: If image format is invalid
            NumericalError: If conversion produces non-finite values
            
        Note:
            If the input is already grayscale (2D), it is returned unchanged.
        """
        if not isinstance(image, np.ndarray):
            raise ValidationError(
                f"image must be a numpy array, got {type(image)}"
            )
        
        if image.size == 0:
            raise ValidationError(
                "image is empty (size 0)"
            )
        
        if image.ndim == 2:
            # Already grayscale
            return image
        elif image.ndim == 3 and image.shape[2] == 3:
            # Color image - convert to grayscale
            R = image[:, :, 0]
            G = image[:, :, 1]
            B = image[:, :, 2]
            gray = 0.299 * R + 0.587 * G + 0.114 * B
            
            # Check for non-finite values
            if not np.all(np.isfinite(gray)):
                nan_count = np.sum(np.isnan(gray))
                inf_count = np.sum(np.isinf(gray))
                raise NumericalError(
                    f"Grayscale conversion produced non-finite values. "
                    f"NaN count: {nan_count}, Inf count: {inf_count}"
                )
            
            return gray
        else:
            raise ValidationError(
                f"Image must be 2D (grayscale) or 3D with 3 channels (color), "
                f"got shape {image.shape}"
            )
    
    def calculate_plcc(self, sharpened: np.ndarray, reference: np.ndarray) -> float:
        """Calculate Pearson Linear Correlation Coefficient.
        
        PLCC measures the linear correlation between two images:
        PLCC = cov(X, Y) / (σ_X * σ_Y)
        
        Args:
            sharpened: Sharpened image
            reference: Reference image (same shape as sharpened)
            
        Returns:
            PLCC value in [-1, 1]
            
        Raises:
            ValidationError: If inputs are invalid or shapes don't match
            NumericalError: If calculation produces non-finite values
            
        Note:
            - Returns 0.0 for constant images (σ = 0)
            - Automatically converts color images to grayscale
        """
        # Validate inputs
        if not isinstance(sharpened, np.ndarray):
            raise ValidationError(
                f"sharpened must be a numpy array, got {type(sharpened)}"
            )
        
        if not isinstance(reference, np.ndarray):
            raise ValidationError(
                f"reference must be a numpy array, got {type(reference)}"
            )
        
        if sharpened.shape != reference.shape:
            raise ValidationError(
                f"Image shapes must match. "
                f"Sharpened: {sharpened.shape}, Reference: {reference.shape}"
            )
        
        # Check for non-finite values in inputs
        if not np.all(np.isfinite(sharpened)):
            nan_count = np.sum(np.isnan(sharpened))
            inf_count = np.sum(np.isinf(sharpened))
            raise NumericalError(
                f"sharpened image contains non-finite values. "
                f"NaN count: {nan_count}, Inf count: {inf_count}"
            )
        
        if not np.all(np.isfinite(reference)):
            nan_count = np.sum(np.isnan(reference))
            inf_count = np.sum(np.isinf(reference))
            raise NumericalError(
                f"reference image contains non-finite values. "
                f"NaN count: {nan_count}, Inf count: {inf_count}"
            )
        
        # Convert to grayscale if needed
        gray_sharpened = self._to_grayscale(sharpened)
        gray_reference = self._to_grayscale(reference)
        
        # Flatten to 1D arrays
        x = gray_sharpened.flatten().astype(np.float64)
        y = gray_reference.flatten().astype(np.float64)
        
        # Calculate means
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        # Calculate covariance and standard deviations
        numerator = np.sum((x - mean_x) * (y - mean_y))
        denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
        
        # Handle constant images (denominator = 0)
        if denominator < 1e-10:
            return 0.0
        
        plcc = numerator / denominator
        
        # Check for non-finite result
        if not np.isfinite(plcc):
            raise NumericalError(
                f"PLCC calculation produced non-finite value: {plcc}. "
                f"Numerator: {numerator}, Denominator: {denominator}"
            )
        
        # Clamp to valid range [-1, 1] (handle numerical precision issues)
        plcc = np.clip(plcc, -1.0, 1.0)
        
        return float(plcc)
    
    def calculate_srocc(self, sharpened: np.ndarray, reference: np.ndarray) -> float:
        """Calculate Spearman's Rank Ordered Correlation Coefficient.
        
        SROCC measures the rank correlation between two images by:
        1. Computing ranks for each image
        2. Calculating Pearson correlation on the ranks
        
        Ranks are computed using: rank(x) = argsort(argsort(x))
        
        Args:
            sharpened: Sharpened image
            reference: Reference image (same shape as sharpened)
            
        Returns:
            SROCC value in [-1, 1]
            
        Raises:
            ValidationError: If inputs are invalid or shapes don't match
            NumericalError: If calculation produces non-finite values
            
        Note:
            - More robust to non-linear relationships than PLCC
            - Automatically converts color images to grayscale
        """
        # Validate inputs
        if not isinstance(sharpened, np.ndarray):
            raise ValidationError(
                f"sharpened must be a numpy array, got {type(sharpened)}"
            )
        
        if not isinstance(reference, np.ndarray):
            raise ValidationError(
                f"reference must be a numpy array, got {type(reference)}"
            )
        
        if sharpened.shape != reference.shape:
            raise ValidationError(
                f"Image shapes must match. "
                f"Sharpened: {sharpened.shape}, Reference: {reference.shape}"
            )
        
        # Check for non-finite values in inputs
        if not np.all(np.isfinite(sharpened)):
            nan_count = np.sum(np.isnan(sharpened))
            inf_count = np.sum(np.isinf(sharpened))
            raise NumericalError(
                f"sharpened image contains non-finite values. "
                f"NaN count: {nan_count}, Inf count: {inf_count}"
            )
        
        if not np.all(np.isfinite(reference)):
            nan_count = np.sum(np.isnan(reference))
            inf_count = np.sum(np.isinf(reference))
            raise NumericalError(
                f"reference image contains non-finite values. "
                f"NaN count: {nan_count}, Inf count: {inf_count}"
            )
        
        # Convert to grayscale if needed
        gray_sharpened = self._to_grayscale(sharpened)
        gray_reference = self._to_grayscale(reference)
        
        # Flatten to 1D arrays
        x = gray_sharpened.flatten().astype(np.float64)
        y = gray_reference.flatten().astype(np.float64)
        
        # Compute ranks: rank(x) = argsort(argsort(x))
        rank_x = np.argsort(np.argsort(x)).astype(np.float64)
        rank_y = np.argsort(np.argsort(y)).astype(np.float64)
        
        # Calculate Pearson correlation on ranks
        mean_rank_x = np.mean(rank_x)
        mean_rank_y = np.mean(rank_y)
        
        numerator = np.sum((rank_x - mean_rank_x) * (rank_y - mean_rank_y))
        denominator = np.sqrt(np.sum((rank_x - mean_rank_x)**2) * np.sum((rank_y - mean_rank_y)**2))
        
        # Handle constant images (denominator = 0)
        if denominator < 1e-10:
            return 0.0
        
        srocc = numerator / denominator
        
        # Check for non-finite result
        if not np.isfinite(srocc):
            raise NumericalError(
                f"SROCC calculation produced non-finite value: {srocc}. "
                f"Numerator: {numerator}, Denominator: {denominator}"
            )
        
        # Clamp to valid range [-1, 1] (handle numerical precision issues)
        srocc = np.clip(srocc, -1.0, 1.0)
        
        return float(srocc)
