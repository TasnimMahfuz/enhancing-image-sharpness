"""Gaussian filter implementation with custom kernel generation."""

import numpy as np
from scipy.ndimage import convolve
from src.exceptions import ValidationError, ProcessingError, NumericalError


class GaussianFilter:
    """Gaussian filter with kernel generated from mathematical formula."""
    
    def __init__(self, sigma: float):
        """
        Initialize Gaussian filter with standard deviation.
        
        Args:
            sigma: Standard deviation of Gaussian distribution (σ > 0)
            
        Raises:
            ValidationError: If sigma is not positive
        """
        if sigma <= 0:
            raise ValidationError(
                f"Gaussian sigma must be positive, got {sigma}. "
                f"Valid range: σ > 0"
            )
        
        self.sigma = sigma
        
        try:
            self.kernel = self._create_kernel()
        except Exception as e:
            raise ProcessingError(
                f"Failed to create Gaussian kernel with sigma={sigma}: {str(e)}"
            )
    
    def _create_kernel(self) -> np.ndarray:
        """
        Generate Gaussian kernel from formula G(x,y) = (1/2πσ²) * e^(-(x²+y²)/2σ²).
        
        Kernel size is determined to capture 99.7% of distribution (±3σ).
        
        Returns:
            2D numpy array with normalized Gaussian kernel
            
        Raises:
            NumericalError: If kernel contains non-finite values
        """
        # Kernel size to capture 99.7% of distribution (±3σ)
        radius = int(np.ceil(3 * self.sigma))
        kernel_size = 2 * radius + 1
        
        # Create coordinate meshgrid centered at 0
        x = np.arange(-radius, radius + 1)
        y = np.arange(-radius, radius + 1)
        X, Y = np.meshgrid(x, y)
        
        # Apply Gaussian formula: G(x,y) = (1/2πσ²) * exp(-(x²+y²)/2σ²)
        coefficient = 1.0 / (2 * np.pi * self.sigma**2)
        exponent = -(X**2 + Y**2) / (2 * self.sigma**2)
        kernel = coefficient * np.exp(exponent)
        
        # Check for numerical issues
        if not np.all(np.isfinite(kernel)):
            raise NumericalError(
                f"Gaussian kernel contains non-finite values. "
                f"sigma={self.sigma}, kernel_size={kernel_size}"
            )
        
        # Normalize so sum equals 1.0
        kernel_sum = np.sum(kernel)
        if kernel_sum < 1e-10:
            raise NumericalError(
                f"Gaussian kernel sum too small: {kernel_sum}. "
                f"sigma={self.sigma}"
            )
        
        kernel = kernel / kernel_sum
        
        return kernel
    
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian filter to image using scipy's convolve.
        
        Args:
            image: Input image, shape (H, W) or (H, W, C)
            
        Returns:
            Smoothed image with same shape as input
        """
        if image.ndim == 2:
            # Grayscale image
            return convolve(image, self.kernel, mode='reflect')
        elif image.ndim == 3:
            # Color image - apply to each channel independently
            smoothed = np.zeros_like(image, dtype=image.dtype)
            for c in range(image.shape[2]):
                smoothed[:, :, c] = convolve(image[:, :, c], self.kernel, mode='reflect')
            return smoothed
        else:
            raise ValueError(f"Image must be 2D or 3D array, got {image.ndim}D")
