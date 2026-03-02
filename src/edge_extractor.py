"""Edge extraction by subtracting smoothed from original image."""

import numpy as np
from src.exceptions import ValidationError, NumericalError


class EdgeExtractor:
    """Extract edge information from images."""
    
    def extract(self, reference: np.ndarray, smoothed: np.ndarray) -> np.ndarray:
        """
        Extract edge information by computing difference.
        
        Args:
            reference: Original reference image
            smoothed: Gaussian-smoothed version of reference
            
        Returns:
            Edge image (reference - smoothed)
            
        Raises:
            ValidationError: If image shapes don't match or inputs are invalid
            NumericalError: If edge extraction produces non-finite values
            
        Note:
            Preserves negative values without clipping.
            Output may have values outside [0, 255] range.
        """
        # Validate inputs
        if not isinstance(reference, np.ndarray):
            raise ValidationError(
                f"reference must be a numpy array, got {type(reference)}"
            )
        
        if not isinstance(smoothed, np.ndarray):
            raise ValidationError(
                f"smoothed must be a numpy array, got {type(smoothed)}"
            )
        
        if reference.size == 0:
            raise ValidationError(
                "reference image is empty (size 0)"
            )
        
        if smoothed.size == 0:
            raise ValidationError(
                "smoothed image is empty (size 0)"
            )
        
        if reference.shape != smoothed.shape:
            raise ValidationError(
                f"Image shapes must match. "
                f"Reference: {reference.shape}, Smoothed: {smoothed.shape}"
            )
        
        # Check for non-finite values in inputs
        if not np.all(np.isfinite(reference)):
            nan_count = np.sum(np.isnan(reference))
            inf_count = np.sum(np.isinf(reference))
            raise NumericalError(
                f"reference image contains non-finite values. "
                f"NaN count: {nan_count}, Inf count: {inf_count}"
            )
        
        if not np.all(np.isfinite(smoothed)):
            nan_count = np.sum(np.isnan(smoothed))
            inf_count = np.sum(np.isinf(smoothed))
            raise NumericalError(
                f"smoothed image contains non-finite values. "
                f"NaN count: {nan_count}, Inf count: {inf_count}"
            )
        
        # Convert to float32 to preserve negative values and avoid overflow
        reference_float = reference.astype(np.float32)
        smoothed_float = smoothed.astype(np.float32)
        
        # Simple subtraction - preserves negative values
        edge = reference_float - smoothed_float
        
        # Check for non-finite values in output
        if not np.all(np.isfinite(edge)):
            nan_count = np.sum(np.isnan(edge))
            inf_count = np.sum(np.isinf(edge))
            raise NumericalError(
                f"Edge extraction produced non-finite values. "
                f"NaN count: {nan_count}, Inf count: {inf_count}. "
                f"Reference range: [{np.min(reference)}, {np.max(reference)}], "
                f"Smoothed range: [{np.min(smoothed)}, {np.max(smoothed)}]"
            )
        
        return edge
