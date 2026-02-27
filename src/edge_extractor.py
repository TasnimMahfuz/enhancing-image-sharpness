"""Edge extraction by subtracting smoothed from original image."""

import numpy as np


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
            
        Note:
            Preserves negative values without clipping.
            Output may have values outside [0, 255] range.
        """
        if reference.shape != smoothed.shape:
            raise ValueError(
                f"Image shapes must match. "
                f"Reference: {reference.shape}, Smoothed: {smoothed.shape}"
            )
        
        # Convert to float32 to preserve negative values and avoid overflow
        reference_float = reference.astype(np.float32)
        smoothed_float = smoothed.astype(np.float32)
        
        # Simple subtraction - preserves negative values
        edge = reference_float - smoothed_float
        
        return edge
