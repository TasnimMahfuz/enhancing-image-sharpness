"""
Image Combiner Module

Combines enhanced edge images with blurred input images to produce
sharpened output. This is the final step in the sharpening pipeline.
"""

import numpy as np
from src.exceptions import ValidationError, NumericalError


class ImageCombiner:
    """
    Combine enhanced edges with blurred images to produce sharpened output.
    
    This class implements the final step of the sharpening pipeline:
    sharpened = blurred + enhanced_edge
    
    The combination is performed in float32 to handle potential overflow,
    then clipped to [0, 255] range and converted back to uint8.
    """
    
    def combine(self, blurred: np.ndarray, enhanced_edge: np.ndarray) -> np.ndarray:
        """
        Combine blurred image with enhanced edges.
        
        Args:
            blurred: Original blurred input image
            enhanced_edge: Enhanced edge information from CoefficientEnhancer
            
        Returns:
            Sharpened image with pixel values clipped to [0, 255] as uint8
            
        Raises:
            ValidationError: If image dimensions don't match or inputs are invalid
            NumericalError: If combination produces non-finite values
        """
        # Validate inputs
        if not isinstance(blurred, np.ndarray):
            raise ValidationError(
                f"blurred must be a numpy array, got {type(blurred)}"
            )
        
        if not isinstance(enhanced_edge, np.ndarray):
            raise ValidationError(
                f"enhanced_edge must be a numpy array, got {type(enhanced_edge)}"
            )
        
        if blurred.size == 0:
            raise ValidationError(
                "blurred image is empty (size 0)"
            )
        
        if enhanced_edge.size == 0:
            raise ValidationError(
                "enhanced_edge image is empty (size 0)"
            )
        
        if blurred.shape != enhanced_edge.shape:
            raise ValidationError(
                f"Image dimensions must match. "
                f"Blurred: {blurred.shape}, Enhanced edge: {enhanced_edge.shape}"
            )
        
        # Check for non-finite values in inputs
        if not np.all(np.isfinite(blurred)):
            nan_count = np.sum(np.isnan(blurred))
            inf_count = np.sum(np.isinf(blurred))
            raise NumericalError(
                f"blurred image contains non-finite values. "
                f"NaN count: {nan_count}, Inf count: {inf_count}"
            )
        
        if not np.all(np.isfinite(enhanced_edge)):
            nan_count = np.sum(np.isnan(enhanced_edge))
            inf_count = np.sum(np.isinf(enhanced_edge))
            raise NumericalError(
                f"enhanced_edge image contains non-finite values. "
                f"NaN count: {nan_count}, Inf count: {inf_count}"
            )
        
        # Convert to float32 for safe addition (handles overflow)
        blurred_float = blurred.astype(np.float32)
        enhanced_edge_float = enhanced_edge.astype(np.float32)
        
        # Add enhanced edges to blurred image
        sharpened = blurred_float + enhanced_edge_float
        
        # Check for non-finite values before clipping
        if not np.all(np.isfinite(sharpened)):
            nan_count = np.sum(np.isnan(sharpened))
            inf_count = np.sum(np.isinf(sharpened))
            raise NumericalError(
                f"Image combination produced non-finite values. "
                f"NaN count: {nan_count}, Inf count: {inf_count}. "
                f"Blurred range: [{np.min(blurred)}, {np.max(blurred)}], "
                f"Enhanced edge range: [{np.min(enhanced_edge)}, {np.max(enhanced_edge)}]"
            )
        
        # Clip to valid range [0, 255]
        sharpened = np.clip(sharpened, 0, 255)
        
        # Convert back to uint8 for output
        return sharpened.astype(np.uint8)
