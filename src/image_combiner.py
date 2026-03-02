"""
Image Combiner Module

Combines enhanced edge images with blurred input images to produce
sharpened output. This is the final step in the sharpening pipeline.
"""

import numpy as np


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
            ValueError: If image dimensions don't match
        """
        if blurred.shape != enhanced_edge.shape:
            raise ValueError(
                f"Image dimensions must match. "
                f"Blurred: {blurred.shape}, Enhanced edge: {enhanced_edge.shape}"
            )
        
        # Convert to float32 for safe addition (handles overflow)
        blurred_float = blurred.astype(np.float32)
        enhanced_edge_float = enhanced_edge.astype(np.float32)
        
        # Add enhanced edges to blurred image
        sharpened = blurred_float + enhanced_edge_float
        
        # Clip to valid range [0, 255]
        sharpened = np.clip(sharpened, 0, 255)
        
        # Convert back to uint8 for output
        return sharpened.astype(np.uint8)
