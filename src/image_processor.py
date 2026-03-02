"""
Image Processor Module

Main orchestrator for the image sharpening workflow. Coordinates all components
to execute the complete pipeline from input images to sharpened output with metrics.
"""

import time
import numpy as np
from typing import Optional

from src.data_models import ProcessingConfig, ProcessingResult
from src.gaussian_filter import GaussianFilter
from src.edge_extractor import EdgeExtractor
from src.coefficient_enhancer import CoefficientEnhancer
from src.image_combiner import ImageCombiner
from src.metrics_calculator import MetricsCalculator


class ImageProcessor:
    """
    Main orchestrator for image sharpening workflow.
    
    This class coordinates all components to execute the complete pipeline:
    1. Validate configuration and images
    2. Apply Gaussian filter to reference image
    3. Extract edges (reference - smoothed)
    4. Calculate coefficient bounds and enhance edges
    5. Combine enhanced edges with blurred image
    6. Calculate quality metrics (PLCC, SROCC)
    7. Return results with processing time
    """
    
    def validate_config(self, config: ProcessingConfig) -> None:
        """
        Validate configuration parameters.
        
        Checks:
        - sigma > 0
        - t ∈ [0, 1]
        - omega > 0
        
        Args:
            config: Processing configuration to validate
            
        Raises:
            ValueError: If any parameter is invalid with descriptive message
        """
        if config.sigma <= 0:
            raise ValueError(
                f"Gaussian sigma must be positive, got {config.sigma}. "
                f"Valid range: σ > 0"
            )
        
        if not 0 <= config.t <= 1:
            raise ValueError(
                f"Parameter t must be in [0, 1], got {config.t}. "
                f"Valid range: 0 ≤ t ≤ 1"
            )
        
        if config.omega <= 0:
            raise ValueError(
                f"Parameter omega must be positive, got {config.omega}. "
                f"Valid range: Ω > 0"
            )
    
    def _validate_images(self, blurred: np.ndarray, reference: np.ndarray) -> None:
        """
        Validate image inputs.
        
        Checks:
        - Images have matching dimensions
        - Images are 2D (grayscale) or 3D (color)
        - Color images have 3 channels
        
        Args:
            blurred: Blurred input image
            reference: Reference image
            
        Raises:
            ValueError: If images are incompatible
        """
        if blurred.shape != reference.shape:
            raise ValueError(
                f"Image dimensions must match. "
                f"Blurred: {blurred.shape}, Reference: {reference.shape}"
            )
        
        if blurred.ndim not in [2, 3]:
            raise ValueError(
                f"Images must be 2D (grayscale) or 3D (color), got {blurred.ndim}D"
            )
        
        if blurred.ndim == 3 and blurred.shape[2] != 3:
            raise ValueError(
                f"Color images must have 3 channels (RGB), got {blurred.shape[2]}"
            )
    
    def process(
        self,
        blurred: np.ndarray,
        reference: np.ndarray,
        config: ProcessingConfig,
        store_intermediates: bool = False
    ) -> ProcessingResult:
        """
        Execute complete sharpening workflow.
        
        Pipeline:
        1. Validate configuration and images
        2. Convert images to float32
        3. Apply Gaussian filter to reference
        4. Extract edges
        5. Calculate coefficients and enhance edges
        6. Combine with blurred image
        7. Calculate metrics
        8. Return results
        
        Args:
            blurred: Blurred input image
            reference: Reference image for edge extraction
            config: Processing parameters
            store_intermediates: If True, store intermediate images in result
            
        Returns:
            ProcessingResult with sharpened image, metrics, and timing
            
        Raises:
            ValueError: If parameters or images are invalid
            RuntimeError: If processing fails at any step
        """
        start_time = time.time()
        intermediate_images = {} if store_intermediates else None
        
        try:
            # Step 1: Validate inputs
            self.validate_config(config)
            self._validate_images(blurred, reference)
            
            # Step 2: Convert to float32 for processing
            blurred_float = blurred.astype(np.float32)
            reference_float = reference.astype(np.float32)
            
            # Step 3: Apply Gaussian filter to reference
            try:
                gaussian_filter = GaussianFilter(config.sigma)
                smoothed_image = gaussian_filter.apply(reference_float)
                if store_intermediates:
                    intermediate_images['smoothed'] = smoothed_image
            except Exception as e:
                raise RuntimeError(f"Gaussian filtering failed: {str(e)}")
            
            # Step 4: Extract edges
            try:
                edge_extractor = EdgeExtractor()
                edge_image = edge_extractor.extract(reference_float, smoothed_image)
                if store_intermediates:
                    intermediate_images['edges'] = edge_image
            except Exception as e:
                raise RuntimeError(f"Edge extraction failed: {str(e)}")
            
            # Step 5: Calculate coefficient bounds and enhance edges
            try:
                enhancer = CoefficientEnhancer(
                    config.t,
                    config.lambda_param,
                    config.omega
                )
                enhanced_edge = enhancer.enhance(edge_image)
                if store_intermediates:
                    intermediate_images['enhanced_edges'] = enhanced_edge
                    intermediate_images['a2_bound'] = enhancer.a2_bound
                    intermediate_images['a3_bound'] = enhancer.a3_bound
            except Exception as e:
                raise RuntimeError(f"Edge enhancement failed: {str(e)}")
            
            # Step 6: Combine with blurred image
            try:
                combiner = ImageCombiner()
                sharpened_image = combiner.combine(blurred_float, enhanced_edge)
                if store_intermediates:
                    intermediate_images['sharpened'] = sharpened_image
            except Exception as e:
                raise RuntimeError(f"Image combination failed: {str(e)}")
            
            # Step 7: Calculate quality metrics
            try:
                metrics_calc = MetricsCalculator()
                plcc = metrics_calc.calculate_plcc(sharpened_image, reference_float)
                srocc = metrics_calc.calculate_srocc(sharpened_image, reference_float)
            except Exception as e:
                raise RuntimeError(f"Metrics calculation failed: {str(e)}")
            
            # Step 8: Package results
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                sharpened_image=sharpened_image,
                plcc=plcc,
                srocc=srocc,
                parameters=config,
                processing_time=processing_time,
                intermediate_images=intermediate_images
            )
            
        except (ValueError, RuntimeError):
            # Re-raise validation and processing errors as-is
            raise
        except Exception as e:
            # Catch any unexpected errors
            raise RuntimeError(f"Unexpected error during processing: {str(e)}")
