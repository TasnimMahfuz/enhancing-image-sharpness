"""
Parameter Tester Module

Provides functionality for testing multiple parameter combinations to find optimal
sharpening settings. Tests all combinations of t, λ, and σ parameters and identifies
the best configuration based on quality metrics.
"""

import numpy as np
from typing import List
import logging

from src.data_models import TestResult, ProcessingConfig
from src.image_processor import ImageProcessor
from src.exceptions import ValidationError, NumericalError


# Set up logging
logger = logging.getLogger(__name__)


class ParameterTester:
    """
    Tests multiple parameter combinations to find optimal sharpening settings.
    
    This class systematically explores the parameter space by testing all
    combinations of t, λ, and σ values. For each combination, it processes
    the image and stores the results with quality metrics.
    """
    
    def __init__(self):
        """Initialize the parameter tester with an image processor."""
        self.processor = ImageProcessor()
    
    def test_parameters(
        self,
        blurred: np.ndarray,
        reference: np.ndarray,
        t_values: List[float],
        lambda_values: List[float],
        sigma_values: List[float]
    ) -> List[TestResult]:
        """
        Test all combinations of parameters.
        
        Generates the cartesian product of all parameter lists and processes
        the image with each combination. Failed combinations are skipped with
        a warning logged.
        
        Args:
            blurred: Blurred input image
            reference: Reference image for edge extraction
            t_values: List of t parameters to test (0 ≤ t ≤ 1)
            lambda_values: List of λ parameters to test (real numbers)
            sigma_values: List of σ parameters to test (σ > 0)
            
        Returns:
            List of TestResult objects with metrics for each successful combination
            
        Raises:
            ValidationError: If inputs are invalid
            
        Example:
            >>> tester = ParameterTester()
            >>> results = tester.test_parameters(
            ...     blurred, reference,
            ...     t_values=[0.5, 0.6, 0.7],
            ...     lambda_values=[0.0, 0.5],
            ...     sigma_values=[1.0, 2.0]
            ... )
            >>> len(results)  # Should be 3 * 2 * 2 = 12 if all succeed
            12
        """
        # Validate inputs
        if not isinstance(blurred, np.ndarray):
            raise ValidationError(
                f"blurred must be a numpy array, got {type(blurred)}"
            )
        
        if not isinstance(reference, np.ndarray):
            raise ValidationError(
                f"reference must be a numpy array, got {type(reference)}"
            )
        
        if blurred.size == 0:
            raise ValidationError("blurred image is empty (size 0)")
        
        if reference.size == 0:
            raise ValidationError("reference image is empty (size 0)")
        
        if blurred.shape != reference.shape:
            raise ValidationError(
                f"Image shapes must match. "
                f"Blurred: {blurred.shape}, Reference: {reference.shape}"
            )
        
        # Check for non-finite values
        if not np.all(np.isfinite(blurred)):
            nan_count = np.sum(np.isnan(blurred))
            inf_count = np.sum(np.isinf(blurred))
            raise NumericalError(
                f"blurred image contains non-finite values. "
                f"NaN count: {nan_count}, Inf count: {inf_count}"
            )
        
        if not np.all(np.isfinite(reference)):
            nan_count = np.sum(np.isnan(reference))
            inf_count = np.sum(np.isinf(reference))
            raise NumericalError(
                f"reference image contains non-finite values. "
                f"NaN count: {nan_count}, Inf count: {inf_count}"
            )
        
        # Validate parameter lists
        if not isinstance(t_values, list) or not t_values:
            raise ValidationError(
                f"t_values must be a non-empty list, got {type(t_values)}"
            )
        
        if not isinstance(lambda_values, list) or not lambda_values:
            raise ValidationError(
                f"lambda_values must be a non-empty list, got {type(lambda_values)}"
            )
        
        if not isinstance(sigma_values, list) or not sigma_values:
            raise ValidationError(
                f"sigma_values must be a non-empty list, got {type(sigma_values)}"
            )
        
        results = []
        total_combinations = len(t_values) * len(lambda_values) * len(sigma_values)
        
        logger.info(
            f"Testing {total_combinations} parameter combinations: "
            f"{len(sigma_values)} sigma × {len(t_values)} t × {len(lambda_values)} lambda"
        )
        
        # Test all combinations (cartesian product)
        for sigma in sigma_values:
            for t in t_values:
                for lambda_param in lambda_values:
                    try:
                        # Create configuration
                        config = ProcessingConfig(
                            sigma=sigma,
                            t=t,
                            lambda_param=lambda_param,
                            omega=1.0  # Keep omega fixed at 1.0 as per paper
                        )
                        
                        # Process image with this configuration
                        result = self.processor.process(blurred, reference, config)
                        
                        # Store result with parameters
                        test_result = TestResult(
                            t=t,
                            lambda_param=lambda_param,
                            sigma=sigma,
                            sharpened_image=result.sharpened_image,
                            plcc=result.plcc,
                            srocc=result.srocc
                        )
                        
                        results.append(test_result)
                        
                        logger.debug(
                            f"Success: σ={sigma:.2f}, t={t:.2f}, λ={lambda_param:.2f} "
                            f"→ PLCC={result.plcc:.4f}, SROCC={result.srocc:.4f}"
                        )
                        
                    except Exception as e:
                        # Log warning and continue with next combination
                        logger.warning(
                            f"Failed for parameters (σ={sigma}, t={t}, λ={lambda_param}): {e}"
                        )
                        continue
        
        logger.info(
            f"Completed testing: {len(results)}/{total_combinations} combinations succeeded"
        )
        
        return results
    
    def find_optimal(
        self,
        results: List[TestResult],
        metric: str = 'plcc'
    ) -> TestResult:
        """
        Find optimal parameter combination based on metric.
        
        Searches through all test results to find the one with the highest
        value for the specified metric (PLCC or SROCC).
        
        Args:
            results: List of test results from test_parameters()
            metric: Metric to optimize ('plcc' or 'srocc')
            
        Returns:
            TestResult with highest metric value
            
        Raises:
            ValidationError: If results list is empty or metric is invalid
            
        Example:
            >>> optimal = tester.find_optimal(results, metric='plcc')
            >>> print(f"Best parameters: t={optimal.t}, λ={optimal.lambda_param}, σ={optimal.sigma}")
            >>> print(f"PLCC: {optimal.plcc:.4f}")
        """
        if not isinstance(results, list):
            raise ValidationError(
                f"results must be a list, got {type(results)}"
            )
        
        if not results:
            raise ValidationError("Cannot find optimal parameters: results list is empty")
        
        if not isinstance(metric, str):
            raise ValidationError(
                f"metric must be a string, got {type(metric)}"
            )
        
        if metric not in ['plcc', 'srocc']:
            raise ValidationError(
                f"Invalid metric '{metric}'. Must be 'plcc' or 'srocc'"
            )
        
        # Find result with maximum metric value
        if metric == 'plcc':
            optimal = max(results, key=lambda r: r.plcc)
            logger.info(
                f"Optimal parameters for PLCC: σ={optimal.sigma:.2f}, "
                f"t={optimal.t:.2f}, λ={optimal.lambda_param:.2f} "
                f"→ PLCC={optimal.plcc:.4f}"
            )
        else:  # metric == 'srocc'
            optimal = max(results, key=lambda r: r.srocc)
            logger.info(
                f"Optimal parameters for SROCC: σ={optimal.sigma:.2f}, "
                f"t={optimal.t:.2f}, λ={optimal.lambda_param:.2f} "
                f"→ SROCC={optimal.srocc:.4f}"
            )
        
        return optimal
