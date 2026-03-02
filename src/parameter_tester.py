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
            ValueError: If results list is empty or metric is invalid
            
        Example:
            >>> optimal = tester.find_optimal(results, metric='plcc')
            >>> print(f"Best parameters: t={optimal.t}, λ={optimal.lambda_param}, σ={optimal.sigma}")
            >>> print(f"PLCC: {optimal.plcc:.4f}")
        """
        if not results:
            raise ValueError("Cannot find optimal parameters: results list is empty")
        
        if metric not in ['plcc', 'srocc']:
            raise ValueError(
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
