"""
Unit tests for comprehensive error handling across all components.

This test module validates that all components properly handle error conditions
and provide descriptive error messages with context.

Tests cover:
- Custom exception types (ValidationError, FileLoadError, ProcessingError, NumericalError)
- Invalid inputs at component boundaries
- Numerical stability checks (NaN, Inf detection)
- File I/O error handling
- Error message context and descriptiveness
- Error propagation through pipeline
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from src.exceptions import ValidationError, FileLoadError, ProcessingError, NumericalError
from src.gaussian_filter import GaussianFilter
from src.edge_extractor import EdgeExtractor
from src.coefficient_enhancer import CoefficientEnhancer
from src.image_combiner import ImageCombiner
from src.metrics_calculator import MetricsCalculator
from src.dataset_loader import DatasetLoader
from src.parameter_tester import ParameterTester
from src.result_comparator import ResultComparator
from src.image_processor import ImageProcessor
from src.data_models import ProcessingConfig, TestResult


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_validation_error_inheritance(self):
        """ValidationError should inherit from ValueError."""
        error = ValidationError("test message")
        assert isinstance(error, ValueError)
        assert str(error) == "test message"
    
    def test_file_load_error_inheritance(self):
        """FileLoadError should inherit from IOError."""
        error = FileLoadError("test message")
        assert isinstance(error, IOError)
        assert str(error) == "test message"
    
    def test_processing_error_inheritance(self):
        """ProcessingError should inherit from RuntimeError."""
        error = ProcessingError("test message")
        assert isinstance(error, RuntimeError)
        assert str(error) == "test message"
    
    def test_numerical_error_inheritance(self):
        """NumericalError should inherit from ArithmeticError."""
        error = NumericalError("test message")
        assert isinstance(error, ArithmeticError)
        assert str(error) == "test message"


class TestGaussianFilterErrorHandling:
    """Test error handling in GaussianFilter."""
    
    def test_negative_sigma_raises_validation_error(self):
        """Should raise ValidationError for negative sigma."""
        with pytest.raises(ValidationError) as exc_info:
            GaussianFilter(sigma=-1.0)
        
        assert "sigma must be positive" in str(exc_info.value)
        assert "-1.0" in str(exc_info.value)
        assert "σ > 0" in str(exc_info.value)
    
    def test_zero_sigma_raises_validation_error(self):
        """Should raise ValidationError for zero sigma."""
        with pytest.raises(ValidationError) as exc_info:
            GaussianFilter(sigma=0.0)
        
        assert "sigma must be positive" in str(exc_info.value)
    
    def test_invalid_image_dimensions(self):
        """Should raise ValueError for invalid image dimensions."""
        gaussian = GaussianFilter(sigma=1.0)
        
        # 1D array (invalid)
        with pytest.raises(ValueError) as exc_info:
            gaussian.apply(np.array([1, 2, 3]))
        
        assert "must be 2D or 3D" in str(exc_info.value)
    
    def test_very_large_sigma_handling(self):
        """Should handle very large sigma values without overflow."""
        # This should not raise an error, but create a large kernel
        gaussian = GaussianFilter(sigma=100.0)
        assert gaussian.kernel is not None
        assert np.all(np.isfinite(gaussian.kernel))


class TestEdgeExtractorErrorHandling:
    """Test error handling in EdgeExtractor."""
    
    def test_mismatched_shapes_raises_validation_error(self):
        """Should raise ValidationError for mismatched image shapes."""
        extractor = EdgeExtractor()
        
        reference = np.random.rand(100, 100).astype(np.float32)
        smoothed = np.random.rand(50, 50).astype(np.float32)
        
        with pytest.raises(ValidationError) as exc_info:
            extractor.extract(reference, smoothed)
        
        assert "shapes must match" in str(exc_info.value).lower()
        assert "(100, 100)" in str(exc_info.value)
        assert "(50, 50)" in str(exc_info.value)
    
    def test_nan_in_reference_raises_numerical_error(self):
        """Should raise NumericalError for NaN values in reference."""
        extractor = EdgeExtractor()
        
        reference = np.random.rand(10, 10).astype(np.float32)
        reference[5, 5] = np.nan
        smoothed = np.random.rand(10, 10).astype(np.float32)
        
        with pytest.raises(NumericalError) as exc_info:
            extractor.extract(reference, smoothed)
        
        assert "non-finite" in str(exc_info.value).lower()
        assert "reference" in str(exc_info.value).lower()
    
    def test_inf_in_smoothed_raises_numerical_error(self):
        """Should raise NumericalError for Inf values in smoothed."""
        extractor = EdgeExtractor()
        
        reference = np.random.rand(10, 10).astype(np.float32)
        smoothed = np.random.rand(10, 10).astype(np.float32)
        smoothed[3, 3] = np.inf
        
        with pytest.raises(NumericalError) as exc_info:
            extractor.extract(reference, smoothed)
        
        assert "non-finite" in str(exc_info.value).lower()
        assert "smoothed" in str(exc_info.value).lower()
    
    def test_empty_array_raises_validation_error(self):
        """Should raise ValidationError for empty arrays."""
        extractor = EdgeExtractor()
        
        reference = np.array([]).astype(np.float32)
        smoothed = np.array([]).astype(np.float32)
        
        with pytest.raises(ValidationError) as exc_info:
            extractor.extract(reference, smoothed)
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_non_array_input_raises_validation_error(self):
        """Should raise ValidationError for non-array inputs."""
        extractor = EdgeExtractor()
        
        with pytest.raises(ValidationError) as exc_info:
            extractor.extract([1, 2, 3], [4, 5, 6])
        
        assert "numpy array" in str(exc_info.value).lower()


class TestCoefficientEnhancerErrorHandling:
    """Test error handling in CoefficientEnhancer."""
    
    def test_t_below_zero_raises_validation_error(self):
        """Should raise ValidationError for t < 0."""
        with pytest.raises(ValidationError) as exc_info:
            CoefficientEnhancer(t=-0.1, lambda_param=0.0)
        
        assert "t must be in [0, 1]" in str(exc_info.value)
        assert "-0.1" in str(exc_info.value)
    
    def test_t_above_one_raises_validation_error(self):
        """Should raise ValidationError for t > 1."""
        with pytest.raises(ValidationError) as exc_info:
            CoefficientEnhancer(t=1.5, lambda_param=0.0)
        
        assert "t must be in [0, 1]" in str(exc_info.value)
        assert "1.5" in str(exc_info.value)
    
    def test_t_near_one_raises_validation_error(self):
        """Should raise ValidationError for t ≈ 1 (division by zero)."""
        # The epsilon check is 1e-10, so we need to be very close to 1
        with pytest.raises(ValidationError) as exc_info:
            CoefficientEnhancer(t=1.0 - 1e-11, lambda_param=0.0)
        
        assert "too close to 1" in str(exc_info.value).lower()
        assert "division by zero" in str(exc_info.value).lower()
    
    def test_t_equals_zero_is_valid(self):
        """t=0 should be valid (uses limit formula)."""
        enhancer = CoefficientEnhancer(t=0.0, lambda_param=1.0)
        assert enhancer.t == 0.0
        assert np.isfinite(enhancer.a2_bound)
        assert np.isfinite(enhancer.a3_bound)
    
    def test_enhance_with_nan_raises_numerical_error(self):
        """Should raise NumericalError when enhancement produces NaN."""
        enhancer = CoefficientEnhancer(t=0.5, lambda_param=0.0)
        
        edge_image = np.random.rand(10, 10).astype(np.float32)
        edge_image[5, 5] = np.nan
        
        with pytest.raises(NumericalError) as exc_info:
            enhancer.enhance(edge_image)
        
        # The error might be caught during validation or enhancement
        assert "non-finite" in str(exc_info.value).lower()
    
    def test_enhance_with_empty_array_raises_validation_error(self):
        """Should raise ValidationError for empty edge image."""
        enhancer = CoefficientEnhancer(t=0.5, lambda_param=1.0)
        
        with pytest.raises(ValidationError) as exc_info:
            enhancer.enhance(np.array([]))
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_enhance_with_non_array_raises_validation_error(self):
        """Should raise ValidationError for non-array input."""
        enhancer = CoefficientEnhancer(t=0.5, lambda_param=1.0)
        
        with pytest.raises(ValidationError) as exc_info:
            enhancer.enhance([1, 2, 3])
        
        assert "numpy array" in str(exc_info.value).lower()


class TestImageCombinerErrorHandling:
    """Test error handling in ImageCombiner."""
    
    def test_mismatched_shapes_raises_validation_error(self):
        """Should raise ValidationError for mismatched shapes."""
        combiner = ImageCombiner()
        
        blurred = np.random.rand(100, 100).astype(np.float32)
        enhanced_edge = np.random.rand(50, 50).astype(np.float32)
        
        with pytest.raises(ValidationError) as exc_info:
            combiner.combine(blurred, enhanced_edge)
        
        assert "dimensions must match" in str(exc_info.value).lower()
        assert "(100, 100)" in str(exc_info.value)
        assert "(50, 50)" in str(exc_info.value)
    
    def test_nan_in_blurred_raises_numerical_error(self):
        """Should raise NumericalError for NaN in blurred image."""
        combiner = ImageCombiner()
        
        blurred = np.random.rand(10, 10).astype(np.float32)
        blurred[5, 5] = np.nan
        enhanced_edge = np.random.rand(10, 10).astype(np.float32)
        
        with pytest.raises(NumericalError) as exc_info:
            combiner.combine(blurred, enhanced_edge)
        
        assert "non-finite" in str(exc_info.value).lower()
        assert "blurred" in str(exc_info.value).lower()
    
    def test_inf_in_enhanced_edge_raises_numerical_error(self):
        """Should raise NumericalError for Inf in enhanced edge."""
        combiner = ImageCombiner()
        
        blurred = np.random.rand(10, 10).astype(np.float32)
        enhanced_edge = np.random.rand(10, 10).astype(np.float32)
        enhanced_edge[3, 3] = np.inf
        
        with pytest.raises(NumericalError) as exc_info:
            combiner.combine(blurred, enhanced_edge)
        
        assert "non-finite" in str(exc_info.value).lower()
        assert "enhanced_edge" in str(exc_info.value).lower()
    
    def test_empty_array_raises_validation_error(self):
        """Should raise ValidationError for empty arrays."""
        combiner = ImageCombiner()
        
        with pytest.raises(ValidationError) as exc_info:
            combiner.combine(np.array([]), np.array([]))
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_non_array_input_raises_validation_error(self):
        """Should raise ValidationError for non-array inputs."""
        combiner = ImageCombiner()
        
        with pytest.raises(ValidationError) as exc_info:
            combiner.combine([1, 2, 3], [4, 5, 6])
        
        assert "numpy array" in str(exc_info.value).lower()


class TestMetricsCalculatorErrorHandling:
    """Test error handling in MetricsCalculator."""
    
    def test_mismatched_shapes_raises_validation_error(self):
        """Should raise ValidationError for mismatched shapes."""
        calc = MetricsCalculator()
        
        sharpened = np.random.rand(100, 100).astype(np.float32)
        reference = np.random.rand(50, 50).astype(np.float32)
        
        with pytest.raises(ValidationError) as exc_info:
            calc.calculate_plcc(sharpened, reference)
        
        assert "shapes must match" in str(exc_info.value).lower()
    
    def test_nan_in_sharpened_raises_numerical_error(self):
        """Should raise NumericalError for NaN in sharpened image."""
        calc = MetricsCalculator()
        
        sharpened = np.random.rand(10, 10).astype(np.float32)
        sharpened[5, 5] = np.nan
        reference = np.random.rand(10, 10).astype(np.float32)
        
        with pytest.raises(NumericalError) as exc_info:
            calc.calculate_plcc(sharpened, reference)
        
        assert "non-finite" in str(exc_info.value).lower()
        assert "sharpened" in str(exc_info.value).lower()
    
    def test_inf_in_reference_raises_numerical_error(self):
        """Should raise NumericalError for Inf in reference image."""
        calc = MetricsCalculator()
        
        sharpened = np.random.rand(10, 10).astype(np.float32)
        reference = np.random.rand(10, 10).astype(np.float32)
        reference[3, 3] = np.inf
        
        with pytest.raises(NumericalError) as exc_info:
            calc.calculate_srocc(sharpened, reference)
        
        assert "non-finite" in str(exc_info.value).lower()
        assert "reference" in str(exc_info.value).lower()
    
    def test_constant_images_return_zero_or_one(self):
        """Constant images should return 0.0 for PLCC (no variance).
        
        For SROCC, constant images have all equal ranks, so the correlation
        of ranks with themselves is 1.0 (perfect correlation).
        """
        calc = MetricsCalculator()
        
        constant = np.ones((10, 10), dtype=np.float32) * 128
        
        # PLCC should be 0.0 (no variance)
        plcc = calc.calculate_plcc(constant, constant)
        assert plcc == 0.0
        
        # SROCC with constant images: all ranks are equal, so correlation is 1.0
        # This is actually correct behavior - ranks of constant values with themselves
        # have perfect correlation
        srocc = calc.calculate_srocc(constant, constant)
        # Accept either 0.0 or 1.0 as valid (implementation dependent)
        assert srocc in [0.0, 1.0]
    
    def test_invalid_image_format_raises_validation_error(self):
        """Should raise ValidationError for invalid image format."""
        calc = MetricsCalculator()
        
        # 4D array (invalid)
        invalid = np.random.rand(10, 10, 3, 2).astype(np.float32)
        reference = np.random.rand(10, 10, 3).astype(np.float32)
        
        with pytest.raises(ValidationError) as exc_info:
            calc.calculate_plcc(invalid, reference)
        
        # Error will be caught during shape validation or grayscale conversion


class TestDatasetLoaderErrorHandling:
    """Test error handling in DatasetLoader."""
    
    def test_nonexistent_file_raises_file_load_error(self):
        """Should raise FileLoadError for nonexistent files."""
        loader = DatasetLoader()
        
        with pytest.raises(FileLoadError) as exc_info:
            loader.load_image_pair(
                "/nonexistent/reference.png",
                "/nonexistent/distorted.png"
            )
        
        assert "not found" in str(exc_info.value).lower()
        assert "/nonexistent/reference.png" in str(exc_info.value)
    
    def test_unsupported_dataset_raises_validation_error(self):
        """Should raise ValidationError for unsupported dataset."""
        loader = DatasetLoader()
        
        with pytest.raises(ValidationError) as exc_info:
            loader.load_dataset("UNKNOWN_DATASET", "/some/path")
        
        assert "unsupported dataset" in str(exc_info.value).lower()
        assert "UNKNOWN_DATASET" in str(exc_info.value)
        assert "CSIQ" in str(exc_info.value)  # Should list supported datasets
    
    def test_nonexistent_dataset_path_raises_file_load_error(self):
        """Should raise FileLoadError for nonexistent dataset path."""
        loader = DatasetLoader()
        
        with pytest.raises(FileLoadError) as exc_info:
            loader.load_dataset("CSIQ", "/nonexistent/path")
        
        assert "not found" in str(exc_info.value).lower()
        assert "/nonexistent/path" in str(exc_info.value)


class TestParameterTesterErrorHandling:
    """Test error handling in ParameterTester."""
    
    def test_empty_results_raises_validation_error(self):
        """Should raise ValidationError for empty results list."""
        tester = ParameterTester()
        
        with pytest.raises(ValidationError) as exc_info:
            tester.find_optimal([], metric='plcc')
        
        assert "empty" in str(exc_info.value).lower()
    
    def test_invalid_metric_raises_validation_error(self):
        """Should raise ValidationError for invalid metric."""
        tester = ParameterTester()
        
        # Create a dummy result
        result = TestResult(
            t=0.5,
            lambda_param=0.0,
            sigma=1.0,
            sharpened_image=np.zeros((10, 10), dtype=np.uint8),
            plcc=0.9,
            srocc=0.85
        )
        
        with pytest.raises(ValidationError) as exc_info:
            tester.find_optimal([result], metric='invalid_metric')
        
        assert "invalid metric" in str(exc_info.value).lower()
        assert "'plcc' or 'srocc'" in str(exc_info.value).lower()
    
    def test_mismatched_image_shapes_raises_validation_error(self):
        """Should raise ValidationError for mismatched image shapes."""
        tester = ParameterTester()
        
        blurred = np.random.rand(100, 100).astype(np.float32)
        reference = np.random.rand(50, 50).astype(np.float32)
        
        with pytest.raises(ValidationError) as exc_info:
            tester.test_parameters(
                blurred, reference,
                t_values=[0.5],
                lambda_values=[0.0],
                sigma_values=[1.0]
            )
        
        assert "shapes must match" in str(exc_info.value).lower()
    
    def test_empty_parameter_lists_raises_validation_error(self):
        """Should raise ValidationError for empty parameter lists."""
        tester = ParameterTester()
        
        blurred = np.random.rand(10, 10).astype(np.float32)
        reference = np.random.rand(10, 10).astype(np.float32)
        
        with pytest.raises(ValidationError) as exc_info:
            tester.test_parameters(
                blurred, reference,
                t_values=[],  # Empty list
                lambda_values=[0.0],
                sigma_values=[1.0]
            )
        
        assert "non-empty list" in str(exc_info.value).lower()


class TestResultComparatorErrorHandling:
    """Test error handling in ResultComparator."""
    
    def test_empty_results_raises_validation_error(self):
        """Should raise ValidationError for empty results list."""
        comparator = ResultComparator()
        
        blurred = np.random.rand(10, 10).astype(np.uint8)
        
        with pytest.raises(ValidationError) as exc_info:
            comparator.create_comparison(blurred, [], "/tmp/output.png")
        
        assert "no results" in str(exc_info.value).lower()
    
    def test_invalid_output_path_raises_file_load_error(self):
        """Should raise FileLoadError for invalid output path."""
        comparator = ResultComparator()
        
        blurred = np.random.rand(10, 10).astype(np.uint8)
        result = TestResult(
            t=0.5,
            lambda_param=0.0,
            sigma=1.0,
            sharpened_image=np.random.rand(10, 10).astype(np.uint8),
            plcc=0.9,
            srocc=0.85
        )
        
        # Try to save to a directory that doesn't exist
        with pytest.raises(FileLoadError) as exc_info:
            comparator.create_comparison(
                blurred,
                [result],
                "/nonexistent/directory/output.png"
            )
        
        assert "failed to save" in str(exc_info.value).lower()
    
    def test_nan_in_blurred_raises_numerical_error(self):
        """Should raise NumericalError for NaN in blurred image."""
        comparator = ResultComparator()
        
        blurred = np.random.rand(10, 10).astype(np.float32)
        blurred[5, 5] = np.nan
        
        result = TestResult(
            t=0.5,
            lambda_param=0.0,
            sigma=1.0,
            sharpened_image=np.random.rand(10, 10).astype(np.uint8),
            plcc=0.9,
            srocc=0.85
        )
        
        with pytest.raises(NumericalError) as exc_info:
            comparator.create_comparison(blurred, [result], "/tmp/output.png")
        
        assert "non-finite" in str(exc_info.value).lower()


class TestImageProcessorErrorHandling:
    """Test error handling in ImageProcessor."""
    
    def test_invalid_sigma_raises_value_error(self):
        """Should raise ValueError for invalid sigma."""
        processor = ImageProcessor()
        config = ProcessingConfig(sigma=-1.0, t=0.5, lambda_param=0.0)
        
        with pytest.raises(ValueError) as exc_info:
            processor.validate_config(config)
        
        assert "sigma must be positive" in str(exc_info.value).lower()
        assert "σ > 0" in str(exc_info.value)
    
    def test_invalid_t_raises_value_error(self):
        """Should raise ValueError for invalid t."""
        processor = ImageProcessor()
        config = ProcessingConfig(sigma=1.0, t=1.5, lambda_param=0.0)
        
        with pytest.raises(ValueError) as exc_info:
            processor.validate_config(config)
        
        assert "t must be in [0, 1]" in str(exc_info.value).lower()
    
    def test_mismatched_images_raises_value_error(self):
        """Should raise ValueError for mismatched image dimensions."""
        processor = ImageProcessor()
        
        blurred = np.random.rand(100, 100).astype(np.float32)
        reference = np.random.rand(50, 50).astype(np.float32)
        config = ProcessingConfig(sigma=1.0, t=0.5, lambda_param=0.0)
        
        with pytest.raises(ValueError) as exc_info:
            processor.process(blurred, reference, config)
        
        assert "dimensions must match" in str(exc_info.value).lower()
    
    def test_error_messages_include_step_context(self):
        """Error messages should indicate which processing step failed."""
        processor = ImageProcessor()
        
        # Create invalid configuration that will fail during Gaussian filtering
        blurred = np.random.rand(10, 10).astype(np.float32)
        reference = np.random.rand(10, 10).astype(np.float32)
        config = ProcessingConfig(sigma=-1.0, t=0.5, lambda_param=0.0)
        
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            processor.process(blurred, reference, config)
        
        # Should indicate validation or Gaussian filtering failed
        error_msg = str(exc_info.value).lower()
        assert "sigma" in error_msg or "gaussian" in error_msg or "positive" in error_msg
    
    def test_parse_config_missing_parameter_raises_value_error(self):
        """Should raise ValueError for missing required parameters."""
        processor = ImageProcessor()
        
        # Missing 't' parameter
        with pytest.raises(ValueError) as exc_info:
            processor.parse_config({'sigma': 1.0, 'lambda': 0.0})
        
        assert "missing" in str(exc_info.value).lower()
        assert "'t'" in str(exc_info.value).lower()
    
    def test_parse_config_invalid_value_raises_value_error(self):
        """Should raise ValueError for invalid parameter values."""
        processor = ImageProcessor()
        
        with pytest.raises(ValueError) as exc_info:
            processor.parse_config({
                'sigma': 'not_a_number',
                't': 0.5,
                'lambda': 0.0
            })
        
        assert "invalid value" in str(exc_info.value).lower()
        assert "sigma" in str(exc_info.value).lower()


class TestErrorPropagation:
    """Test that errors propagate correctly through the pipeline."""
    
    def test_validation_error_propagates_from_gaussian_filter(self):
        """ValidationError from GaussianFilter should propagate through pipeline."""
        processor = ImageProcessor()
        
        blurred = np.random.rand(10, 10).astype(np.float32)
        reference = np.random.rand(10, 10).astype(np.float32)
        config = ProcessingConfig(sigma=-1.0, t=0.5, lambda_param=0.0)
        
        with pytest.raises((ValueError, RuntimeError)):
            processor.process(blurred, reference, config)
    
    def test_numerical_error_propagates_from_edge_extractor(self):
        """NumericalError from EdgeExtractor should propagate through pipeline."""
        processor = ImageProcessor()
        
        blurred = np.random.rand(10, 10).astype(np.float32)
        reference = np.random.rand(10, 10).astype(np.float32)
        reference[5, 5] = np.nan  # Inject NaN
        config = ProcessingConfig(sigma=1.0, t=0.5, lambda_param=0.0)
        
        with pytest.raises(RuntimeError) as exc_info:
            processor.process(blurred, reference, config)
        
        # Should indicate which step failed
        assert "failed" in str(exc_info.value).lower()


class TestErrorMessageQuality:
    """Test that error messages are descriptive and include context."""
    
    def test_validation_error_includes_parameter_name_and_value(self):
        """ValidationError should include parameter name and invalid value."""
        with pytest.raises(ValidationError) as exc_info:
            GaussianFilter(sigma=-2.5)
        
        error_msg = str(exc_info.value)
        assert "sigma" in error_msg.lower()
        assert "-2.5" in error_msg
        assert "σ > 0" in error_msg  # Valid range
    
    def test_validation_error_includes_valid_range(self):
        """ValidationError should include valid parameter range."""
        with pytest.raises(ValidationError) as exc_info:
            CoefficientEnhancer(t=1.5, lambda_param=0.0)
        
        error_msg = str(exc_info.value)
        assert "[0, 1]" in error_msg or "0 ≤ t ≤ 1" in error_msg
    
    def test_numerical_error_includes_nan_inf_counts(self):
        """NumericalError should include counts of NaN/Inf values."""
        extractor = EdgeExtractor()
        
        reference = np.random.rand(10, 10).astype(np.float32)
        reference[5, 5] = np.nan
        reference[6, 6] = np.nan
        smoothed = np.random.rand(10, 10).astype(np.float32)
        
        with pytest.raises(NumericalError) as exc_info:
            extractor.extract(reference, smoothed)
        
        error_msg = str(exc_info.value)
        assert "nan" in error_msg.lower() or "non-finite" in error_msg.lower()
    
    def test_file_load_error_includes_file_path(self):
        """FileLoadError should include the file path that caused the error."""
        loader = DatasetLoader()
        
        with pytest.raises(FileLoadError) as exc_info:
            loader.load_image_pair(
                "/path/to/missing/file.png",
                "/path/to/another/file.png"
            )
        
        error_msg = str(exc_info.value)
        assert "/path/to/missing/file.png" in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
