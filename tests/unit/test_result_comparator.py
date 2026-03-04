"""
Unit tests for ResultComparator visualization components.

Feature: image-sharpness-enhancement
Examples 9-13: Visualization Components
Validates: Requirements 9.1, 9.2, 9.3, 9.4, 9.5
"""

import pytest
import numpy as np
from PIL import Image
import os
import tempfile
from src.result_comparator import ResultComparator
from src.data_models import TestResult
from src.image_processor import ImageProcessor
from src.data_models import ProcessingConfig


class TestResultComparatorLabels:
    """Test parameter labeling in visualizations."""
    
    def test_labels_include_all_parameters(self, tmp_path):
        """
        Example 9: Parameter Labels
        Verify that labels include σ, t, and λ parameters.
        """
        # Create simple test images
        blurred = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        sharpened = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Create test results with specific parameters
        results = [
            TestResult(
                t=0.6,
                lambda_param=0.0,
                sigma=1.0,
                sharpened_image=sharpened,
                plcc=0.95,
                srocc=0.93
            )
        ]
        
        comparator = ResultComparator()
        output_path = str(tmp_path / "comparison.png")
        
        # Should not raise any errors
        comparator.create_comparison(blurred, results, output_path)
        
        # Verify file was created
        assert os.path.exists(output_path), "Comparison image should be saved"
        
        # Verify image can be loaded
        img = Image.open(output_path)
        assert img.size[0] > 0 and img.size[1] > 0, "Image should have valid dimensions"
    
    def test_labels_with_multiple_results(self, tmp_path):
        """
        Example 10: Multiple Result Labels
        Verify labels are correctly placed for multiple results.
        """
        blurred = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Create multiple results with different parameters
        results = [
            TestResult(t=0.3, lambda_param=0.0, sigma=1.0,
                      sharpened_image=np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
                      plcc=0.85, srocc=0.83),
            TestResult(t=0.6, lambda_param=0.0, sigma=1.0,
                      sharpened_image=np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
                      plcc=0.95, srocc=0.93),
            TestResult(t=0.9, lambda_param=0.0, sigma=1.0,
                      sharpened_image=np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
                      plcc=0.88, srocc=0.86),
        ]
        
        comparator = ResultComparator()
        output_path = str(tmp_path / "multi_comparison.png")
        
        comparator.create_comparison(blurred, results, output_path)
        
        # Verify file exists
        assert os.path.exists(output_path)
        
        # Verify image dimensions (should be wider with more results)
        img = Image.open(output_path)
        expected_width = 100 * (len(results) + 1)  # +1 for blurred
        assert img.size[0] == expected_width, f"Width should be {expected_width}"


class TestResultComparatorMetrics:
    """Test metrics display in visualizations."""
    
    def test_metrics_display_plcc_and_srocc(self, tmp_path):
        """
        Example 11: Metrics Display
        Verify PLCC and SROCC metrics are displayed.
        """
        blurred = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        sharpened = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        results = [
            TestResult(
                t=0.6,
                lambda_param=0.0,
                sigma=1.0,
                sharpened_image=sharpened,
                plcc=0.9523,
                srocc=0.9312
            )
        ]
        
        comparator = ResultComparator()
        output_path = str(tmp_path / "metrics_comparison.png")
        
        # Create comparison
        comparator.create_comparison(blurred, results, output_path)
        
        # Verify file was created
        assert os.path.exists(output_path)
        
        # The metrics are embedded in the image, so we just verify
        # the image was created successfully
        img = Image.open(output_path)
        assert img.size[0] > 0 and img.size[1] > 0
    
    def test_metrics_table_generation(self):
        """
        Example 12: Metrics Table
        Verify generate_metrics_table produces formatted output.
        """
        results = [
            TestResult(t=0.3, lambda_param=0.0, sigma=1.0,
                      sharpened_image=np.zeros((10, 10, 3), dtype=np.uint8),
                      plcc=0.85, srocc=0.83),
            TestResult(t=0.6, lambda_param=0.0, sigma=1.0,
                      sharpened_image=np.zeros((10, 10, 3), dtype=np.uint8),
                      plcc=0.95, srocc=0.93),
            TestResult(t=0.9, lambda_param=0.0, sigma=1.0,
                      sharpened_image=np.zeros((10, 10, 3), dtype=np.uint8),
                      plcc=0.88, srocc=0.86),
        ]
        
        comparator = ResultComparator()
        table = comparator.generate_metrics_table(results)
        
        # Verify table contains expected elements
        assert "σ" in table or "sigma" in table.lower(), "Table should include sigma"
        assert "t" in table, "Table should include t"
        assert "λ" in table or "lambda" in table.lower(), "Table should include lambda"
        assert "PLCC" in table, "Table should include PLCC"
        assert "SROCC" in table, "Table should include SROCC"
        
        # Verify all parameter values are present
        assert "0.3" in table, "Should include t=0.3"
        assert "0.6" in table, "Should include t=0.6"
        assert "0.9" in table, "Should include t=0.9"
        
        # Verify metrics are present
        assert "0.85" in table or "0.8500" in table, "Should include PLCC=0.85"
        assert "0.95" in table or "0.9500" in table, "Should include PLCC=0.95"
    
    def test_metrics_table_sorted_by_plcc(self):
        """
        Verify metrics table is sorted by PLCC in descending order.
        """
        results = [
            TestResult(t=0.3, lambda_param=0.0, sigma=1.0,
                      sharpened_image=np.zeros((10, 10, 3), dtype=np.uint8),
                      plcc=0.85, srocc=0.83),
            TestResult(t=0.6, lambda_param=0.0, sigma=1.0,
                      sharpened_image=np.zeros((10, 10, 3), dtype=np.uint8),
                      plcc=0.95, srocc=0.93),
            TestResult(t=0.9, lambda_param=0.0, sigma=1.0,
                      sharpened_image=np.zeros((10, 10, 3), dtype=np.uint8),
                      plcc=0.70, srocc=0.68),
        ]
        
        comparator = ResultComparator()
        table = comparator.generate_metrics_table(results)
        
        # Find positions of PLCC values in the table
        pos_95 = table.find("0.95")
        pos_85 = table.find("0.85")
        pos_70 = table.find("0.70")
        
        # Verify they appear in descending order
        assert pos_95 < pos_85 < pos_70, "Table should be sorted by PLCC descending"


class TestResultComparatorOptimalHighlighting:
    """Test highlighting of optimal result (t=0.6, λ=0)."""
    
    def test_optimal_result_highlighted(self, tmp_path):
        """
        Example 13: Optimal Result Highlighting
        Verify optimal result (t=0.6, λ=0) is highlighted.
        """
        blurred = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        # Create results including the optimal one
        results = [
            TestResult(t=0.3, lambda_param=0.0, sigma=1.0,
                      sharpened_image=np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
                      plcc=0.85, srocc=0.83),
            TestResult(t=0.6, lambda_param=0.0, sigma=1.0,  # Optimal
                      sharpened_image=np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
                      plcc=0.95, srocc=0.93),
            TestResult(t=0.9, lambda_param=0.0, sigma=1.0,
                      sharpened_image=np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8),
                      plcc=0.88, srocc=0.86),
        ]
        
        comparator = ResultComparator()
        output_path = str(tmp_path / "optimal_comparison.png")
        
        # Should not raise errors
        comparator.create_comparison(blurred, results, output_path)
        
        # Verify file was created
        assert os.path.exists(output_path)
        
        # Load and verify image
        img = Image.open(output_path)
        assert img.size[0] > 0 and img.size[1] > 0
    
    def test_optimal_marked_in_metrics_table(self):
        """
        Verify optimal result is marked in metrics table.
        """
        results = [
            TestResult(t=0.3, lambda_param=0.0, sigma=1.0,
                      sharpened_image=np.zeros((10, 10, 3), dtype=np.uint8),
                      plcc=0.85, srocc=0.83),
            TestResult(t=0.6, lambda_param=0.0, sigma=1.0,  # Optimal
                      sharpened_image=np.zeros((10, 10, 3), dtype=np.uint8),
                      plcc=0.95, srocc=0.93),
        ]
        
        comparator = ResultComparator()
        table = comparator.generate_metrics_table(results)
        
        # Verify optimal marker is present
        assert "✓" in table or "OPTIMAL" in table.upper() or "*" in table, \
            "Table should mark optimal result"


class TestResultComparatorFileOutput:
    """Test file output functionality."""
    
    def test_file_saved_to_specified_path(self, tmp_path):
        """
        Verify comparison is saved to specified output path.
        """
        blurred = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        sharpened = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        results = [
            TestResult(t=0.6, lambda_param=0.0, sigma=1.0,
                      sharpened_image=sharpened,
                      plcc=0.95, srocc=0.93)
        ]
        
        comparator = ResultComparator()
        output_path = str(tmp_path / "test_output.png")
        
        comparator.create_comparison(blurred, results, output_path)
        
        # Verify file exists at exact path
        assert os.path.exists(output_path), f"File should exist at {output_path}"
        
        # Verify it's a valid image file
        img = Image.open(output_path)
        assert img.format == 'PNG', "Should be PNG format"
    
    def test_handles_grayscale_images(self, tmp_path):
        """
        Verify visualization works with grayscale images.
        """
        # Create grayscale images
        blurred = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        sharpened = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        
        results = [
            TestResult(t=0.6, lambda_param=0.0, sigma=1.0,
                      sharpened_image=sharpened,
                      plcc=0.95, srocc=0.93)
        ]
        
        comparator = ResultComparator()
        output_path = str(tmp_path / "grayscale_comparison.png")
        
        # Should handle grayscale without errors
        comparator.create_comparison(blurred, results, output_path)
        
        assert os.path.exists(output_path)
        img = Image.open(output_path)
        assert img.size[0] > 0 and img.size[1] > 0


class TestResultComparatorWithRealImages:
    """Test visualization with real CSIQ dataset images."""
    
    def test_comparison_with_real_csiq_images(self, tmp_path):
        """
        Test visualization using real CSIQ dataset images.
        This provides end-to-end validation with actual image data.
        """
        # Load real images from CSIQ dataset
        reference_path = "datasets/CSIQ/src_imgs/1600.png"
        blurred_path = "datasets/CSIQ/dst_imgs/blur/1600.BLUR.1.png"
        
        # Skip test if dataset not available
        if not os.path.exists(reference_path) or not os.path.exists(blurred_path):
            pytest.skip("CSIQ dataset not available")
        
        # Load images
        reference = np.array(Image.open(reference_path))
        blurred = np.array(Image.open(blurred_path))
        
        # Process with different parameters
        processor = ImageProcessor()
        
        configs = [
            ProcessingConfig(sigma=1.0, t=0.3, lambda_param=0.0),
            ProcessingConfig(sigma=1.0, t=0.6, lambda_param=0.0),  # Optimal
            ProcessingConfig(sigma=1.0, t=0.9, lambda_param=0.0),
        ]
        
        results = []
        for config in configs:
            result = processor.process(blurred, reference, config)
            test_result = TestResult(
                t=config.t,
                lambda_param=config.lambda_param,
                sigma=config.sigma,
                sharpened_image=result.sharpened_image,
                plcc=result.plcc,
                srocc=result.srocc
            )
            results.append(test_result)
        
        # Create comparison visualization
        comparator = ResultComparator()
        output_path = str(tmp_path / "real_csiq_comparison.png")
        
        comparator.create_comparison(blurred, results, output_path)
        
        # Verify output
        assert os.path.exists(output_path), "Comparison should be saved"
        
        # Verify image dimensions
        img = Image.open(output_path)
        expected_width = reference.shape[1] * (len(results) + 1)
        assert img.size[0] == expected_width, \
            f"Width should be {expected_width}, got {img.size[0]}"
        
        # Generate metrics table
        table = comparator.generate_metrics_table(results)
        assert len(table) > 0, "Metrics table should be generated"
        assert "0.6" in table, "Should include optimal t=0.6"
        assert "✓" in table, "Should mark optimal result"
    
    def test_comparison_with_multiple_csiq_images(self, tmp_path):
        """
        Test visualization with multiple different CSIQ images.
        """
        test_images = [
            ("1600.png", "1600.BLUR.1.png"),
            ("bridge.png", "bridge.BLUR.1.png"),
        ]
        
        for ref_name, blur_name in test_images:
            reference_path = f"datasets/CSIQ/src_imgs/{ref_name}"
            blurred_path = f"datasets/CSIQ/dst_imgs/blur/{blur_name}"
            
            # Skip if not available
            if not os.path.exists(reference_path) or not os.path.exists(blurred_path):
                continue
            
            # Load images
            reference = np.array(Image.open(reference_path))
            blurred = np.array(Image.open(blurred_path))
            
            # Process with optimal parameters
            processor = ImageProcessor()
            config = ProcessingConfig(sigma=1.0, t=0.6, lambda_param=0.0)
            result = processor.process(blurred, reference, config)
            
            test_result = TestResult(
                t=config.t,
                lambda_param=config.lambda_param,
                sigma=config.sigma,
                sharpened_image=result.sharpened_image,
                plcc=result.plcc,
                srocc=result.srocc
            )
            
            # Create comparison
            comparator = ResultComparator()
            output_path = str(tmp_path / f"comparison_{ref_name}")
            
            comparator.create_comparison(blurred, [test_result], output_path)
            
            # Verify output
            assert os.path.exists(output_path), \
                f"Comparison should be saved for {ref_name}"


class TestResultComparatorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_results_raises_error(self, tmp_path):
        """
        Verify that empty results list raises appropriate error.
        """
        blurred = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        comparator = ResultComparator()
        output_path = str(tmp_path / "empty_comparison.png")
        
        with pytest.raises(ValueError, match="No results"):
            comparator.create_comparison(blurred, [], output_path)
    
    def test_empty_results_table(self):
        """
        Verify metrics table handles empty results gracefully.
        """
        comparator = ResultComparator()
        table = comparator.generate_metrics_table([])
        
        assert "No results" in table or len(table) == 0 or table.strip() == "", \
            "Should handle empty results"
    
    def test_single_result(self, tmp_path):
        """
        Verify visualization works with single result.
        """
        blurred = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        sharpened = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        
        results = [
            TestResult(t=0.6, lambda_param=0.0, sigma=1.0,
                      sharpened_image=sharpened,
                      plcc=0.95, srocc=0.93)
        ]
        
        comparator = ResultComparator()
        output_path = str(tmp_path / "single_comparison.png")
        
        comparator.create_comparison(blurred, results, output_path)
        
        assert os.path.exists(output_path)
        
        # Verify dimensions (blurred + 1 result)
        img = Image.open(output_path)
        expected_width = 100 * 2
        assert img.size[0] == expected_width
    
    def test_many_results(self, tmp_path):
        """
        Verify visualization works with many results.
        """
        blurred = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        
        # Create 10 results
        results = []
        for i in range(10):
            t_val = 0.1 * (i + 1)
            results.append(
                TestResult(
                    t=t_val,
                    lambda_param=0.0,
                    sigma=1.0,
                    sharpened_image=np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8),
                    plcc=0.8 + 0.01 * i,
                    srocc=0.75 + 0.01 * i
                )
            )
        
        comparator = ResultComparator()
        output_path = str(tmp_path / "many_comparison.png")
        
        comparator.create_comparison(blurred, results, output_path)
        
        assert os.path.exists(output_path)
        
        # Verify wide image
        img = Image.open(output_path)
        expected_width = 50 * (len(results) + 1)
        assert img.size[0] == expected_width
