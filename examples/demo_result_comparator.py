"""
Demonstration of ResultComparator functionality.

This script shows how to use the ResultComparator to:
1. Create side-by-side visual comparisons
2. Generate formatted metrics tables
3. Highlight optimal results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image
from src.result_comparator import ResultComparator
from src.image_processor import ImageProcessor
from src.data_models import ProcessingConfig, TestResult


def demo_with_real_images():
    """Demonstrate ResultComparator with real CSIQ images."""
    
    print("=" * 70)
    print("ResultComparator Demonstration")
    print("=" * 70)
    
    # Load real images from CSIQ dataset
    reference_path = "datasets/CSIQ/src_imgs/boston.png"
    blurred_path = "datasets/CSIQ/dst_imgs/blur/boston.BLUR.1.png"
    
    if not os.path.exists(reference_path) or not os.path.exists(blurred_path):
        print("ERROR: CSIQ dataset not found. Please ensure dataset is available.")
        return
    
    print(f"\nLoading images:")
    print(f"  Reference: {reference_path}")
    print(f"  Blurred:   {blurred_path}")
    
    reference = np.array(Image.open(reference_path))
    blurred = np.array(Image.open(blurred_path))
    
    print(f"  Image shape: {reference.shape}")
    
    # Process with different parameter combinations
    print("\nProcessing with different parameters...")
    processor = ImageProcessor()
    
    configs = [
        ProcessingConfig(sigma=1.0, t=0.3, lambda_param=0.0),
        ProcessingConfig(sigma=1.0, t=0.6, lambda_param=0.0),  # Optimal (paper's recommendation)
        ProcessingConfig(sigma=1.0, t=0.9, lambda_param=0.0),
        # Additional lambda variations
        ProcessingConfig(sigma=1.0, t=0.6, lambda_param=0.1),
        ProcessingConfig(sigma=1.0, t=0.6, lambda_param=1.0),
        ProcessingConfig(sigma=1.0, t=0.6, lambda_param=-0.1),
    ]
    
    results = []
    for i, config in enumerate(configs, 1):
        print(f"  [{i}/{len(configs)}] Processing with σ={config.sigma}, t={config.t}, λ={config.lambda_param}...")
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
        print(f"      PLCC: {result.plcc:.4f}, SROCC: {result.srocc:.4f}")
    
    # Create comparison visualization
    print("\nCreating visual comparison...")
    comparator = ResultComparator()
    output_path = "output/comparison_demo.png"
    
    # Ensure output directory exists
    os.makedirs("output", exist_ok=True)
    
    comparator.create_comparison(blurred, results, output_path)
    print(f"  ✓ Saved to: {output_path}")
    
    # Generate metrics table
    print("\nGenerating metrics table...")
    table = comparator.generate_metrics_table(results)
    print(table)
    
    print("\n" + "=" * 70)
    print("Demonstration complete!")
    print("=" * 70)
    print(f"\nView the comparison image at: {output_path}")
    print("\nKey features demonstrated:")
    print("  • Side-by-side comparison of blurred and sharpened images")
    print("  • Parameter labels (σ, t, λ) for each result")
    print("  • PLCC and SROCC metrics displayed")
    print("  • Optimal result (t=0.6, λ=0) highlighted with green border and ★ badge")
    print("  • Formatted metrics table sorted by PLCC")


def demo_metrics_table_only():
    """Demonstrate metrics table generation without image processing."""
    
    print("\n" + "=" * 70)
    print("Metrics Table Demonstration (Synthetic Data)")
    print("=" * 70)
    
    # Create synthetic results
    results = [
        TestResult(t=0.2, lambda_param=0.0, sigma=1.0,
                  sharpened_image=np.zeros((10, 10, 3), dtype=np.uint8),
                  plcc=0.82, srocc=0.80),
        TestResult(t=0.4, lambda_param=0.0, sigma=1.0,
                  sharpened_image=np.zeros((10, 10, 3), dtype=np.uint8),
                  plcc=0.91, srocc=0.89),
        TestResult(t=0.6, lambda_param=0.0, sigma=1.0,  # Optimal
                  sharpened_image=np.zeros((10, 10, 3), dtype=np.uint8),
                  plcc=0.95, srocc=0.93),
        TestResult(t=0.8, lambda_param=0.0, sigma=1.0,
                  sharpened_image=np.zeros((10, 10, 3), dtype=np.uint8),
                  plcc=0.88, srocc=0.86),
        TestResult(t=1.0, lambda_param=0.0, sigma=1.0,
                  sharpened_image=np.zeros((10, 10, 3), dtype=np.uint8),
                  plcc=0.79, srocc=0.77),
    ]
    
    comparator = ResultComparator()
    table = comparator.generate_metrics_table(results)
    
    print("\n" + table)
    print("\nNote: Results are sorted by PLCC in descending order.")
    print("      The optimal result (t=0.6, λ=0) is marked with ✓")


if __name__ == "__main__":
    # Run demonstration with real images
    demo_with_real_images()
    
    # Also show metrics table with synthetic data
    demo_metrics_table_only()
