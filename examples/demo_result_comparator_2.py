"""
Comprehensive CSIQ Dataset Processing with ResultComparator.

This script processes all CSIQ dataset images with various distortion types
and generates:
1. Visual comparisons for each image
2. Average PLCC and SROCC scores for proposed parameters (σ=1.0, t=0.6, λ=0.0)
3. Organized output in output_2 folder
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image
from src.result_comparator import ResultComparator
from src.image_processor import ImageProcessor
from src.data_models import ProcessingConfig, TestResult


def get_all_csiq_images():
    """Get all source images and their distorted versions from CSIQ dataset."""
    
    src_dir = "datasets/CSIQ/src_imgs"
    dst_base = "datasets/CSIQ/dst_imgs"
    
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"CSIQ source directory not found: {src_dir}")
    
    # Get all source images
    src_images = [f for f in os.listdir(src_dir) if f.endswith('.png')]
    src_images.sort()
    
    # Distortion types available in CSIQ
    distortion_types = ['blur', 'awgn', 'contrast', 'fnoise', 'jpeg', 'jpeg2000']
    
    # Build list of (reference, distorted) pairs
    image_pairs = []
    
    for src_img in src_images:
        base_name = src_img.replace('.png', '')
        reference_path = os.path.join(src_dir, src_img)
        
        for dist_type in distortion_types:
            dist_dir = os.path.join(dst_base, dist_type)
            
            if not os.path.exists(dist_dir):
                continue
            
            # CSIQ has 5 distortion levels per type
            for level in range(1, 6):
                if dist_type == 'blur':
                    dist_name = f"{base_name}.BLUR.{level}.png"
                elif dist_type == 'awgn':
                    dist_name = f"{base_name}.AWGN.{level}.png"
                elif dist_type == 'contrast':
                    dist_name = f"{base_name}.CONTRAST.{level}.png"
                elif dist_type == 'fnoise':
                    dist_name = f"{base_name}.FNOISE.{level}.png"
                elif dist_type == 'jpeg':
                    dist_name = f"{base_name}.JPEG.{level}.png"
                elif dist_type == 'jpeg2000':
                    dist_name = f"{base_name}.JPEG2000.{level}.png"
                
                dist_path = os.path.join(dist_dir, dist_name)
                
                if os.path.exists(dist_path):
                    image_pairs.append({
                        'reference': reference_path,
                        'distorted': dist_path,
                        'base_name': base_name,
                        'distortion': dist_type,
                        'level': level
                    })
    
    return image_pairs


def process_all_images():
    """Process all CSIQ images and generate comparisons."""
    
    print("=" * 80)
    print("CSIQ Dataset Comprehensive Processing")
    print("=" * 80)
    
    # Get all image pairs
    print("\nScanning CSIQ dataset...")
    image_pairs = get_all_csiq_images()
    print(f"Found {len(image_pairs)} distorted images to process")
    
    # Create output directory
    output_dir = "output_2"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Initialize processor and comparator
    processor = ImageProcessor()
    comparator = ResultComparator()
    
    # Proposed parameters (from paper)
    proposed_config = ProcessingConfig(sigma=1.0, t=0.6, lambda_param=0.0)
    
    # Additional parameter variations for comparison
    configs = [
        ProcessingConfig(sigma=1.0, t=0.3, lambda_param=0.0),
        proposed_config,  # Optimal
        ProcessingConfig(sigma=1.0, t=0.9, lambda_param=0.0),
        ProcessingConfig(sigma=1.0, t=0.6, lambda_param=0.1),
        ProcessingConfig(sigma=1.0, t=0.6, lambda_param=-0.1),
    ]
    
    # Track metrics for proposed parameters
    proposed_plcc_scores = []
    proposed_srocc_scores = []
    
    # Process each image
    print("\nProcessing images...")
    print("-" * 80)
    
    for idx, pair in enumerate(image_pairs, 1):
        print(f"[{idx}/{len(image_pairs)}] {pair['base_name']} - "
              f"{pair['distortion'].upper()} level {pair['level']}")
        
        try:
            # Load images
            reference = np.array(Image.open(pair['reference']))
            distorted = np.array(Image.open(pair['distorted']))
            
            # Process with all configurations
            results = []
            proposed_result = None
            
            for config in configs:
                result = processor.process(distorted, reference, config)
                
                test_result = TestResult(
                    t=config.t,
                    lambda_param=config.lambda_param,
                    sigma=config.sigma,
                    sharpened_image=result.sharpened_image,
                    plcc=result.plcc,
                    srocc=result.srocc
                )
                results.append(test_result)
                
                # Track proposed parameters result
                if (abs(config.t - 0.6) < 0.01 and 
                    abs(config.lambda_param - 0.0) < 0.01):
                    proposed_result = result
            
            # Store proposed metrics
            if proposed_result:
                proposed_plcc_scores.append(proposed_result.plcc)
                proposed_srocc_scores.append(proposed_result.srocc)
                print(f"  Proposed (t=0.6, λ=0): PLCC={proposed_result.plcc:.4f}, "
                      f"SROCC={proposed_result.srocc:.4f}")
            
            # Create comparison visualization
            output_filename = (f"{pair['base_name']}_{pair['distortion']}_"
                             f"level{pair['level']}_comparison.png")
            output_path = os.path.join(output_dir, output_filename)
            
            comparator.create_comparison(distorted, results, output_path)
            
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            continue
    
    print("-" * 80)
    
    # Calculate and display average metrics
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    if proposed_plcc_scores:
        avg_plcc = np.mean(proposed_plcc_scores)
        avg_srocc = np.mean(proposed_srocc_scores)
        std_plcc = np.std(proposed_plcc_scores)
        std_srocc = np.std(proposed_srocc_scores)
        
        print(f"\nProposed Parameters (σ=1.0, t=0.6, λ=0.0):")
        print(f"  Images processed: {len(proposed_plcc_scores)}")
        print(f"  Average PLCC:     {avg_plcc:.4f} (±{std_plcc:.4f})")
        print(f"  Average SROCC:    {avg_srocc:.4f} (±{std_srocc:.4f})")
        print(f"  Min PLCC:         {min(proposed_plcc_scores):.4f}")
        print(f"  Max PLCC:         {max(proposed_plcc_scores):.4f}")
        print(f"  Min SROCC:        {min(proposed_srocc_scores):.4f}")
        print(f"  Max SROCC:        {max(proposed_srocc_scores):.4f}")
        
        # Save summary to file
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write("CSIQ Dataset Processing Summary\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Proposed Parameters: σ=1.0, t=0.6, λ=0.0\n\n")
            f.write(f"Images processed: {len(proposed_plcc_scores)}\n")
            f.write(f"Average PLCC:     {avg_plcc:.4f} (±{std_plcc:.4f})\n")
            f.write(f"Average SROCC:    {avg_srocc:.4f} (±{std_srocc:.4f})\n")
            f.write(f"Min PLCC:         {min(proposed_plcc_scores):.4f}\n")
            f.write(f"Max PLCC:         {max(proposed_plcc_scores):.4f}\n")
            f.write(f"Min SROCC:        {min(proposed_srocc_scores):.4f}\n")
            f.write(f"Max SROCC:        {max(proposed_srocc_scores):.4f}\n")
        
        print(f"\nSummary saved to: {summary_path}")
    else:
        print("\nNo results to summarize.")
    
    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"\nAll comparison images saved to: {output_dir}/")
    print(f"Total images processed: {len(proposed_plcc_scores)}")


if __name__ == "__main__":
    process_all_images()
