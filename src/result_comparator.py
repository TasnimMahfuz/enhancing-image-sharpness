"""Result comparison and visualization for image sharpness enhancement."""

from typing import List
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .data_models import TestResult
from .exceptions import ValidationError, FileLoadError, ProcessingError, NumericalError


class ResultComparator:
    """Generate visual and quantitative comparisons of sharpening results."""
    
    def create_comparison(
        self,
        blurred: np.ndarray,
        results: List[TestResult],
        output_path: str
    ) -> None:
        """
        Create side-by-side comparison visualization.
        
        Args:
            blurred: Original blurred image
            results: List of sharpening results to compare
            output_path: Path to save comparison image
            
        The visualization shows:
        - Blurred image on the left
        - Each result side-by-side
        - Labels with parameters (σ, t, λ)
        - PLCC and SROCC metrics
        - Highlighting for optimal result (t=0.6, λ=0)
            
        Raises:
            ValidationError: If inputs are invalid
            FileLoadError: If output file cannot be written
            ProcessingError: If visualization generation fails
        """
        # Validate inputs
        if not isinstance(blurred, np.ndarray):
            raise ValidationError(
                f"blurred must be a numpy array, got {type(blurred)}"
            )
        
        if blurred.size == 0:
            raise ValidationError(
                "blurred image is empty (size 0)"
            )
        
        if not results:
            raise ValidationError("No results provided for comparison")
        
        if not isinstance(results, list):
            raise ValidationError(
                f"results must be a list, got {type(results)}"
            )
        
        # Check for non-finite values in blurred image
        if not np.all(np.isfinite(blurred)):
            nan_count = np.sum(np.isnan(blurred))
            inf_count = np.sum(np.isinf(blurred))
            raise NumericalError(
                f"blurred image contains non-finite values. "
                f"NaN count: {nan_count}, Inf count: {inf_count}"
            )
        
        # Validate each result
        for idx, result in enumerate(results):
            if not isinstance(result, TestResult):
                raise ValidationError(
                    f"results[{idx}] must be a TestResult object, got {type(result)}"
                )
            
            if not isinstance(result.sharpened_image, np.ndarray):
                raise ValidationError(
                    f"results[{idx}].sharpened_image must be a numpy array"
                )
            
            if result.sharpened_image.size == 0:
                raise ValidationError(
                    f"results[{idx}].sharpened_image is empty (size 0)"
                )
            
            # Check for non-finite values
            if not np.all(np.isfinite(result.sharpened_image)):
                nan_count = np.sum(np.isnan(result.sharpened_image))
                inf_count = np.sum(np.isinf(result.sharpened_image))
                raise NumericalError(
                    f"results[{idx}].sharpened_image contains non-finite values. "
                    f"NaN count: {nan_count}, Inf count: {inf_count}"
                )
        
        try:
            # Convert blurred image to PIL format
            blurred_pil = self._array_to_pil(blurred)
        except Exception as e:
            raise ProcessingError(
                f"Failed to convert blurred image to PIL format: {str(e)}"
            )
        
        # Get dimensions
        img_height, img_width = blurred.shape[:2]
        label_height = 80  # Space for text labels
        
        # Calculate total width (blurred + all results)
        num_images = len(results) + 1  # +1 for blurred
        total_width = img_width * num_images
        total_height = img_height + label_height
        
        try:
            # Create canvas
            canvas = Image.new('RGB', (total_width, total_height), color='white')
        except Exception as e:
            raise ProcessingError(
                f"Failed to create canvas with size ({total_width}, {total_height}): {str(e)}"
            )
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()
            font_bold = ImageFont.load_default()
        
        try:
            draw = ImageDraw.Draw(canvas)
            
            # Place blurred image
            canvas.paste(blurred_pil, (0, label_height))
            
            # Draw label for blurred image
            label_text = "Blurred\n(Original)"
            self._draw_label(draw, label_text, 0, 0, img_width, label_height, 
                            font, is_optimal=False, border_color=None)
            
            # Place each result
            for idx, result in enumerate(results):
                x_offset = (idx + 1) * img_width
                
                # Convert result image to PIL
                try:
                    result_pil = self._array_to_pil(result.sharpened_image)
                except Exception as e:
                    raise ProcessingError(
                        f"Failed to convert result[{idx}] image to PIL format: {str(e)}"
                    )
                
                canvas.paste(result_pil, (x_offset, label_height))
                
                # Check if this is the optimal result (t=0.6, λ=0)
                is_optimal = (abs(result.t - 0.6) < 0.01 and 
                             abs(result.lambda_param - 0.0) < 0.01)
                
                # Create label with parameters and metrics
                label_text = (
                    f"σ={result.sigma:.1f}, t={result.t:.1f}, λ={result.lambda_param:.1f}\n"
                    f"PLCC={result.plcc:.4f}, SROCC={result.srocc:.4f}"
                )
                
                # Highlight optimal result with green border
                border_color = 'green' if is_optimal else None
                self._draw_label(draw, label_text, x_offset, 0, img_width, 
                               label_height, font, is_optimal, border_color)
        except ProcessingError:
            raise
        except Exception as e:
            raise ProcessingError(
                f"Failed to generate comparison visualization: {str(e)}"
            )
        
        # Save the comparison
        try:
            canvas.save(output_path)
        except Exception as e:
            raise FileLoadError(
                f"Failed to save comparison image to {output_path}: {str(e)}"
            )
    
    def generate_metrics_table(self, results: List[TestResult]) -> str:
        """
        Generate formatted table of metrics.
        
        Args:
            results: List of test results
            
        Returns:
            Formatted string table with parameters and metrics
            
        Raises:
            ValidationError: If results list is invalid
            
        The table includes columns: σ, t, λ, PLCC, SROCC
        Results are sorted by PLCC (descending)
        """
        if not isinstance(results, list):
            raise ValidationError(
                f"results must be a list, got {type(results)}"
            )
        
        if not results:
            return "No results to display"
        
        # Validate each result
        for idx, result in enumerate(results):
            if not isinstance(result, TestResult):
                raise ValidationError(
                    f"results[{idx}] must be a TestResult object, got {type(result)}"
                )
        
        # Sort by PLCC descending
        sorted_results = sorted(results, key=lambda r: r.plcc, reverse=True)
        
        # Build table header
        header = "┌" + "─" * 70 + "┐\n"
        header += "│" + " " * 20 + "Sharpening Results Metrics" + " " * 24 + "│\n"
        header += "├" + "─" * 70 + "┤\n"
        header += "│  σ    │   t   │   λ   │   PLCC   │  SROCC   │  Optimal  │\n"
        header += "├" + "─" * 70 + "┤\n"
        
        # Build table rows
        rows = []
        for result in sorted_results:
            is_optimal = (abs(result.t - 0.6) < 0.01 and 
                         abs(result.lambda_param - 0.0) < 0.01)
            optimal_mark = "   ✓   " if is_optimal else "        "
            
            row = (
                f"│ {result.sigma:4.1f} │ {result.t:5.2f} │ "
                f"{result.lambda_param:5.1f} │ {result.plcc:8.4f} │ "
                f"{result.srocc:8.4f} │{optimal_mark}│"
            )
            rows.append(row)
        
        # Build table footer
        footer = "└" + "─" * 70 + "┘"
        
        # Combine all parts
        table = header + "\n".join(rows) + "\n" + footer
        return table
    
    def _array_to_pil(self, img_array: np.ndarray) -> Image.Image:
        """
        Convert numpy array to PIL Image.
        
        Args:
            img_array: Numpy array (H, W) or (H, W, C)
            
        Returns:
            PIL Image
            
        Raises:
            ValidationError: If image array is invalid
            ProcessingError: If conversion fails
        """
        if not isinstance(img_array, np.ndarray):
            raise ValidationError(
                f"img_array must be a numpy array, got {type(img_array)}"
            )
        
        if img_array.size == 0:
            raise ValidationError(
                "img_array is empty (size 0)"
            )
        
        if img_array.ndim not in [2, 3]:
            raise ValidationError(
                f"Image must be 2D (grayscale) or 3D (color), got {img_array.ndim}D"
            )
        
        # Check for non-finite values
        if not np.all(np.isfinite(img_array)):
            nan_count = np.sum(np.isnan(img_array))
            inf_count = np.sum(np.isinf(img_array))
            raise NumericalError(
                f"img_array contains non-finite values. "
                f"NaN count: {nan_count}, Inf count: {inf_count}"
            )
        
        try:
            # Ensure uint8 type
            if img_array.dtype != np.uint8:
                img_array = np.clip(img_array, 0, 255).astype(np.uint8)
            
            # Handle grayscale
            if img_array.ndim == 2:
                return Image.fromarray(img_array, mode='L')
            # Handle color
            elif img_array.ndim == 3:
                return Image.fromarray(img_array, mode='RGB')
        except Exception as e:
            raise ProcessingError(
                f"Failed to convert numpy array to PIL Image: {str(e)}"
            )
    
    def _draw_label(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int,
        y: int,
        width: int,
        height: int,
        font: ImageFont.ImageFont,
        is_optimal: bool,
        border_color: str = None
    ) -> None:
        """
        Draw text label with optional border.
        
        Args:
            draw: PIL ImageDraw object
            text: Text to draw
            x, y: Top-left corner position
            width, height: Label dimensions
            font: Font to use
            is_optimal: Whether this is the optimal result
            border_color: Color for border (None for no border)
        """
        # Draw border if specified
        if border_color:
            # Draw thick border
            border_width = 3
            for i in range(border_width):
                draw.rectangle(
                    [x + i, y + i, x + width - i - 1, y + height - i - 1],
                    outline=border_color
                )
        
        # Draw background
        bg_color = '#e8f5e9' if is_optimal else 'white'
        draw.rectangle(
            [x + 3, y + 3, x + width - 4, y + height - 4],
            fill=bg_color
        )
        
        # Draw text (centered)
        lines = text.split('\n')
        y_offset = y + 10
        for line in lines:
            # Get text bounding box for centering
            bbox = draw.textbbox((0, 0), line, font=font)
            text_width = bbox[2] - bbox[0]
            text_x = x + (width - text_width) // 2
            
            draw.text((text_x, y_offset), line, fill='black', font=font)
            y_offset += 20
        
        # Add "OPTIMAL" badge if applicable
        if is_optimal:
            badge_text = "★ OPTIMAL ★"
            bbox = draw.textbbox((0, 0), badge_text, font=font)
            badge_width = bbox[2] - bbox[0]
            badge_x = x + (width - badge_width) // 2
            badge_y = y + height - 25
            
            # Draw badge background
            draw.rectangle(
                [badge_x - 5, badge_y - 2, badge_x + badge_width + 5, badge_y + 15],
                fill='green'
            )
            draw.text((badge_x, badge_y), badge_text, fill='white', font=font)
