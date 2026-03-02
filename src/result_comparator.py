"""Result comparison and visualization for image sharpness enhancement."""

from typing import List
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .data_models import TestResult


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
        """
        if not results:
            raise ValueError("No results provided for comparison")
        
        # Convert blurred image to PIL format
        blurred_pil = self._array_to_pil(blurred)
        
        # Get dimensions
        img_height, img_width = blurred.shape[:2]
        label_height = 80  # Space for text labels
        
        # Calculate total width (blurred + all results)
        num_images = len(results) + 1  # +1 for blurred
        total_width = img_width * num_images
        total_height = img_height + label_height
        
        # Create canvas
        canvas = Image.new('RGB', (total_width, total_height), color='white')
        
        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except:
            font = ImageFont.load_default()
            font_bold = ImageFont.load_default()
        
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
            result_pil = self._array_to_pil(result.sharpened_image)
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
        
        # Save the comparison
        canvas.save(output_path)
    
    def generate_metrics_table(self, results: List[TestResult]) -> str:
        """
        Generate formatted table of metrics.
        
        Args:
            results: List of test results
            
        Returns:
            Formatted string table with parameters and metrics
            
        The table includes columns: σ, t, λ, PLCC, SROCC
        Results are sorted by PLCC (descending)
        """
        if not results:
            return "No results to display"
        
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
        """
        # Ensure uint8 type
        if img_array.dtype != np.uint8:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        # Handle grayscale
        if img_array.ndim == 2:
            return Image.fromarray(img_array, mode='L')
        # Handle color
        elif img_array.ndim == 3:
            return Image.fromarray(img_array, mode='RGB')
        else:
            raise ValueError(f"Unsupported image dimensions: {img_array.ndim}")
    
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
