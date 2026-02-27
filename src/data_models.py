"""Core data models for image sharpness enhancement system."""

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np


@dataclass
class ImagePair:
    """Pair of reference and distorted images."""
    reference: np.ndarray
    distorted: np.ndarray
    reference_path: str
    distorted_path: str


@dataclass
class ProcessingConfig:
    """Configuration for image sharpening."""
    sigma: float = 1.0          # Gaussian filter standard deviation (σ > 0)
    t: float = 0.6              # Coefficient parameter (0 ≤ t ≤ 1)
    lambda_param: float = 0.0   # Coefficient parameter (λ ∈ ℝ)
    omega: float = 1.0          # Coefficient parameter (Ω > 0)
    
    def validate(self) -> None:
        """Validate parameter ranges.
        
        Raises:
            ValueError: If any parameter is invalid
        """
        if self.sigma <= 0:
            raise ValueError(
                f"Gaussian sigma must be positive, got {self.sigma}. "
                f"Valid range: σ > 0"
            )
        if not 0 <= self.t <= 1:
            raise ValueError(
                f"Parameter t must be in [0, 1], got {self.t}. "
                f"Valid range: 0 ≤ t ≤ 1"
            )
        if self.omega <= 0:
            raise ValueError(
                f"Parameter omega must be positive, got {self.omega}. "
                f"Valid range: Ω > 0"
            )


@dataclass
class ProcessingResult:
    """Result of image sharpening operation."""
    sharpened_image: np.ndarray
    plcc: float
    srocc: float
    parameters: ProcessingConfig
    processing_time: float
    intermediate_images: Optional[Dict[str, np.ndarray]] = None


@dataclass
class TestResult:
    """Result of parameter testing."""
    t: float
    lambda_param: float
    sigma: float
    sharpened_image: np.ndarray
    plcc: float
    srocc: float
