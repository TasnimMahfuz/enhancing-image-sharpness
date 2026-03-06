"""
Coefficient Enhancer Module

Implements coefficient bound calculations from Equations 32 and 34 of the paper.
Calculates |a₂| and |a₃| bounds using analytic function theory and applies them
to enhance edge contrast in images.
"""

import numpy as np
from src.exceptions import ValidationError, NumericalError


class CoefficientEnhancer:
    """
    Calculate and apply coefficient bounds for edge enhancement.
    
    This class implements the mathematical formulas from the paper:
    - Equation 32: |a₂| ≤ √[3 / |2{3u₂² - 3u₃ - 6u₂(1+λ) + 9(1+2λ) + 7[2(1+λ)-u₂]²}|]
    - Equation 34: |a₃| ≤ 1/(4[2(1+λ)-u₂]) + 1/(2[3(1+2λ)-u₃])
    
    Where u_n = (Ω^n - t^n) / (1 - t), with special handling for t=0.
    """
    
    def __init__(self, t: float, lambda_param: float, omega: float = 1.0):
        """
        Initialize coefficient enhancer.
        
        Args:
            t: Parameter t ∈ [0, 1]
            lambda_param: Parameter λ (real number)
            omega: Parameter Ω (default 1.0 as per paper)
            
        Raises:
            ValueError: If t not in [0, 1] or if t is too close to 1
        """
        # Validate t parameter
        if not 0 <= t <= 1:
            raise ValidationError(
                f"Parameter t must be in [0, 1], got {t}. "
                f"Valid range: 0 ≤ t ≤ 1"
            )
        
        # Check for t ≈ 1 (causes division by zero)
        EPSILON = 1e-10
        if abs(1 - t) < EPSILON:
            raise ValidationError(
                f"Parameter t too close to 1 (t={t}). "
                f"This causes division by zero in u_n calculation."
            )
        
        self.t = t
        self.lambda_param = lambda_param
        self.omega = omega
        
        # Calculate coefficient bounds
        self.a2_bound = self._calculate_a2()
        self.a3_bound = self._calculate_a3()
    
    def _calculate_u_n(self, n: int) -> float:
        """
        Calculate u_n term with special handling for t=0.
        
        Formula:
        - General case: u_n = (Ω^n - t^n) / (1 - t)
        - Special case (t ≈ 0): u_n = Ω^n (limit as t → 0)
        
        Args:
            n: Power term
            
        Returns:
            u_n value
        """
        EPSILON = 1e-10
        
        # Handle t ≈ 0 using limit formula
        if abs(self.t) < EPSILON:
            return self.omega ** n
        
        # General formula
        numerator = self.omega ** n - self.t ** n
        denominator = 1 - self.t
        result = numerator / denominator
        
        # Check for numerical issues
        if not np.isfinite(result):
            raise NumericalError(
                f"Non-finite value in u_n calculation: {result}. "
                f"Parameters: n={n}, t={self.t}, omega={self.omega}"
            )
        
        return result
    
    def _calculate_a2(self) -> float:
        """
        Calculate |a₂| bound from Equation 32.
        
        Formula: |a₂| ≤ √[3 / |2{3u₂² - 3u₃ - 6u₂(1+λ) + 9(1+2λ) + 7[2(1+λ)-u₂]²}|]
        
        Returns:
            |a₂| bound value
        """
        u2 = self._calculate_u_n(2)
        u3 = self._calculate_u_n(3)
        
        denominator = 2 * (3*u2**2 - 3*u3 - 6*u2*(1 + self.lambda_param) + 
                          9*(1 + 2*self.lambda_param) + 
                          7*(2*(1 + self.lambda_param) - u2)**2)
        
        if abs(denominator) < 1e-10:
            return 0.0
        
        a2 = np.sqrt(3 / abs(denominator))
        return abs(a2)
    
    def _calculate_a3(self) -> float:
        """
        Calculate |a₃| bound from Equation 34.
        
        Formula: |a₃| ≤ 1/(4[2(1+λ)-u₂]) + 1/(2[3(1+2λ)-u₃])
        
        Returns:
            |a₃| bound value
        """
        u2 = self._calculate_u_n(2)
        u3 = self._calculate_u_n(3)
        
        term1_denom = 4 * (2*(1 + self.lambda_param) - u2)
        term2_denom = 2 * (3*(1 + 2*self.lambda_param) - u3)
        
        term1 = 1 / term1_denom if abs(term1_denom) > 1e-10 else 0.0
        term2 = 1 / term2_denom if abs(term2_denom) > 1e-10 else 0.0
        
        a3 = term1 + term2
        return abs(a3)
    
    def enhance(self, edge_image: np.ndarray) -> np.ndarray:
        """
        Apply coefficient bounds to enhance edge image.
        
        Enhancement strategy: Use both |a₂| and |a₃| coefficient bounds.
        Scales the edge image by the combined coefficient bounds as per
        the paper's methodology (Equations 32 and 34).
        
        Args:
            edge_image: Edge information from EdgeExtractor
            
        Returns:
            Enhanced edge image with same shape as input
            
        Raises:
            ValidationError: If edge_image is not a valid numpy array
            NumericalError: If enhancement produces non-finite values
        """
        # Validate input
        if not isinstance(edge_image, np.ndarray):
            raise ValidationError(
                f"edge_image must be a numpy array, got {type(edge_image)}"
            )
        
        if edge_image.size == 0:
            raise ValidationError(
                "edge_image is empty (size 0)"
            )
        
        # Use scaled average of both coefficient bounds 
        enhanced_edge = edge_image * ((self.a2_bound + self.a3_bound) / 2) * 0.1


        
        # Check for numerical issues
        if not np.all(np.isfinite(enhanced_edge)):
            nan_count = np.sum(np.isnan(enhanced_edge))
            inf_count = np.sum(np.isinf(enhanced_edge))
            raise NumericalError(
                f"Enhancement produced non-finite values. "
                f"NaN count: {nan_count}, Inf count: {inf_count}. "
                f"a2_bound: {self.a2_bound}, a3_bound: {self.a3_bound}, "
                f"edge_image range: [{np.min(edge_image)}, {np.max(edge_image)}]"
            )
        
        return enhanced_edge
