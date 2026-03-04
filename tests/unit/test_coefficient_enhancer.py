"""
Unit tests for CoefficientEnhancer class.

Tests mathematical correctness of coefficient bound calculations,
edge cases, and specific examples from the design document.
"""

import pytest
import numpy as np
from src.coefficient_enhancer import CoefficientEnhancer


class TestCoefficientEnhancerInitialization:
    """Test CoefficientEnhancer initialization and validation."""
    
    def test_valid_parameters(self):
        """Test initialization with valid parameters."""
        enhancer = CoefficientEnhancer(t=0.6, lambda_param=0.0, omega=1.0)
        assert enhancer.t == 0.6
        assert enhancer.lambda_param == 0.0
        assert enhancer.omega == 1.0
    
    def test_t_below_range_raises_error(self):
        """Test that t < 0 raises ValueError."""
        with pytest.raises(ValueError, match="Parameter t must be in"):
            CoefficientEnhancer(t=-0.1, lambda_param=0.0)
    
    def test_t_above_range_raises_error(self):
        """Test that t > 1 raises ValueError."""
        with pytest.raises(ValueError, match="Parameter t must be in"):
            CoefficientEnhancer(t=1.5, lambda_param=0.0)
    
    def test_t_equals_one_raises_error(self):
        """Test that t = 1 raises ValueError (division by zero)."""
        with pytest.raises(ValueError, match="too close to 1"):
            CoefficientEnhancer(t=1.0, lambda_param=0.0)
    
    def test_t_very_close_to_one_raises_error(self):
        """Test that t ≈ 1 raises ValueError."""
        with pytest.raises(ValueError, match="too close to 1"):
            CoefficientEnhancer(t=0.99999999999, lambda_param=0.0)


class TestUnCalculation:
    """Test u_n calculation with various inputs."""
    
    def test_t_zero_edge_case(self):
        """
        Feature: image-sharpness-enhancement
        Edge Case 1: t=0 Handling
        
        When t=0, u_n should equal Ω^n (limit formula).
        Validates: Requirements 4.4
        """
        enhancer = CoefficientEnhancer(t=0.0, lambda_param=1.0, omega=2.0)
        
        # For t=0, u₁ should equal Ω^1 = 2.0
        u1 = enhancer._calculate_u_n(1)
        assert abs(u1 - 2.0) < 1e-10, f"Expected 2.0, got {u1}"
        
        # For t=0, u₂ should equal Ω^2 = 4.0
        u2 = enhancer._calculate_u_n(2)
        assert abs(u2 - 4.0) < 1e-10, f"Expected 4.0, got {u2}"
        
        # For t=0, u₃ should equal Ω^3 = 8.0
        u3 = enhancer._calculate_u_n(3)
        assert abs(u3 - 8.0) < 1e-10, f"Expected 8.0, got {u3}"
    
    def test_u_n_formula_correctness(self):
        """
        Feature: image-sharpness-enhancement
        Example 3: u_n Formula Correctness
        
        Verify u_n for known values using general formula.
        Validates: Requirements 4.3
        """
        enhancer = CoefficientEnhancer(t=0.5, lambda_param=0.0, omega=1.0)
        
        # For t=0.5, Ω=1, n=1: u₁ = (1 - 0.5) / (1 - 0.5) = 1.0
        u1 = enhancer._calculate_u_n(1)
        assert abs(u1 - 1.0) < 1e-10, f"Expected 1.0, got {u1}"
        
        # For t=0.5, Ω=1, n=2: u₂ = (1 - 0.25) / (1 - 0.5) = 0.75 / 0.5 = 1.5
        u2 = enhancer._calculate_u_n(2)
        assert abs(u2 - 1.5) < 1e-10, f"Expected 1.5, got {u2}"
    
    def test_u_n_with_omega_not_one(self):
        """Test u_n calculation with Ω ≠ 1."""
        enhancer = CoefficientEnhancer(t=0.3, lambda_param=0.0, omega=2.0)
        
        # For t=0.3, Ω=2, n=1: u₁ = (2 - 0.3) / (1 - 0.3) = 1.7 / 0.7 ≈ 2.4286
        u1 = enhancer._calculate_u_n(1)
        expected_u1 = (2.0 - 0.3) / (1.0 - 0.3)
        assert abs(u1 - expected_u1) < 1e-10, f"Expected {expected_u1}, got {u1}"


class TestA2Calculation:
    """Test |a₂| calculation from Equation 32."""
    
    def test_equation_32_correctness(self):
        """
        Feature: image-sharpness-enhancement
        Example 1: Equation 32 Correctness
        
        Verify |a₂| calculation for known values.
        Validates: Requirements 4.1
        """
        # Known values: t=0.6, λ=0, Ω=1
        enhancer = CoefficientEnhancer(t=0.6, lambda_param=0.0, omega=1.0)
        
        # Hand-computed: u₁ = (1 - 0.6)/(1 - 0.6) = 1.0
        # |a₂| = (2 * 0 / (1 + 1)) * 1.0 = 0
        expected_a2 = 0.0
        
        assert abs(enhancer.a2_bound - expected_a2) < 1e-10, \
            f"Expected {expected_a2}, got {enhancer.a2_bound}"
    
    def test_a2_with_nonzero_lambda(self):
        """Test |a₂| calculation with λ ≠ 0."""
        # t=0.5, λ=1.0, Ω=1.0
        enhancer = CoefficientEnhancer(t=0.5, lambda_param=1.0, omega=1.0)
        
        # u₁ = (1 - 0.5) / (1 - 0.5) = 1.0
        # |a₂| = |(2 * 1.0 / (1 + 1)) * 1.0| = |1.0| = 1.0
        expected_a2 = 1.0
        
        assert abs(enhancer.a2_bound - expected_a2) < 1e-10, \
            f"Expected {expected_a2}, got {enhancer.a2_bound}"
    
    def test_a2_with_negative_lambda(self):
        """Test |a₂| calculation with negative λ."""
        # t=0.5, λ=-2.0, Ω=1.0
        enhancer = CoefficientEnhancer(t=0.5, lambda_param=-2.0, omega=1.0)
        
        # u₁ = 1.0
        # |a₂| = |(2 * -2.0 / 2) * 1.0| = |-2.0| = 2.0
        expected_a2 = 2.0
        
        assert abs(enhancer.a2_bound - expected_a2) < 1e-10, \
            f"Expected {expected_a2}, got {enhancer.a2_bound}"


class TestA3Calculation:
    """Test |a₃| calculation from Equation 34."""
    
    def test_equation_34_correctness(self):
        """
        Feature: image-sharpness-enhancement
        Example 2: Equation 34 Correctness
        
        Verify |a₃| calculation for known values.
        Validates: Requirements 4.2
        """
        # Known values: t=0.6, λ=0, Ω=1
        enhancer = CoefficientEnhancer(t=0.6, lambda_param=0.0, omega=1.0)
        
        # Hand-computed: u₁ = 1.0, u₂ = (1 - 0.36)/(1 - 0.6) = 0.64/0.4 = 1.6
        # |a₃| = |(0 / 2) * [1.0 + (0 / 2) * 1.6]| = 0
        expected_a3 = 0.0
        
        assert abs(enhancer.a3_bound - expected_a3) < 1e-10, \
            f"Expected {expected_a3}, got {enhancer.a3_bound}"
    
    def test_a3_with_nonzero_lambda(self):
        """Test |a₃| calculation with λ ≠ 0."""
        # t=0.5, λ=1.0, Ω=1.0
        enhancer = CoefficientEnhancer(t=0.5, lambda_param=1.0, omega=1.0)
        
        # u₁ = 1.0, u₂ = 1.5
        # term1 = 1.0 / 2 = 0.5
        # term2 = 1.0 + 0.5 * 1.5 = 1.0 + 0.75 = 1.75
        # |a₃| = |0.5 * 1.75| = 0.875
        expected_a3 = 0.875
        
        assert abs(enhancer.a3_bound - expected_a3) < 1e-10, \
            f"Expected {expected_a3}, got {enhancer.a3_bound}"


class TestEnhanceMethod:
    """Test edge enhancement functionality."""
    
    def test_enhance_preserves_shape(self):
        """Test that enhance preserves image dimensions."""
        enhancer = CoefficientEnhancer(t=0.6, lambda_param=1.0, omega=1.0)
        
        # Create test edge image
        edge_image = np.random.randn(100, 100).astype(np.float32)
        
        enhanced = enhancer.enhance(edge_image)
        
        assert enhanced.shape == edge_image.shape, \
            f"Expected shape {edge_image.shape}, got {enhanced.shape}"
    
    def test_enhance_scales_by_a2_bound(self):
        """Test that enhance scales edge image by a2_bound."""
        enhancer = CoefficientEnhancer(t=0.5, lambda_param=1.0, omega=1.0)
        
        # Create simple edge image
        edge_image = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        
        enhanced = enhancer.enhance(edge_image)
        
        # Should be scaled by a2_bound = 1.0
        expected = edge_image * enhancer.a2_bound
        
        np.testing.assert_allclose(enhanced, expected, rtol=1e-10)
    
    def test_enhance_preserves_sign(self):
        """Test that enhance preserves sign of edge values."""
        enhancer = CoefficientEnhancer(t=0.5, lambda_param=2.0, omega=1.0)
        
        # Create edge image with positive and negative values
        edge_image = np.array([[-5.0, -2.0], [3.0, 7.0]], dtype=np.float32)
        
        enhanced = enhancer.enhance(edge_image)
        
        # Check signs are preserved
        assert np.all((edge_image > 0) == (enhanced > 0)), \
            "Positive values should remain positive"
        assert np.all((edge_image < 0) == (enhanced < 0)), \
            "Negative values should remain negative"
    
    def test_enhance_with_color_image(self):
        """Test enhance with multi-channel edge image."""
        enhancer = CoefficientEnhancer(t=0.5, lambda_param=1.0, omega=1.0)
        
        # Create 3-channel edge image
        edge_image = np.random.randn(50, 50, 3).astype(np.float32)
        
        enhanced = enhancer.enhance(edge_image)
        
        assert enhanced.shape == edge_image.shape, \
            f"Expected shape {edge_image.shape}, got {enhanced.shape}"
        
        # Verify scaling is applied uniformly
        expected = edge_image * enhancer.a2_bound
        np.testing.assert_allclose(enhanced, expected, rtol=1e-10)
