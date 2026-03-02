"""
Property-based tests for CoefficientEnhancer class.

Tests universal properties that should hold across all valid inputs,
using hypothesis for automated test case generation.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings
from src.coefficient_enhancer import CoefficientEnhancer


class TestCoefficientParameterValidation:
    """Property tests for parameter validation."""
    
    @given(t=st.floats(min_value=0.0, max_value=1.0).filter(lambda x: abs(1 - x) > 1e-9))
    @settings(max_examples=100)
    def test_property_12_accepts_valid_t(self, t):
        """
        Feature: image-sharpness-enhancement
        Property 12: Coefficient t Parameter Validation (acceptance part)
        
        For any t ∈ [0, 1] (excluding t ≈ 1), CoefficientEnhancer should accept it.
        Validates: Requirements 4.5
        """
        # Should not raise an exception
        enhancer = CoefficientEnhancer(t=t, lambda_param=0.0, omega=1.0)
        assert enhancer.t == t
    
    @given(t=st.one_of(
        st.floats(min_value=-1000.0, max_value=-0.001),
        st.floats(min_value=1.001, max_value=1000.0)
    ))
    @settings(max_examples=100)
    def test_property_12_rejects_invalid_t(self, t):
        """
        Feature: image-sharpness-enhancement
        Property 12: Coefficient t Parameter Validation (rejection part)
        
        For any t ∉ [0, 1], CoefficientEnhancer should reject it.
        Validates: Requirements 4.5
        """
        with pytest.raises(ValueError, match="Parameter t must be in"):
            CoefficientEnhancer(t=t, lambda_param=0.0, omega=1.0)
    
    def test_property_12_rejects_t_near_one(self):
        """
        Feature: image-sharpness-enhancement
        Property 12: Coefficient t Parameter Validation (t ≈ 1 rejection)
        
        CoefficientEnhancer should reject t ≈ 1 (causes division by zero).
        Validates: Requirements 4.5
        """
        # Test exact t=1.0
        with pytest.raises(ValueError, match="too close to 1"):
            CoefficientEnhancer(t=1.0, lambda_param=0.0, omega=1.0)
        
        # Test t very close to 1 (within epsilon = 1e-10)
        with pytest.raises(ValueError, match="too close to 1"):
            CoefficientEnhancer(t=1.0 - 1e-11, lambda_param=0.0, omega=1.0)
    
    @given(lambda_param=st.floats(
        min_value=-100.0,
        max_value=100.0,
        allow_nan=False,
        allow_infinity=False
    ))
    @settings(max_examples=100)
    def test_property_13_accepts_any_real_lambda(self, lambda_param):
        """
        Feature: image-sharpness-enhancement
        Property 13: Coefficient Lambda Parameter Acceptance
        
        For any real number λ (including negative, zero, and positive),
        CoefficientEnhancer should accept it.
        Validates: Requirements 4.6
        """
        # Should not raise an exception for any real lambda
        enhancer = CoefficientEnhancer(t=0.5, lambda_param=lambda_param, omega=1.0)
        assert enhancer.lambda_param == lambda_param


class TestCoefficientCalculationProperties:
    """Property tests for coefficient calculations."""
    
    @given(
        t=st.floats(min_value=0.0, max_value=0.99).filter(lambda x: abs(1 - x) > 1e-9),
        lambda_param=st.floats(min_value=-10.0, max_value=10.0),
        omega=st.floats(min_value=0.1, max_value=5.0)
    )
    @settings(max_examples=100)
    def test_a2_bound_is_non_negative(self, t, lambda_param, omega):
        """
        Property: |a₂| bound should always be non-negative.
        
        Since we take absolute value, a2_bound ≥ 0 for all inputs.
        """
        enhancer = CoefficientEnhancer(t=t, lambda_param=lambda_param, omega=omega)
        assert enhancer.a2_bound >= 0, f"a2_bound should be non-negative, got {enhancer.a2_bound}"
    
    @given(
        t=st.floats(min_value=0.0, max_value=0.99).filter(lambda x: abs(1 - x) > 1e-9),
        lambda_param=st.floats(min_value=-10.0, max_value=10.0),
        omega=st.floats(min_value=0.1, max_value=5.0)
    )
    @settings(max_examples=100)
    def test_a3_bound_is_non_negative(self, t, lambda_param, omega):
        """
        Property: |a₃| bound should always be non-negative.
        
        Since we take absolute value, a3_bound ≥ 0 for all inputs.
        """
        enhancer = CoefficientEnhancer(t=t, lambda_param=lambda_param, omega=omega)
        assert enhancer.a3_bound >= 0, f"a3_bound should be non-negative, got {enhancer.a3_bound}"
    
    @given(
        t=st.floats(min_value=0.0, max_value=0.99).filter(lambda x: abs(1 - x) > 1e-9),
        omega=st.floats(min_value=0.1, max_value=5.0)
    )
    @settings(max_examples=100)
    def test_zero_lambda_gives_zero_bounds(self, t, omega):
        """
        Property: When λ = 0, both |a₂| and |a₃| should equal 0.
        
        From the formulas, both bounds are proportional to λ.
        """
        enhancer = CoefficientEnhancer(t=t, lambda_param=0.0, omega=omega)
        assert abs(enhancer.a2_bound) < 1e-10, f"Expected a2_bound ≈ 0, got {enhancer.a2_bound}"
        assert abs(enhancer.a3_bound) < 1e-10, f"Expected a3_bound ≈ 0, got {enhancer.a3_bound}"
    
    @given(
        n=st.integers(min_value=1, max_value=10),
        omega=st.floats(min_value=0.1, max_value=3.0)
    )
    @settings(max_examples=100)
    def test_u_n_at_t_zero_equals_omega_power_n(self, n, omega):
        """
        Property: When t = 0, u_n should equal Ω^n.
        
        This tests the limit formula implementation.
        """
        enhancer = CoefficientEnhancer(t=0.0, lambda_param=0.0, omega=omega)
        u_n = enhancer._calculate_u_n(n)
        expected = omega ** n
        assert abs(u_n - expected) < 1e-9, f"Expected {expected}, got {u_n}"
    
    @given(
        t=st.floats(min_value=0.01, max_value=0.99).filter(lambda x: abs(1 - x) > 1e-9),
        n=st.integers(min_value=1, max_value=10),
        omega=st.floats(min_value=0.1, max_value=3.0)
    )
    @settings(max_examples=100)
    def test_u_n_formula_consistency(self, t, n, omega):
        """
        Property: u_n should equal (Ω^n - t^n) / (1 - t) for t ≠ 0.
        
        Verifies the general formula is correctly implemented.
        """
        enhancer = CoefficientEnhancer(t=t, lambda_param=0.0, omega=omega)
        u_n = enhancer._calculate_u_n(n)
        expected = (omega ** n - t ** n) / (1 - t)
        assert abs(u_n - expected) < 1e-9, f"Expected {expected}, got {u_n}"


class TestEnhanceProperties:
    """Property tests for edge enhancement."""
    
    @given(
        t=st.floats(min_value=0.0, max_value=0.99).filter(lambda x: abs(1 - x) > 1e-9),
        lambda_param=st.floats(min_value=-10.0, max_value=10.0),
        height=st.integers(min_value=10, max_value=100),
        width=st.integers(min_value=10, max_value=100)
    )
    @settings(max_examples=50)
    def test_enhance_preserves_dimensions(self, t, lambda_param, height, width):
        """
        Feature: image-sharpness-enhancement
        Property 15: Enhancement Preserves Dimensions
        
        For any edge image, the enhanced edge image should have the same dimensions.
        Validates: Requirements 5.3
        """
        enhancer = CoefficientEnhancer(t=t, lambda_param=lambda_param, omega=1.0)
        edge_image = np.random.randn(height, width).astype(np.float32)
        
        enhanced = enhancer.enhance(edge_image)
        
        assert enhanced.shape == edge_image.shape, \
            f"Expected shape {edge_image.shape}, got {enhanced.shape}"
    
    @given(
        t=st.floats(min_value=0.0, max_value=0.99).filter(lambda x: abs(1 - x) > 1e-9),
        lambda_param=st.floats(min_value=0.1, max_value=10.0),  # Positive lambda
        height=st.integers(min_value=10, max_value=50),
        width=st.integers(min_value=10, max_value=50)
    )
    @settings(max_examples=50)
    def test_enhance_preserves_sign_with_positive_coefficient(self, t, lambda_param, height, width):
        """
        Feature: image-sharpness-enhancement
        Property 16: Enhancement Preserves Sign
        
        For any edge image and positive coefficient, if a pixel value is positive 
        (or negative) in the edge image, it should remain positive (or negative) 
        in the enhanced edge image.
        Validates: Requirements 5.4
        """
        enhancer = CoefficientEnhancer(t=t, lambda_param=lambda_param, omega=1.0)
        
        # Create edge image with both positive and negative values
        edge_image = np.random.randn(height, width).astype(np.float32)
        
        enhanced = enhancer.enhance(edge_image)
        
        # Check sign preservation (excluding near-zero values)
        tolerance = 1e-6
        positive_mask = edge_image > tolerance
        negative_mask = edge_image < -tolerance
        
        if np.any(positive_mask):
            assert np.all(enhanced[positive_mask] > 0), \
                "Positive values should remain positive"
        
        if np.any(negative_mask):
            assert np.all(enhanced[negative_mask] < 0), \
                "Negative values should remain negative"
    
    @given(
        t=st.floats(min_value=0.0, max_value=0.99).filter(lambda x: abs(1 - x) > 1e-9),
        lambda_param=st.floats(min_value=-10.0, max_value=10.0),
        height=st.integers(min_value=10, max_value=50),
        width=st.integers(min_value=10, max_value=50)
    )
    @settings(max_examples=50)
    def test_enhance_is_uniform_scaling(self, t, lambda_param, height, width):
        """
        Feature: image-sharpness-enhancement
        Property 14: Edge Enhancement is Uniform Scaling
        
        For any edge image and coefficient bound, the enhanced edge image should 
        equal the edge image multiplied by the coefficient at every pixel.
        Validates: Requirements 5.1, 5.2
        """
        enhancer = CoefficientEnhancer(t=t, lambda_param=lambda_param, omega=1.0)
        edge_image = np.random.randn(height, width).astype(np.float32)
        
        enhanced = enhancer.enhance(edge_image)
        expected = edge_image * enhancer.a2_bound
        
        np.testing.assert_allclose(enhanced, expected, rtol=1e-9, atol=1e-9)
