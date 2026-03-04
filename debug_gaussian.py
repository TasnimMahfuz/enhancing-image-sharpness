import numpy as np
from src.gaussian_filter import GaussianFilter

# Test with sigma = 0.5
sigma = 0.5
gf = GaussianFilter(sigma)
kernel = gf.kernel

radius = int(np.ceil(3 * sigma))
center_idx = radius

print(f"Sigma: {sigma}")
print(f"Radius: {radius}")
print(f"Kernel shape: {kernel.shape}")
print(f"Kernel sum: {np.sum(kernel)}")
print(f"Center value in kernel: {kernel[center_idx, center_idx]}")

# Calculate what the center SHOULD be before normalization
expected_center_unnormalized = (1.0 / (2 * np.pi * sigma**2)) * np.exp(0)
print(f"Expected center (unnormalized): {expected_center_unnormalized}")

# Calculate the sum of the unnormalized kernel
x = np.arange(-radius, radius + 1)
y = np.arange(-radius, radius + 1)
X, Y = np.meshgrid(x, y)
coefficient = 1.0 / (2 * np.pi * sigma**2)
exponent = -(X**2 + Y**2) / (2 * sigma**2)
kernel_unnormalized = coefficient * np.exp(exponent)
unnormalized_sum = np.sum(kernel_unnormalized)
print(f"Unnormalized kernel sum: {unnormalized_sum}")

# The normalized center should be
expected_center_normalized = expected_center_unnormalized / unnormalized_sum
print(f"Expected center (normalized): {expected_center_normalized}")
print(f"Difference: {abs(kernel[center_idx, center_idx] - expected_center_normalized)}")
